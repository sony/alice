from .imports import *


@fig.component('intervention/label')
class ClassLevelLabelIntervention(Machine):
    @staticmethod
    def generate_intervention_device(sel: torch.Tensor) -> torch.Tensor:
        """
        (B, N) mask of allowed values for each sample -> (B,) selected indices

        uniformly sample one of the available options (val == True) for each sample in the batch. 
        Returns (for each sample) index of the selected option or -1 if no option is available.
        """
        B, N = sel.shape
        row, col = torch.as_tensor(np.mgrid[:B,:N], device=sel.device)
        row = row[sel]
        col = col[sel]
        cnt = torch.bincount(row.flatten(), minlength=B)
        pick_idx = torch.rand(B, device=sel.device).mul(cnt).floor().int()
        offset = cnt.cumsum(0)
        idx = torch.empty_like(cnt)
        idx[0] = pick_idx[0]
        idx[1:] = offset[:-1] + pick_idx[1:]
        intv = col[idx]
        intv[cnt == 0] = -1 # invalid interventions
        return intv
    

    @staticmethod
    def generate_intervention(sel: torch.Tensor) -> torch.Tensor:
        """
        (B, N) mask of allowed values for each sample -> (B,) selected indices

        uniformly sample one of the available options (val == True) for each sample in the batch. 
        Returns (for each sample) index of the selected option or -1 if no option is available.
        """
        B, N = sel.shape
        row, col = torch.as_tensor(np.mgrid[:B,:N], device=sel.device)
        row = row[sel]
        col = col[sel]
        cnt = torch.bincount(row.flatten(), minlength=B)
        pick_idx = torch.rand(B, device=sel.device).mul(cnt).floor().int()
        offset = cnt.cumsum(0)
        idx = torch.empty_like(cnt)
        idx[0] = pick_idx[0]
        idx[1:] = offset[:-1] + pick_idx[1:]
        intv = col[idx.clamp(max=len(col)-1)] # need to clamp due to trailing invalid options
        intv[cnt == 0] = -1 # invalid interventions
        return intv
    
    label_space = gear('label')

    @tool('add_class')
    def sample_add_intervention(self, label: torch.Tensor):
        if label.dim() == 1 or label.shape[1] == 1:
            label = F.one_hot(label, num_classes=self.label_space.n)
        intv = self.generate_intervention(~label.bool())
        return intv
    

    @tool('remove_class')
    def sample_remove_intervention(self, label: torch.Tensor):
        if label.dim() == 1 or label.shape[1] == 1:
            label = F.one_hot(label, num_classes=self.label_space.n)
        intv = self.generate_intervention(label.bool())
        return intv
    

    @tool('intervention')
    def apply_intervention(self, label: torch.Tensor, 
                           add_class: torch.Tensor = None, 
                           remove_class: torch.Tensor = None):
        _single = False
        if label.dim() == 1 or label.shape[1] == 1:
            _single = True
            label = F.one_hot(label, num_classes=self.label_space.n)
        intv = label.clone()
        idx = torch.arange(label.shape[0], device=label.device)

        if add_class is not None:
            add_sel = add_class >= 0
            intv[idx[add_sel], add_class[add_sel]] = 1
        
        if remove_class is not None:
            remove_sel = remove_class >= 0
            intv[idx[remove_sel], remove_class[remove_sel]] = 0

        if _single:
            intv = intv.argmax(dim=1)
        return intv



@fig.component('intervention/centroid')
class ClassLevelCentroidIntervention(Model):
    label_space = gear('label')

    def __init__(self, *, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._normalize = normalize
        self.centroids = None

    def _prepare(self, *, device = None):
        self.centroids = nn.Parameter(self._init_centroids(self.probe_space, self.label_space, device=device), 
                                      requires_grad=True)
        return super()._prepare(device=device)
    
    def checkpoint(self, path = None):
        if path is None:
            return self.centroids.detach().cpu().numpy()
        torch.save(self.centroids, path)
        return self

    def load_checkpoint(self, *, path = None, data = None, unsafe = False):
        # device = self.centroids.device if self.centroids is not None else None
        if data is None:
            if not path.exists() and path.suffix == '':
                path = path.with_suffix('.pt')
            assert path.exists(), f'Path {path} does not exist'
            data = torch.load(path)
            # if device is not None:
            #     data = data.to(device)
        self.centroids = nn.Parameter(data, requires_grad=True)
        return self

    def _init_centroids(self, latent_space: spaces.Vector, label_space: spaces.Categorical, device = None):
        N, D = label_space.n, latent_space.size
        return torch.randn(N, D, device=device)


    @tool('probe')
    def apply_intervention(self, ambient: torch.Tensor, add_class: torch.Tensor = None, remove_class: torch.Tensor = None):
        assert add_class is not None or remove_class is not None

        deltas = torch.zeros_like(ambient)

        if add_class is not None:
            sel = add_class >= 0
            deltas[sel] += self.centroids[add_class[sel]]

        if remove_class is not None:
            sel = remove_class >= 0
            deltas[sel] -= self.centroids[remove_class[sel]]

        probe = ambient + deltas
        if self._normalize:
            return F.normalize(probe, p=2, dim=-1)
        return probe
    @apply_intervention.space
    def probe_space(self, ambient: spaces.Vector) -> spaces.Vector:
        return ambient # TODO: copy



@fig.component('intervention/linear')
class ClassLevelLinearIntervention(Model):
    label_space = gear('label')

    def __init__(self, *, normalize: bool = False, affine: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._normalize = normalize
        self._include_offsets = affine
        self.transformations = None
        self.offsets = None

    def _prepare(self, *, device = None):
        self.transformations = nn.Parameter(self._init_transformations(self.probe_space, self.label_space, device=device), 
                                      requires_grad=True)
        if self._include_offsets:
            self.offsets = nn.Parameter(self._init_offsets(self.probe_space, self.label_space, device=device), 
                                      requires_grad=True)
        return super()._prepare(device=device)
    
    def checkpoint(self, path = None):
        data = {'transformations': self.transformations.detach().cpu().numpy()}
        if self._include_offsets:
            data['offsets'] = self.offsets.detach().cpu().numpy()
        if path is None:
            return data
        torch.save(data, path)
        return self
    
    def load_checkpoint(self, *, path = None, data = None, unsafe = False):
        # device = self.transformations.device if self.transformations is not None else None
        if data is None:
            if not path.exists() and path.suffix == '':
                path = path.with_suffix('.pt')
            assert path.exists(), f'Path {path} does not exist'
            data = torch.load(path)
        self.transformations = nn.Parameter(torch.tensor(data['transformations']), requires_grad=True)
        if 'offsets' in data:
            self.offsets = nn.Parameter(torch.tensor(data['offsets']), requires_grad=True)
        # if device is not None:
        #     self.transformations = self.transformations.to(device)
        #     if self.offsets is not None:
        #         self.offsets = self.offsets.to(device)
        return


    def _init_transformations(self, latent_space: spaces.Vector, label_space: spaces.Categorical, device = None):
        N, D = label_space.n, latent_space.size
        return torch.randn(N, D, D, device=device)

    def _init_offsets(self, latent_space: spaces.Vector, label_space: spaces.Categorical, device = None):
        N, D = label_space.n, latent_space.size
        return torch.randn(N, D, device=device)


    @tool('probe')
    def apply_intervention(self, ambient: torch.Tensor, add_class: torch.Tensor = None, remove_class: torch.Tensor = None):
        assert add_class is not None or remove_class is not None

        deltas = torch.zeros_like(ambient)

        if add_class is not None:
            sel = add_class >= 0
            tfm = self.transformations[add_class[sel]]
            deltas[sel] += (tfm @ ambient[sel].unsqueeze(-1)).squeeze(-1)
            if self._include_offsets:
                deltas[sel] += self.offsets[add_class[sel]]

        if remove_class is not None:
            sel = remove_class >= 0
            tfm = self.transformations[remove_class[sel]]
            deltas[sel] -= (tfm @ ambient[sel].unsqueeze(-1)).squeeze(-1)
            if self._include_offsets:
                deltas[sel] -= self.offsets[remove_class[sel]]

        probe = ambient + deltas
        if self._normalize:
            return F.normalize(probe, p=2, dim=-1)
        return probe
    @apply_intervention.space
    def probe_space(self, ambient: spaces.Vector) -> spaces.Vector:
        return ambient # TODO: copy



@fig.component('intervention/module')
class ClassLevelModuleIntervention(Model):
    def __init__(self, module: Model, condition: Union[str, Path, int], *, 
                 load: str = None, regularize_delta: bool = True,
                 residual: bool = True, suppress=None, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.module = module
        self._load_path = Path(load) if load else None
        self._condition = condition
        self._residual = residual
        self._suppress = suppress
        self._normalize = normalize
        self._regularize_delta = regularize_delta

        self._add_codes = None # N x D
        self._remove_codes = None # N x D
        self._replace_codes = None # N x N x D
        self._redundancy = None

    label_space: spaces.Categorical = gear('label')


    def settings(self):
        return {'module': self.module.settings(), 
                'condition': self._condition, 
                'load': self._load_path,
                'regularize_delta': self._regularize_delta,
                'residual': self._residual}

    def checkpoint(self, path = None):
        data = {
            'module': self.module.checkpoint(),
        }
        if isinstance(self._condition, int):
            data.update({
                'add_embeddings': self._add_codes.detach().cpu().numpy(), 
                'remove_embeddings': self._remove_codes.detach().cpu().numpy(), 
                'replace_embeddings': self._replace_codes.detach().cpu().numpy()
            })
        else:
            data['condition'] = str(self._condition)
        if path is None:
            return data
        if path.suffix == '': path = path.with_suffix('.pt')
        torch.save(data, path)
        return self
    
    def load_checkpoint(self, *, path = None, data = None, unsafe = False):
        if data is None:
            if not path.exists() and path.suffix == '':
                path = path.with_suffix('.pt')
            assert path.exists(), f'Path {path} does not exist'
            data = torch.load(path)
        self.module.load_checkpoint(data=data['module'])
        if isinstance(self._condition, int):
            self._add_codes = torch.tensor(data['add_embeddings'])
            self._remove_codes = torch.tensor(data['remove_embeddings'])
            self._replace_codes = torch.tensor(data['replace_embeddings'])
        else:
            self._condition = data['condition']
        return self

    def _prepare(self, *, device = None):

        if isinstance(self._condition, int):
            N = self.label_space.n
            D = self.conditioning_space.size
            self._add_codes = nn.Parameter(torch.randn(N, D, device=device), requires_grad=True)
            self._remove_codes = nn.Parameter(torch.randn(N, D, device=device), requires_grad=True)
            self._replace_codes = nn.Parameter(torch.randn(N, N, D, device=device), requires_grad=True)
        elif isinstance(self._condition, (str, Path)):
            path = Path(self._condition)
            if not path.exists():
                raise FileNotFoundError(f'Path {path} does not exist')
            with hf.File(path, 'r') as f:
                self._add_codes = torch.tensor(f['add_embeddings'][:], device=device)
                self._remove_codes = torch.tensor(f['remove_embeddings'][:], device=device)
                self._replace_codes = torch.tensor(f['replace_embeddings'][:], device=device)
            N, *other = self._add_codes.shape
            assert self.label_space.n == N, f'Number of classes mismatch: {self.label_space.n} != {N}'
            if len(other) > 1:
                assert len(other) == 2, f'Invalid shape: {self._add_codes.shape}'
                self._redundancy = other[0]
        else:
            raise NotImplementedError('initialize the codes: load text embeddings')

        input_space = spaces.Vector(self.conditioning_space.size + self.probe_space.size)

        self.module.prepare(device=device, 
                            input_space=input_space, 
                            output_space=self.probe_space)
        
        if self._suppress is not None:
            self.module[-1].weight.data.div_(self._suppress)
            self.module[-1].bias.data.zero_()

        if self._load_path:
            assert self._load_path.exists(), f'Path {self._load_path} does not exist'
            data = torch.load(self._load_path)
            self.module.load_checkpoint(data=data['module'])
            # self.module.load_checkpoint(path=self._load_path)
            print(f'Loaded module from {self._load_path}')

            for param in self.module.parameters():
                param.requires_grad = False

        return super()._prepare(device=device)


    @tool('conditioning')
    def get_conditioning(self, add_class: torch.Tensor = None, remove_class: torch.Tensor = None):
        assert add_class is not None or remove_class is not None
        ref = add_class if add_class is not None else remove_class
        
        add_sel = add_class >= 0 if add_class is not None else torch.zeros_like(remove_class).bool()
        remove_sel = remove_class >= 0 if remove_class is not None else torch.zeros_like(add_sel).bool()
        replace_sel = add_sel & remove_sel
        add_sel = add_sel & ~replace_sel
        remove_sel = remove_sel & ~replace_sel

        B = ref.shape[0]
        D = self._add_codes.shape[-1]

        cond = torch.zeros(B, D, device=ref.device)

        if add_sel.any():
            if self._redundancy is None:
                add_cond = self._add_codes[add_class[add_sel]]
            else:
                tmpl_idx = torch.randint(self._redundancy, (add_sel.sum(),), device=ref.device)
                add_cond = self._add_codes[add_class[add_sel], tmpl_idx]
            cond[add_sel] = add_cond
        
        if remove_sel.any():
            if self._redundancy is None:
                remove_cond = self._remove_codes[remove_class[remove_sel]]
            else:
                tmpl_idx = torch.randint(self._redundancy, (remove_sel.sum(),), device=ref.device)
                remove_cond = self._remove_codes[remove_class[remove_sel], tmpl_idx]
            cond[remove_sel] = remove_cond

        if replace_sel.any():
            if self._redundancy is None:
                replace_cond = self._replace_codes[add_class[replace_sel], remove_class[replace_sel]]
            else:
                tmpl_idx = torch.randint(self._redundancy, (replace_sel.sum(),), device=ref.device)
                replace_cond = self._replace_codes[add_class[replace_sel], remove_class[replace_sel], tmpl_idx]
            cond[replace_sel] = replace_cond
            
        return cond
    @get_conditioning.space
    def conditioning_space(self) -> spaces.Vector:
        if isinstance(self._condition, int):
            return spaces.Vector(self._condition)
        if self._add_codes is None: # not prepared yet
            raise self._GearFailed
        return spaces.Vector(self._add_codes.shape[-1])

        
    @tool('probe')
    def apply_intervention(self, ambient: torch.Tensor, conditioning: torch.Tensor):
        input = torch.cat([ambient, conditioning], dim=1)
        delta = self.module(input)
        raw = ambient + delta if self._residual else delta
        if self._normalize:
            return F.normalize(raw, p=2, dim=-1)
        return raw
    @apply_intervention.space
    def probe_space(self, ambient: spaces.Vector) -> spaces.Vector:
        return ambient


    @tool('loss_mag')
    def magnitude_loss(self, ambient: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        if self._regularize_delta:
            return F.mse_loss(probe, ambient)
        return probe.pow(2).sum(dim=1).mean()


import wandb

@fig.component('prediction-gap')
class PredictionGap(Machine):
    label_space: spaces.Categorical = gear('label')

    def __init__(self, *, gap_threshold: float = 0.5, quantile: float = 0.5, range_quantile: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        assert 0 < gap_threshold, 'gap_threshold should be positive'
        assert 0 <= range_quantile < 0.5, 'range_quantile should be in [0, 0.5)'
        assert 0 <= quantile <= 1, 'quantile should be in [0, 1]'
        self.range_quantile = range_quantile
        self.quantile = quantile
        self.gap_threshold = gap_threshold

    # @tool('ambient_ranges')
    def compute_gap_ratio(self, ambient: torch.Tensor):
        lower = ambient.quantile(self.range_quantile, dim=0)
        upper = ambient.quantile(1 - self.range_quantile, dim=0)
        return (upper - lower) # per class

    @tool('ambient_gaps')
    def compute_ambient_gap(self, ambient: torch.Tensor, label: torch.tensor):
        
        by_class = []

        for pred, gt in zip(ambient.t(), label.t().bool()):
            pos = pred[gt]
            neg = pred[~gt]

            if pos.numel() == 0 or neg.numel() == 0:
                by_class.append(None)
                continue

            pos_quantile = pos.quantile(self.quantile)
            neg_quantile = neg.quantile(1 - self.quantile)
            # diff = pos_quantile - neg_quantile
            by_class.append([pos_quantile, neg_quantile])
        
        return by_class
    
    # @tool('intervened_gaps')
    def intervened_gap(self, intervened: torch.Tensor, ambient: torch.Tensor, intervention: torch.Tensor):
        return (intervened-ambient) * (-1) ** (~intervention)
    
    @tool('proximity_stats', 'unchanged_f1', 'changed_f1')
    def compute_proximity_stats(self, intervened, ambient, intervention, label):
        
        unchanged_f1 = []
        changed_f1 = []
        
        by_class = []

        for vals, base, ilbl, lbl in zip(intervened.t(), ambient.t(), 
                                                  intervention.bool().t(), label.bool().t()):
            changed = ilbl ^ lbl
            
            pos = base[lbl]
            neg = base[~lbl]
            
            if pos.numel() == 0 or neg.numel() == 0:
                by_class.append([None] * 9)
                continue

            pos = pos.quantile(self.quantile)
            neg = neg.quantile(1 - self.quantile)

            apred = base.sub(pos).abs() < base.sub(neg).abs()
            ipred = vals.sub(pos).abs() < vals.sub(neg).abs()

            valid = apred.eq(lbl)
            valid_changed = valid & changed
            valid_unchanged = valid & ~changed

            ok = valid.float().mean().item()
            n_changed = valid_changed.sum().item()
            n_unchanged = valid_unchanged.sum().item()

            row = [None] * 9
            row[:3] = [ok, n_changed, n_unchanged]

            if n_changed == 0 and n_unchanged == 0:
                by_class.append(row)
                continue

            tp, fp, fn = ipred & ilbl, ipred & ~ilbl, ~ipred & ilbl

            if n_changed > 0:

                ctp, cfp, cfn = tp[valid_changed].sum().item(), fp[valid_changed].sum().item(), fn[valid_changed].sum().item()
            
                cprec = ctp / (ctp + cfp) if ctp + cfp > 0 else 0
                crec = ctp / (ctp + cfn) if ctp + cfn > 0 else 0
                cf1 = 2 * cprec * crec / (cprec + crec) if cprec + crec > 0 else 0
                
                changed_f1.append(cf1)

                row[3:6] = [cprec, crec, cf1]
            
            if n_unchanged > 0:
            
                utp, ufp, ufn = tp[valid_unchanged].sum().item(), fp[valid_unchanged].sum().item(), fn[valid_unchanged].sum().item()

                uprec = utp / (utp + ufp) if utp + ufp > 0 else 0
                urec = utp / (utp + ufn) if utp + ufn > 0 else 0
                uf1 = 2 * uprec * urec / (uprec + urec) if uprec + urec > 0 else 0

                unchanged_f1.append(uf1)
                
                row[6:] = [uprec, urec, uf1]

            by_class.append(row)

        class_names = self.label_space.class_names
        tbl = wandb.Table(columns=['name', 'valid', 'changed', 'unchanged',
                                      'cprec', 'crec', 'cf1',
                                      'uprec', 'urec', 'uf1'])
        for name, row in zip(class_names, by_class):
            tbl.add_data(name, *row)

        return (tbl, 
                sum(unchanged_f1) / len(unchanged_f1) if unchanged_f1 else None, 
                sum(changed_f1) / len(changed_f1) if changed_f1 else None)



    # @tool('gap_stats')
    def gap_stats(self, intervened_gaps: torch.Tensor, ambient_gaps: torch.Tensor, intervention: torch.Tensor, label: torch.Tensor):
        by_class = []

        all_changed = intervention.bool() ^ label.bool()

        for gap, ref, changed in zip(intervened_gaps.t(), ambient_gaps, all_changed.t()):
            if ref is None or not changed.any():
                by_class.append(None)
                continue
            
            unchanged = ~changed

            pos, neg = ref
            reaction = gap / (pos - neg)
            # for reaction[changed]: 
            # 1 = prediction changed as much as from negative to positive


            
            by_class.append(reaction)

            pos_quantile = pos[intv[gt]].quantile(self.quantile)
            neg_quantile = neg[intv[~gt]].quantile(1 - self.quantile)
            diff = pos_quantile - neg_quantile
            by_class.append(diff)



        
        pass

