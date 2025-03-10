from .imports import *


from torchvision import transforms

@fig.component('prep-rawimage')
class PrepareRawImage(Machine):
    def __init__(self, resize: int = 248, crop: int = 224, **kwargs):
        super().__init__(**kwargs)
        self._resize = resize
        self._crop = crop
        self._tfm = None

    def _prepare(self, *, device = None):
        tfms = []
        if self._resize is not None:
            tfms.append(transforms.Resize(self._resize))
        if self._crop is not None:
            tfms.append(transforms.CenterCrop(self._crop))
        tfms.append(transforms.ToTensor())
        self._tfm = transforms.Compose(tfms)
        self._device = device
        return super()._prepare(device=device)

    @tool('image')
    def prepare_rgb(self, rawimage: List[Image.Image]) -> torch.Tensor:
        raw = torch.stack([self._tfm(img.convert('RGB')) for img in rawimage])
        return raw.to(self._device) # as float

    @tool('pixels')
    def prepare_image(self, image: List[Image.Image]) -> torch.Tensor:
        return image.mul(255).to(torch.uint8)



@fig.component('image-ops')
class ImageOps(Machine):
    def __init__(self, *, axis: int):
        super().__init__()
        assert axis in [1, 2, 3], f'Invalid axis: {axis}'
        self.axis = axis

    @tool('concat')
    def concat(self, src1: torch.Tensor, src2: torch.Tensor) -> torch.Tensor:
        return torch.cat([src1, src2], dim=self.axis)
    @concat.space
    def concat_space(self, src1: spaces.Image, src2: spaces.Image) -> spaces.Image:
        assert all(x == y for i, (x, y) in enumerate(zip(src1.shape(), src2.shape())) if i != self.axis), f'Images must have same shape: {src1.shape()} != {src2.shape()}'
        shape = src1.shape()
        shape[self.axis] = src1.shape()[self.axis] + src2.shape()[self.axis]
        return spaces.Image(*shape[1:])




@fig.component('pixel-processor')
class PixelProcessor(Machine):
    @tool('input')
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        B = image.shape[0]
        inp = image.view(B, -1).float().div_(127.5).sub_(1)
        return inp
    @preprocess.space
    def input_space(self, image: spaces.Image) -> spaces.Vector:
        return spaces.Vector(image.size)

    @tool('picture')
    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        out = output.view(*self.picture_space.shape(output.shape[0]))
        out = out.add(1).mul_(127.5).clamp_(0, 255).byte()
        return out
    @postprocess.space
    def picture_space(self, image: spaces.Vector) -> spaces.Image:
        return image




@fig.component('detacher')
class Detacher(Machine):
	@tool('detached')
	def detach(self, original: torch.Tensor) -> torch.Tensor:
		return original.detach()
	@detach.space
	def detached_space(self, original: spaces.AbstractSpace) -> spaces.AbstractSpace:
		return original



@fig.component('splitter')
class Splitter(Machine):
	def __init__(self, size0=None, size1=None, total_size=None, **kwargs):
		spec = sum(x is not None for x in [size0, size1, total_size])
		if spec == 3:
			assert size0 + size1 == total_size, f'total_size must be the sum of size0 and size1: {total_size} != {size0} + {size1}'
		elif spec == 2:
			if size0 is None:
				size0 = total_size - size1
			elif size1 is None:
				size1 = total_size - size0
			else:
				total_size = size0 + size1
		elif spec == 1:
			raise ValueError('Exactly two of the three parameters must be specified: size0, size1, total_size')
		super().__init__(**kwargs)
		self._size0 = size0
		self._size1 = size1


	def settings(self):
		return {**super().settings(), 
				'size0': self._size0, 
				'size1': self._size1}


	@tool('part0')
	def get_part0(self, original: torch.Tensor) -> torch.Tensor:
		return original.narrow(1, 0, self._size0)
	@get_part0.space
	def part0_space(self) -> spaces.Vector:
		return spaces.Vector(self._size0)
	

	@tool('part1')
	def get_part1(self, original: torch.Tensor) -> torch.Tensor:
		return original.narrow(1, self._size0, self._size1)
	@get_part1.space
	def part1_space(self) -> spaces.Vector:
		return spaces.Vector(self._size1)
	

	@tool('parts')
	def get_parts(self, original: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.get_part0(original), self.get_part1(original)
	

	@tool('merged')
	def merge_parts(self, part0: torch.Tensor, part1: torch.Tensor) -> torch.Tensor:
		return torch.cat([part0, part1], dim=1)
	@merge_parts.space
	def merged_space(self) -> spaces.Vector:
		return spaces.Vector(self._size0 + self._size1)



@fig.component('classification')
class Classification(Machine):
	@space('prediction')
	def prediction_space(self, label: spaces.Categorical) -> spaces.Logits:
		return spaces.Logits(label.n)


	@tool('loss')
	def classification_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		return F.cross_entropy(prediction, label)
	

	@tool('correct')
	def get_correct(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		return prediction.argmax(dim=1) == label


	@tool('accuracy') # indicator
	def get_accuracy(self, correct: torch.Tensor) -> float:
		return correct.float().mean().item()
	


from sklearn.metrics import roc_auc_score

@fig.component('multi-classification')
class MultiClassification(Machine):
	@space('prediction')
	def prediction_space(self, label: spaces.Boolean) -> spaces.Logits:
		return spaces.Logits(label.size)
	
	label_space = space('label')
	
	
	@indicator('loss')
	def classification_loss(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		"""
		:param prediction: (batch_size, num_classes)
		:param label: (batch_size, num_classes)
		"""
		return F.binary_cross_entropy_with_logits(prediction, label.float())
	
# class ROC_AUC(Machine):
	def _compute_roc_auc(self, prediction: torch.Tensor, label: torch.Tensor, average='macro', **kwargs) -> float:
		label = label.cpu().numpy()
		sel = label.sum(0) > 0
		label = label[:, sel]
		prediction = prediction.detach().cpu().numpy()[:,sel]
		out = roc_auc_score(label, prediction, multi_class='ovr', average=average, **kwargs)
		if average is None:
			full = np.array([None] * len(sel))
			full[sel] = out
			return full.tolist()
		return out


	@indicator('macro')
	def macro_roc_auc(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
		return self._compute_roc_auc(prediction, label, average='macro')

	@indicator('micro')
	def micro_roc_auc(self, prediction: torch.Tensor, label: torch.Tensor) -> float:
		return self._compute_roc_auc(prediction, label, average='micro')
	
	@tool('roc_auc')
	def roc_auc_by_class(self, prediction: torch.Tensor, label: torch.Tensor) -> List[Optional[float]]:
		return self._compute_roc_auc(prediction, label, average=None)
	@roc_auc_by_class.space
	def roc_auc_space(self, label: spaces.Boolean) -> spaces.Vector:
		return spaces.Bounded(label.size, batched=False, lower=0, upper=1)



@fig.component('difference')
class ScoreDifference(Machine):
	def __init__(self, bias: float = 0, **kwargs):
		super().__init__(**kwargs)
		self._bias = bias


	@tool('difference')
	def get_accuracy_gap(self, score: float, reference: float) -> float:
		return self._bias + score - reference



@fig.component('mse')
class MSE(Machine):
	@tool('loss')
	def mse_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		return F.mse_loss(prediction, target)


@fig.component('equivariance')
class Equivariance(Machine):
	def __init__(self, A: Callable[[torch.Tensor], torch.Tensor], 
				 B: Callable[[torch.Tensor], torch.Tensor], **kwargs):
		super().__init__(**kwargs)
		self.A = A
		self.B = B

	@tool('Ax')
	def apply_A(self, x: torch.Tensor) -> torch.Tensor:
		return self.A(x)
	
	@tool('Bx')
	def apply_B(self, x: torch.Tensor) -> torch.Tensor:
		return self.B(x)
	
	@tool('BAx')
	def apply_BA(self, Ax: torch.Tensor) -> torch.Tensor:
		return self.B(Ax)
	
	@tool('ABx')
	def apply_AB(self, Bx: torch.Tensor) -> torch.Tensor:
		return self.A(Bx)
	
	@tool('loss')
	def equivariance_loss(self, ABx: torch.Tensor, BAx: torch.Tensor) -> torch.Tensor:
		return F.mse_loss(ABx, BAx)



@fig.component('multi-loss')
class MultiLoss(Machine):
	def __init__(self, wts: dict[str, float], **kwargs):
		super().__init__(**kwargs)
		assert len(wts) and any(wt != 0 for wt in wts.values()), 'At least one loss must be specified'
		self._wts = wts
	

	@tool.from_context('loss')
	def full_loss(self, batch: Batch) -> torch.Tensor:
		total = 0
		for key, wt in self._wts.items():
			if wt != 0:
				total += batch[self.gap(f'loss_{key}')] * wt
		return total
	# @full_loss.genes


@fig.component('shadow')
class Shadow(Event):
	def __init__(self, reference: Machine, *, shadow_content: Machine = None, 
			   internal: dict = None, external: dict = None,
			  enable_grads: bool = False, sync_freq: int = 1, **kwargs):
		# if shadow_content is None:
		# 	shadow_content = copy.deepcopy(reference)
		if internal is None:
			internal = {}
		if external is None:
			external = {}
		super().__init__(**kwargs)
		self._enable_grads = enable_grads
		self._sync_freq = sync_freq
		self._reference = reference
		self._shadow = shadow_content
		self._shadow_internal = internal
		self._shadow_external = external

	def gizmos(self):
		yield from self._shadow_external.values() # blind faith

	_Mechanism = VizMechanism
	def _prepare(self, *, device = None, **kwargs):
		self._reference.prepare(device=device)
		if self._shadow is None:
			m = getattr(self._reference, '_mechanics', None)
			if m is not None:
				self._reference.mechanize()
			self._shadow = copy.deepcopy(self._reference)
			if m is not None:
				self._reference.mechanize(m)
		self._shadow.prepare(device=device)
		wrapped = self._Mechanism([self._shadow], internal=self._shadow_internal, external=self._shadow_external)
		self.include(wrapped)
		# for param, ref_param in zip(self._shadow.parameters(), self._reference.parameters()):
		# 	param.data.copy_(ref_param.data)
		if not self._enable_grads:
			for param in self._shadow.parameters():
				param.requires_grad = False
		return super()._prepare(device=device, **kwargs)
	
	def step(self, batch: Batch):
		itr = batch['num_iterations']
		if self._sync_freq and itr % self._sync_freq == 0 and itr > 0:
			for param, ref_param in zip(self._shadow.parameters(), self._reference.parameters()):
				param.data.copy_(ref_param.data)
		return super()._step(batch)



@fig.component('slice-wae')
class SlicedWAERegularization(Machine):
	def __init__(self, num_slices: int = 10, *, latent_dim: int = None, device: str = None, **kwargs):
		super().__init__(**kwargs)
		self._num_slices = num_slices
		self._latent_dim = latent_dim
		self._device = device

	@space('latent')
	def latent_space(self) -> spaces.Vector:
		if self._latent_dim is None:
			raise self._GearFailed('latent_dim not specified')
		return spaces.Vector(self._latent_dim)


	@tool('prior')
	def sample_prior(self, size: int):
		return torch.randn(size, self.latent_space.size).to(self._device)


	@tool('slice')
	def generate_slices(self) -> torch.Tensor:
		slices = torch.randn(self.latent_space.size, self._num_slices).to(self._device)
		slices /= slices.norm(dim=1, keepdim=True)
		return slices
	@generate_slices.space
	def _slice_space(self) -> spaces.Tensor:
		return spaces.Tensor([self.latent_space.size, self._num_slices])
	

	def _compute_sorted_slices(self, samples: torch.Tensor, slice: torch.Tensor) -> torch.Tensor:
		p = samples @ slice
		return p.sort(dim=1).values
	

	@tool('loss')
	def sliced_wae_loss(self, latent: torch.Tensor, prior: torch.Tensor, slice: torch.Tensor) -> torch.Tensor:
		proj_latent = self._compute_sorted_slices(latent, slice)
		proj_prior = self._compute_sorted_slices(prior, slice)
		return F.mse_loss(proj_latent, proj_prior)




