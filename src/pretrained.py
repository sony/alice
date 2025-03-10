from .imports import *
from .models import Model



@fig.component('pretrained-clip')
class CLIP(Model):
    def __init__(self, vit='ViT-B/32', **kwargs):
        super().__init__(**kwargs)
        self._vit_name = vit
        self._vit_code = self.parse_vit_model_name(vit)
        self._clip = None

    
    @property
    def name(self) -> str:
        return f'clip-{self._vit_code}'
    

    def _prepare(self, dataset: Optional[Dataset] = None, *, device: Optional[str] = None) -> Self:
        if self._clip is None:
            import clip
            model, preprocess = clip.load(self._vit_name, device)
            self._clip = model
            self._preprocess = preprocess
            self._device = device
            self._tokenizer = clip.tokenize
        return self


    def tokenize(self, text: Iterable[str]) -> torch.Tensor:
        return self._tokenizer(text).to(self._device)

    
    @staticmethod
    def parse_vit_model_name(fullname: str) -> str:
        '''
        fullname is something like: 'ViT-B/32' or 'vit-b/14' 
        outputs a "model code" like 'b32' or 'b14'
        '''
        full = fullname.lower()
        if full.startswith('vit-'):
            full = full[4:]
        if '/' not in full: # probably already a code
            return full
        size, patch_size = full.split('/')
        assert size in ['b'], f'Unknown vit size: {size} in {fullname}'
        assert patch_size in ['32', '16', '14'], f'Unknown patch size: {patch_size} in {fullname}'
        return f'{size}{patch_size}'


    @staticmethod
    def parse_model_code(code: str) -> str:
        '''recovers the full model name from the model code'''
        if code.lower().startswith('vit-'):
            return code
        if '/' in code:
            return f'ViT-{code}'
        size = code[0].upper()
        patch_size = code[1:]
        return f'ViT-{size}/{patch_size}'


    @tool('image')
    def load_images(self, image_path: Iterable[str]) -> torch.Tensor:
        return torch.stack([self._preprocess(Image.open(p)) for p in image_path]).to(self._device)
    

    @tool('image_embedding')
    def embed_images(self, image: torch.Tensor) -> torch.Tensor:
        return self._clip.encode_image(image)
    

    @tool('text_embedding')
    def embed_text(self, text: Iterable[str]) -> torch.Tensor:
        tokens = self.tokenize(text)
        return self._clip.encode_text(tokens)
    
    @tool('similarity_matrix')
    def similarity_matrix(self, image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(image_embedding.unsqueeze(1), text_embedding.unsqueeze(0), dim=-1)




@fig.component('clip-cls-imagenet')
class CLIP_ImageNet_Classifier(Model):
    def __init__(self, dataroot: Path = '/data/felix/imagenet/', vit='ViT-B/32', **kwargs):
        assert vit == 'ViT-B/32', f'Only ViT-B/32 is supported for now, got {vit}'
        if dataroot is not None:
            dataroot = Path(dataroot)
            assert dataroot.exists(), f'Dataroot {dataroot} does not exist'
        super().__init__(**kwargs)
        self._path = dataroot.joinpath('class_embeddings.pt')
        self._class_embeddings = None


    def _prepare(self, *, device = None):
        self._class_embeddings = torch.load(self._path).unsqueeze(0)
        if device is not None:
            self._class_embeddings = self._class_embeddings.to(device)
        return self


    @tool('prediction')
    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        if self._class_embeddings is None:
            self.prepare(device=embedding.device)
        # embedding: (batch, 512), class_embeddings: (1, 1000, 512)
        return F.softmax(F.cosine_similarity(embedding.unsqueeze(1), self._class_embeddings, dim=2), dim=1)
    
    
    def get_correct(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.argmax(prediction, dim=1) == label
    

    @tool('accuracy')
    def get_accuracy(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.get_correct(prediction, label).float())



@fig.component('timm')
class TIMM(Model):
    def __init__(self, name: str, pretrained: bool = True, replace_head: Optional[Union[int, bool]] = None,
                 as_eval: bool = None, **kwargs):
        if as_eval is None:
            as_eval = not pretrained and replace_head is None
        assert not isinstance(replace_head, int) or replace_head > 0, 'replace_head must be > 0, if set manually'
        super().__init__(**kwargs)
        self._name = name
        self._pretrained = pretrained
        self._as_eval = as_eval
        self._replace_head = replace_head
        self.timm_model = None


    @property
    def name(self) -> str:
        return self._name if self._pretrained else f'{self._name}-untrained'
    

    @staticmethod
    def infer_feature_space(model_name: str) -> Optional[spaces.Vector]:
        return {
            'vit_base_patch32_224': spaces.Sequence(768, 50, channels_first=False),
        }.get(model_name)

    @staticmethod
    def infer_image_space(model_name: str) -> Optional[spaces.Image]:
        return {
            'vit_base_patch32_224': spaces.Pixels(3, 224, 224),
        }.get(model_name)

    @staticmethod
    def infer_output_space(model_name: str) -> Optional[spaces.Image]:
        return {
            'vit_base_patch32_224': spaces.Logits(1000),
        }.get(model_name)

    def _prepare(self, *, device = None):
        try:
            import timm
        except ImportError:
            raise
        else:
            kwargs = {}
            if self._replace_head is not None:
                kwargs['num_classes'] = self.output_space.size
            self.timm_model = timm.create_model(self._name, pretrained=self._pretrained, **kwargs)
            if device is not None:
                self.timm_model.to(device)
            if self._as_eval:
                self.timm_model.eval()
            else:
                self.timm_model.train()

        return super()._prepare(device=device)
    

    @space('input')
    def input_space(self) -> spaces.Pixels:
        sp = self.infer_image_space(self._name)
        if sp is None:
            raise self._GearFailed(f'Unknown image space for {self._name!r}')
        return sp


    @tool('features')
    def get_features(self, input: torch.Tensor) -> torch.Tensor:
        return self.timm_model.forward_features(input)
    @get_features.space
    def features_space(self) -> spaces.Vector:
        feats = self.infer_feature_space(self._name)
        if feats is None:
            raise self._GearFailed(f'Unknown feature space for {self._name!r}')
        return feats


    @tool('prediction')
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.timm_model.forward_head(features)
    
    
    @tool('output')
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.timm_model(input)
    @forward.space
    @predict.space
    def output_space(self) -> spaces.Logits:
        if type(self._replace_head) is int:
            return spaces.Logits(self._replace_head)
        elif self._replace_head:
            raise self._GearFailed('Expecting output space to be set elsewhere to replace head')
        sp = self.infer_output_space(self._name)
        if sp is None:
            raise self._GearFailed(f'Unknown output space for {self._name!r}')
        return sp


@fig.component('vit')
class ViT(TIMM):
    def __init__(self, name: str = 'vit_base_patch16_224', only_cls: bool = True, pool: str = None, **kwargs):
        if not name.startswith('vit_'):
            print(f'WARNING: ViT model name should start with "vit_": {name!r}')
        super().__init__(name=name, **kwargs)
        self._only_cls = only_cls
        self._pool = pool


    @tool('prediction')
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        if self._only_cls:
            return self.timm_model.head(features)
        else:
            raise ValueError('Not implemented')
    
    
    @tool('features')
    def get_features(self, input: torch.Tensor) -> torch.Tensor:
        full = super().get_features(input)
        if self._only_cls:
            return full[:, 0]
        if self._pool is not None:
            if self._pool == 'mean':
                return full.mean(dim=1)
            if self._pool == 'max':
                return full.max(dim=1)
        return full
    @get_features.space
    def features_space(self) -> spaces.Vector:
        feats = self.infer_feature_space(self._name)
        if feats is None:
            raise self._GearFailed(f'Unknown feature space for {self._name!r}')
        if self._only_cls or self._pool is not None:
            return spaces.Vector(feats.channels)
        return feats
        

@fig.script('check-timm')
def check_timm_spaces(cfg: fig.Configuration):

    cfg.push('model._type', 'timm', overwrite=False, silent=True)
    model: TIMM = cfg.pull('model')

    input_space = cfg.pull('input-shape', None)

    inp_label = cfg.pull('input-label', 'input')
    labels = cfg.pull('products', list(model.gizmos()))

    if input_space is None:
        try:
            input_space = model.input_space
        except GearGrabError:
            print('WARNING: Could not infer input space, defaulting to (3, 224, 224)')
            input_space = spaces.Pixels(3, 224, 224)
    elif not isinstance(input_space, spaces.SpaceBase):
        input_space = spaces.Pixels(*input_space)
    
    print(f'Model: {model.name}')
    model.prepare()

    inp = torch.zeros(input_space.shape(1))
    print(f'{inp_label}: {inp.shape}')
    ctx = Context(DictGadget({inp_label: inp}), model)
    ctx.mechanize()
    mech = ctx.mechanics()

    table = [[inp_label, inp.shape, mech.grab(inp_label, None)]]

    with torch.no_grad():
        for label in labels:
            try:
                out = ctx[label]
            except Exception as e:
                out = e

            expected = mech.grab(label, None)

            if isinstance(out, Exception):
                table.append([label, out, expected])
            else:
                table.append([label, out.shape, expected])

    def safe_verify(actual, expected):
        try:
            return actual == expected.shape(1)
        except:
            return False

    for row in table:
        _, actual, expected = row
        if isinstance(actual, Exception):
            row.append(colorize('FAILED', 'red'))
            row[1] = str(actual)
        else:
            if safe_verify(actual, expected):
                row.append(colorize('CORRECT', 'green'))
            elif expected is None:
                row.append(colorize('MISSING', 'yellow'))
            else:
                row.append(colorize('INVALID', 'red'))
            row[1] = tuple(actual)

    print(tabulate(table, headers=['Label', 'Actual', 'Expected', 'Status']))
    return ctx



@fig.component('vit-raw')
class RawViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transform = None


    def _prepare(self, *, device = None):
        out = super()._prepare(device=device)
        if self._transform is None:
            assert self.timm_model is not None, 'Model not initialized'
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            self._config = resolve_data_config({}, model=self.timm_model)
            self._transform = create_transform(**self._config)
            self._device = device
        return out


    @tool('image')
    def transform_image(self, rawimage: List[Image.Image]) -> torch.Tensor:
        return torch.stack([self._transform(img.convert('RGB')) for img in rawimage]).to(self._device)



class HF_TextEncoder(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None
        self.transformer = None


    def _load_model_and_tokenizer(self, device: Optional[str] = None):
        raise NotImplementedError('Subclasses must implement this method')


    def _prepare(self, *, device = None):
        transformer, tokenizer = self._load_model_and_tokenizer(device=device)
        self.tokenizer = tokenizer
        self.transformer = transformer
        self._device = device
        return super()._prepare(device=device)



@fig.component('gpt2')
class GPT2(HF_TextEncoder):
    def __init__(self, *args, pool: str = 'mean', **kwargs):
        assert pool in ['mean', 'sum', 'last'], f'Unknown pooling method: {pool}'
        super().__init__(*args, **kwargs)
        self._pool = pool

    @property
    def name(self) -> str:
        return f'gpt2-{self._pool}'

    def _load_model_and_tokenizer(self, device: Optional[str] = None):
        from transformers import GPT2Tokenizer, GPT2Model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        if device is not None:
            model.to(device)
        return model, tokenizer


    @tool('tokens', 'attention_mask')
    def tokenize(self, text: Iterable[str]) -> torch.Tensor:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self._device)
        assert len(inputs) == 2, f'Tokenizer did not return input_ids and attention_mask: {inputs.keys()}'
        assert 'input_ids' in inputs and 'attention_mask' in inputs, 'Tokenizer did not return input_ids and attention_mask'
        return inputs.input_ids, inputs.attention_mask


    @tool('token_embedding')
    def embed_tokens(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens)
        rawemb = self.transformer(input_ids=tokens, attention_mask=attention_mask).last_hidden_state
        return rawemb
        

    @tool('output')
    def aggregate(self, token_embedding: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self._pool == 'last':
            return token_embedding[:, -1]
        sumemb = token_embedding.sum(dim=1) if attention_mask is None \
                else (token_embedding * attention_mask.unsqueeze(-1)).sum(dim=1)
        if self._pool == 'mean':
            return sumemb.div(sumemb.shape[1]) if attention_mask is None \
                    else sumemb.div(attention_mask.sum(dim=1, keepdim=True))
        return sumemb
    @aggregate.space
    def embedding_space(self) -> spaces.Vector:
        return spaces.Vector(768)



@fig.component('bert')
class BERT(HF_TextEncoder):
    def __init__(self, *args, cased: bool = False, pool: str = 'cls', include_cls: bool = False, **kwargs):
        assert pool in ['cls', 'mean', 'sum'], f'Unknown pooling method: {pool}'
        super().__init__(*args, **kwargs)
        self._cased = cased
        self._pool = pool
        self._include_cls = include_cls

    @property
    def name(self) -> str:
        return f'bert{"-cased" if self._cased else ""}-{self._pool}'
    
    def _load_model_and_tokenizer(self, device: Optional[str] = None):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased' if self._cased else 'bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-cased' if self._cased else 'bert-base-uncased')
        if device is not None:
            model.to(device)
        return model, tokenizer
    
    @tool('tokens', 'attention_mask', 'token_type_ids')
    def tokenize(self, text: Iterable[str]) -> torch.Tensor:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self._device)
        assert len(inputs) == 3, f'Tokenizer did not return input_ids, attention_mask and token_type_ids: {inputs.keys()}'
        assert 'input_ids' in inputs and 'attention_mask' in inputs and 'token_type_ids' in inputs, 'Tokenizer did not return input_ids, attention_mask and token_type_ids'
        return inputs.input_ids, inputs.attention_mask, inputs.token_type_ids
    
    @tool('token_embedding')
    def embed_tokens(self, tokens: torch.Tensor, attention_mask: torch.Tensor = None, token_type_ids: torch.Tensor = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(tokens)
        rawemb = self.transformer(input_ids=tokens, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        return rawemb
    
    @tool('output')
    def aggregate(self, token_embedding: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self._pool == 'cls':
            return token_embedding[:, 0]
        if not self._include_cls:
            token_embedding = token_embedding[:, 1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1:]
        sumemb = token_embedding.sum(dim=1) if attention_mask is None \
                else (token_embedding * attention_mask.unsqueeze(-1)).sum(dim=1)
        if self._pool == 'mean':
            return sumemb.div(sumemb.shape[1]) if attention_mask is None \
                    else sumemb.div(attention_mask.sum(dim=1, keepdim=True))
        return sumemb
    @aggregate.space
    def embedding_space(self) -> spaces.Vector:
        return spaces.Vector(768)

    

