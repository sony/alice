from .imports import *


@fig.component('concept')
class ConceptAlgebra(Machine):
    def __init__(self, as_delta: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._projs = None
        self._as_delta = as_delta

    def extract_projections(self, embedding: torch.Tensor, label: torch.Tensor):
        projs = []
        for lbl in range(label.shape[1]):
            pos = embedding[label[:, lbl]]
            proj = pos.T @ pos / label[:, lbl].sum()
            projs.append(proj)
        projs.append(torch.eye(embedding.shape[1], device=embedding.device))
        projs = torch.stack(projs)
        self._projs = projs

    @tool('probe')
    def apply_intervention(self, ambient: torch.Tensor, add_class: torch.Tensor = None, remove_class: torch.Tensor = None):
        assert self._projs is not None
        delta = 0
        if add_class is not None:
            delta += self._projs[add_class] @ ambient.unsqueeze(-1)
        if remove_class is not None:
            delta -= self._projs[remove_class] @ ambient.unsqueeze(-1)
        probe = (ambient + delta.squeeze(-1)) if self._as_delta else delta.squeeze(-1)
        probe = F.normalize(probe, dim=-1)
        return probe


@fig.component('clip')
class CLIP(Model):
    def __init__(self, text_encoder: Model, image_encoder: Model, *, latent_dim: int = None,
              load_text_encoder: str = None, load_image_encoder: str = None,  
                 normalize: bool = True, no_scaling: bool = False, logit_scale: float = np.log(1 / 0.07), **kwargs):
        super().__init__(**kwargs)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self._text_encoder_load = None if load_text_encoder is None else Path(load_text_encoder)
        self._image_encoder_load = None if load_image_encoder is None else Path(load_image_encoder)
        self._latent_dim = latent_dim
        self._normalize = normalize
        self._no_scaling = no_scaling
        self._initial_logit_scale = logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale)

    @property
    def name(self) -> str:
        return 'clip' if self._latent_dim is None else f'clip{self._latent_dim}'

    def _prepare(self, *, device: Optional[str] = None) -> Self:
        self.text_encoder.prepare(device=device, input_space=self.text_features_space, output_space=self.embedding_space)
        self.image_encoder.prepare(device=device, input_space=self.image_features_space, output_space=self.embedding_space)
        
        if self._text_encoder_load:
            self.text_encoder.load_checkpoint(path=self._text_encoder_load)
        
        if self._image_encoder_load:
            self.image_encoder.load_checkpoint(path=self._image_encoder_load)

        return self
    
    # def settings(self):
    # 	return {'latent_space': self._latent_dim or self.latent_space.json(), 
    # 			'encoder': self.encoder.settings(), 
    # 			'decoder': self.decoder.settings()}

    def settings(self):
        return {'text_encoder': self.text_encoder.settings(), 
          'image_encoder': self.image_encoder.settings(),
          'latent_dim': self._latent_dim or self.latent_space.json(), 
          'normalize': self._normalize, 'initial_logit_scale': self._initial_logit_scale}

    def checkpoint(self, path = None):
        if path is None:
            return {'text_encoder': self.text_encoder.checkpoint(), 
                    'image_encoder': self.image_encoder.checkpoint(),
                    'logit_scale': self.logit_scale.item()}
        self.text_encoder.checkpoint(path.parent / f'{path.stem}-text-encoder')
        self.image_encoder.checkpoint(path.parent / f'{path.stem}-image-encoder')
        return self
    

    def load_checkpoint(self, *, path=None, data=None):
        assert (path is None) ^ (data is None), 'Exactly one of path or data must be specified'
        if data is not None:
            self.text_encoder.load_checkpoint(data=data['text_encoder'])
            self.image_encoder.load_checkpoint(data=data['image_encoder'])
            return self
        self.text_encoder.load_checkpoint(path=path.parent / f'{path.stem}-text-encoder')
        self.image_encoder.load_checkpoint(path=path.parent / f'{path.stem}-image-encoder')
        return self


    text_features_space = gear('text_features')
    image_features_space = gear('image_features')

    @space('text_embedding')
    @space('image_embedding')
    def embedding_space(self) -> spaces.Vector:
        if self._latent_dim is None:
            raise self._GearFailed('latent_dim not specified')
        return spaces.Vector(self._latent_dim)


    @tool('text_embedding')
    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        emb = self.text_encoder(text_features)
        if self._normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @tool('image_embedding')
    def encode_image(self, image_features: torch.Tensor) -> torch.Tensor:
        emb = self.image_encoder(image_features)
        if self._normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb
    
    @tool('logits')
    def similarity(self, text_embedding: torch.Tensor, image_embedding: torch.Tensor):
        if not self._normalize:
            text_embedding = F.normalize(text_embedding, p=2, dim=-1)
            image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        return image_embedding @ text_embedding.t()

    @indicator('logit_scale')
    def current_logit_scale(self):
        return self.logit_scale.item()

    @indicator('positive_mean')
    def viz_positives(self, logits_per_image: torch.Tensor):
        return logits_per_image.diag().mean().item()
    @indicator('negative_mean')
    def viz_negatives(self, logits_per_image: torch.Tensor):
        total = logits_per_image.sum() - logits_per_image.diag().sum()
        N = logits_per_image.shape[0]
        return total.item() / (N * (N - 1))

    @indicator('loss')
    def clip_loss(self, logits: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        if self._no_scaling:
            scaled_logits = logits
        else:
            scaled_logits = logits * self.logit_scale.exp().clamp(max=100)

        image_loss = F.cross_entropy(scaled_logits, labels)
        text_loss = F.cross_entropy(scaled_logits.t(), labels)

        return (image_loss + text_loss) / 2


@fig.component('cyclip')
class CyCLIP(CLIP):
    def _shift_embeddings(self, embeddings):
        return torch.cat([embeddings[1:], embeddings[:1]], dim=0)
        
    @tool('shifted_text_embedding')
    def shift_text_embeddings(self, text_embedding: torch.Tensor):
        return self._shift_embeddings(text_embedding)
    
    @tool('shifted_image_embedding')
    def shift_image_embeddings(self, image_embedding: torch.Tensor):
        return self._shift_embeddings(image_embedding)

    @indicator('loss_inmodal')
    def in_modal_consistency(self, text_embedding: torch.Tensor, image_embedding: torch.Tensor,
                          shifted_text_embedding: torch.Tensor, shifted_image_embedding: torch.Tensor) -> torch.Tensor:
        
        t, st = text_embedding.unsqueeze(1), shifted_text_embedding.unsqueeze(2)
        i, si = image_embedding.unsqueeze(1), shifted_image_embedding.unsqueeze(2)

        text = t @ st
        image = i @ si

        return F.mse_loss(text.squeeze(), image.squeeze())

    @indicator('loss_cross')
    def cross_modality_consistency(self, text_embedding: torch.Tensor, image_embedding: torch.Tensor,
                        shifted_text_embedding: torch.Tensor, shifted_image_embedding: torch.Tensor) -> torch.Tensor:

        t, si = text_embedding.unsqueeze(1), shifted_image_embedding.unsqueeze(2)
        i, st = image_embedding.unsqueeze(1), shifted_text_embedding.unsqueeze(2)

        text = t @ si
        image = i @ st

        return F.mse_loss(text.squeeze(), image.squeeze())



@fig.component('alignment')
class AlignmentMetrics(Machine):
    @indicator('alignment')
    def compute_alignment(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        return similarity_matrix.diag().mean().item()
    
    @indicator('uniformity')
    def compute_uniformity(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        ex = similarity_matrix.double().mul(-1).exp()
        total = ex.sum() - ex.diag().sum()
        N = similarity_matrix.shape[0]
        return np.log(total.item() / (N * (N - 1))).item()
    
    @indicator('unalignment')
    def compute_unalignment(self, similarity_matrix: torch.Tensor) -> torch.Tensor:
        mat = similarity_matrix.double()
        total = mat.sum() - mat.diag().sum()
        N = similarity_matrix.shape[0]
        return total.item() / (N * (N - 1))
    
    @tool('retrieval_order')
    def compute_retrieval(self, similarity_matrix: torch.Tensor):
        order = similarity_matrix.sort(dim=1).indices.cpu()
        order = order - torch.arange(order.shape[1], device=order.device).unsqueeze(1)
        order %= order.shape[1]
        return order

    @indicator('ret1')
    def compute_retrieval_1(self, retrieval_order: torch.Tensor) -> torch.Tensor:
        return retrieval_order[:, -1:].eq(0).sum(dim=1).float().mean().item()
    
    @indicator('ret5')
    def compute_retrieval_5(self, retrieval_order: torch.Tensor) -> torch.Tensor:
        return retrieval_order[:, -5:].eq(0).sum(dim=1).float().mean().item()
    
    @indicator('ret10')
    def compute_retrieval_10(self, retrieval_order: torch.Tensor) -> torch.Tensor:
        return retrieval_order[:, -10:].eq(0).sum(dim=1).float().mean().item()
