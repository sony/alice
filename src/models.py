from .imports import *
from omnilearn.op import Machine



@fig.component('autoencoder')
class Autoencoder(Model):
	def __init__(self, encoder: Model, decoder: Model, *, latent_dim: int = None, rec_type: str = 'mse', **kwargs):
		super().__init__(**kwargs)
		# encoder.gauge_apply({'input': self.gap('observation'), 
		# 					 'output': self.gap('latent')})
		# decoder.gauge_apply({'input': self.gap('latent'),
		# 					 'output': self.gap('reconstruction')})
		self.encoder = encoder
		self.decoder = decoder
		# self.include(self.encoder, self.decoder)
		self._latent_dim = latent_dim
		self._rec_type = rec_type


	@space('observation')
	def observation_space(self, reconstruction: spaces.AbstractSpace = None) -> spaces.AbstractSpace:
		if reconstruction is None:
			raise self._GearFailed('reconstruction space not specified')
		return reconstruction

	@property
	def name(self) -> str:
		return f'ae{self.latent_space.size}'


	def settings(self):
		return {'latent_space': self._latent_dim or self.latent_space.json(), 
				'encoder': self.encoder.settings(), 
				'decoder': self.decoder.settings()}


	def checkpoint(self, path = None):
		if path is None:
			return {'encoder': self.encoder.checkpoint(), 
					'decoder': self.decoder.checkpoint()}
		self.encoder.checkpoint(path.parent / f'{path.stem}-encoder')
		self.decoder.checkpoint(path.parent / f'{path.stem}-decoder')
		return self
	

	def load_checkpoint(self, *, path=None, data=None):
		assert (path is None) ^ (data is None), 'Exactly one of path or data must be specified'
		if data is not None:
			self.encoder.load_checkpoint(data=data['encoder'])
			self.decoder.load_checkpoint(data=data['decoder'])
			return self
		self.encoder.load_checkpoint(path=path.parent / f'{path.stem}-encoder')
		self.decoder.load_checkpoint(path=path.parent / f'{path.stem}-decoder')
		return self


	def _prepare(self, *, device: Optional[str] = None) -> Self:
		self.encoder.prepare(device=device, input_space=self.observation_space, output_space=self.latent_space)
		self.decoder.prepare(device=device, input_space=self.latent_space, output_space=self.reconstruction_space)
		return self
	

	@tool('latent')
	def encode(self, observation: torch.Tensor) -> torch.Tensor:
		return self.encoder(observation)
	@encode.space
	def latent_space(self) -> spaces.Vector:
		if self._latent_dim is None:
			raise self._GearFailed('latent_dim not specified')
		return spaces.Vector(self._latent_dim)

	@tool('reconstruction')
	def decode(self, latent: torch.Tensor) -> torch.Tensor:
		return self.decoder(latent)
	@decode.space
	def reconstruction_space(self, observation: spaces.AbstractSpace = None) -> spaces.AbstractSpace:
		if observation is None:
			raise self._GearFailed('observation space not specified')
		return observation

	@tool('loss')
	def reconstruction_loss(self, observation: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
		if self._rec_type == 'bce':
			return F.binary_cross_entropy(reconstruction.add(1).div(2), observation.add(1).div(2))
		return F.mse_loss(reconstruction, observation)



@fig.component('sae')
class SphericalAutoencoder(Autoencoder):
	@tool('latent')
	def encode(self, observation: torch.Tensor) -> torch.Tensor:
		unnorm_latent = super().encode(observation)
		return F.normalize(unnorm_latent, p=2, dim=-1)

	@property
	def name(self) -> str:
		return f'sae{self.latent_space.size}'


import wandb, pandas as pd, seaborn as sns, matplotlib.pyplot as plt, numpy as np

@fig.component('vae')
class VAE(Autoencoder):
	def __init__(self, encoder: Model, decoder: Model, *, latent_dim: int = None, std_threshold: float = 0.95, **kwargs):
		super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim, **kwargs)
		self._std_threshold = std_threshold
	
	@property
	def name(self) -> str:
		return f'vae{self.latent_space.size}'

	def _prepare(self, *, device: Optional[str] = None) -> Self:
		posterior_params = spaces.Vector(2*self.latent_space.size)
		self.encoder.prepare(device=device, input_space=self.observation_space, output_space=posterior_params)
		self.decoder.prepare(device=device, input_space=self.latent_space, output_space=self.reconstruction_space)
		return self
	
	@tool('posterior')
	def encode(self, observation: torch.Tensor) -> torch.distributions.Distribution:
		posterior_params = self.encoder(observation)
		mean, logstd = torch.chunk(posterior_params, 2, dim=-1)
		std = F.softplus(logstd).clamp(min=1e-6)
		return torch.distributions.Normal(mean, std)
	
	@tool('latent')
	def sample_posterior(self, posterior: torch.distributions.Distribution) -> torch.Tensor:
		return posterior.rsample()

	@tool('mean')
	def posterior_mean(self, posterior: torch.distributions.Distribution) -> torch.Tensor:
		return posterior.mean()

	@tool('loss_kl')
	def kl_divergence(self, posterior: torch.distributions.Distribution) -> torch.Tensor:
		return torch.distributions.kl_divergence(posterior, torch.distributions.Normal(0, 1)).sum(-1).mean()

	@tool('latent_usage')
	def viz_latent_usage(self, posterior: torch.distributions.Normal) -> float:
		std = posterior.scale

		samples = std.detach().cpu().numpy()

		N, D = samples.shape

				# --- 2. Convert data into a "long" pandas DataFrame for Seaborn ---
		df = pd.DataFrame(samples, columns=[f"Dim {i}" for i in range(D)])
		df_melt = df.melt(var_name="Dimension", value_name="Value")
		# Now df_melt has columns: ["Dimension", "Value"], 
		# where each dimension is repeated N times.

		# --- 3. Create the violin plot ---
		plt.figure(figsize=(8, 5))
		sns.violinplot(x="Dimension", y="Value", data=df_melt, inner='points', color="skyblue", cut=0)

		# --- 4. Compute statistics (min, max, mean) by dimension ---
		means = samples.mean(axis=0)
		plt.scatter(np.arange(D), means, color="red",   marker="o",  s=30,  zorder=3)

		# show a grid line every 5 dimensions
		plt.xticks(ticks=np.arange(0, D, 5), labels=[f"" for i in range(0, D, 5)])
		plt.grid(axis='x', which='major', color='gray', linestyle='--', linewidth=0.5)

		plt.ylim(0, 1.5)

		# remove xlabel and ylabel
		plt.xlabel("")
		plt.ylabel("")

		# remove surrounding white space
		plt.tight_layout()
		img = wandb.Image(plt)
		plt.close()
		return img



@fig.modifier('supervised')
class Supervised(Autoencoder):
	def __init__(self, encoder: Model, decoder: Model, *, classifier: Model = None, fix_classifier: str = None, **kwargs):
		if fix_classifier is not None:
			fix_classifier = Path(fix_classifier)
		super().__init__(encoder=encoder, decoder=decoder, **kwargs)
		self.classifier = classifier
		self._fix_classifier = fix_classifier

	feature_space = gear('features')

	def _prepare(self, *, device: Optional[str] = None, **kwargs) -> Self:
		if self.classifier is not None:
			self.classifier.prepare(device=device, input_space=self.feature_space, output_space=self.prediction_space)

			if self._fix_classifier:
				self.classifier.load_checkpoint(path=self._fix_classifier)
				for param in self.classifier.parameters():
					param.requires_grad = False
		
		return super()._prepare(device=device, **kwargs)


	@tool('prediction')
	def predict(self, features: torch.Tensor) -> torch.Tensor:
		return self.classifier(features)
	@predict.space
	def prediction_space(self, label: spaces.Categorical) -> spaces.AbstractSpace:
		return spaces.Logits(label.n)


	def settings(self):
		return {**super().settings(), 
				'classifier': self.classifier.settings()}


	def checkpoint(self, path = None):
		if path is None:
			return {**super().checkpoint(), 'classifier': self.classifier.checkpoint()}
		self.classifier.checkpoint(path.parent / f'{path.stem}-classifier')
		return super().checkpoint(path=path)
	

	def load_checkpoint(self, *, path=None, data=None):
		assert (path is None) != (data is None), 'Exactly one of path or data must be specified'
		if data is not None:
			self.classifier.load_checkpoint(data=data['classifier'])
			return super().load_checkpoint(data=data)
		self.classifier.load_checkpoint(path=path.parent / f'{path.stem}-classifier')
		return super().load_checkpoint(path=path)



@fig.component('classifier')
class Classifier(Model):
	def __init__(self, head: Model, extractor: Model = None, **kwargs):
		super().__init__(**kwargs)
		if extractor is None:
			head.gauge_apply({'input': self.gap('observation'), 
							  'output': self.gap('prediction')})
		else:
			extractor.gauge_apply({'input': self.gap('observation'), 
								   'output': self.gap('features')})
			head.gauge_apply({'input': self.gap('features'), 
							  'output': self.gap('prediction')})
			self.extractor = extractor
			self.include(self.extractor)
		self.include(head)
		self.head = head

	@property
	def name(self) -> str:
		return f'{self.head.name}{f"-{self.extractor.name}" if hasattr(self, "extractor") else ""}'
	

	def _prepare(self, *, device: Optional[str] = None) -> Self:
		self.head.prepare(device=device)
		if hasattr(self, 'extractor'):
			self.extractor.prepare(device=device)
		return self
	
	def settings(self):
		return {**super().settings(), 
				'head': self.head.settings(), 
				'extractor': getattr(self, 'extractor', None) and self.extractor.settings()}



