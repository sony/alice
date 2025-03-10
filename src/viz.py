from .imports import *



@fig.component('vecinfo')
class VectorInfo(Machine):
	@indicator('magnitude')
	def magnitude(self, vector: torch.Tensor) -> torch.Tensor:
		return vector.norm(dim=-1).mean()
	
	@indicator('scale')
	def scale(self, vector: torch.Tensor) -> torch.Tensor:
		return vector.abs().mean()



@fig.component('support')
class VectorSupport(Machine):
	@indicator('amd') # average minimum distance
	def support(self, src, tgt):
		'''
		computes the average minimum distance between the source and target vectors
		src: B x D
		tgt: B x D
		return: mean of the minimum distances
		'''
		return torch.cdist(src.unsqueeze(1), tgt.unsqueeze(0)).min(dim=-1).values.mean()
		


@fig.component('image-comparison')
class ImageComparison(Machine):
	def __init__(self, max_imgs = 12, **kwargs):
		super().__init__(**kwargs)
		self._max_imgs = max_imgs
		from torchvision.transforms import ToPILImage
		self._to_pil = ToPILImage()

	@tool('vcomp')
	def compare_images(self, source, target, caption = None):
		import torchvision, wandb
		imgs = torch.cat([source, target], dim=3)
		n = min(imgs.size(0), self._max_imgs)
		# B, C, H, W = raw.shape
		# imgs = wandb.Image(self._to_pil(raw[:n].permute(1,2,0,3).reshape(C, H, n*W)))
		nrows = int(n ** 0.5)
		grid = torchvision.utils.make_grid(imgs[:n], nrow=nrows)

		if caption is not None:
			if isinstance(caption, torch.Tensor):
				caption = str(caption[:n].tolist())
			elif isinstance(caption, (list, tuple)):
				caption = str(caption)
			elif isinstance(caption, str):
				pass
			else:
				raise ValueError(f'Invalid caption type: {type(caption)}')
		return wandb.Image(self._to_pil(grid), caption=caption)



