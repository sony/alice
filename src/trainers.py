from .imports import *
# from omnilearn import Trainer #as TrainerBase
# from .machines import Machine



# class Trainer(TrainerBase):
# 	def _setup_fit(self, src: Dataset, *, device: str = None, **settings: Any) -> Planner:
# 		planner = super()._setup_fit(src, device=device, **settings)
# 		for e in self._env:
# 			if isinstance(e, Machine):
# 				e.prepare(src, device=device)
# 		return planner


@fig.component('intervention-trainer')
class InterventionTrainer(Trainer):
	def __init__(self, intervention: Model, intervention_optim: Adam, *, 
			  clear_cache: bool = False, step2: str = None, 
			  **kwargs):
		super().__init__(**kwargs)
		self.intervention = intervention
		self.intervention_optim = intervention_optim
		self._clear_cache = clear_cache
		self._step2 = step2


	def checkpoint(self, path: Path) -> None:
		super().checkpoint(path)
		self.intervention.checkpoint(path / 'interventions')
		self.intervention_optim.checkpoint(path / 'intervention-optimizer')
		return path
	
	def settings(self):
		data = super().settings()
		data['intervention'] = self.intervention.settings()
		data['intervention-optimizer'] = self.intervention_optim.settings()
		return data

	def _prepare(self, *, device: str = None) -> Self:
		super()._prepare(device=device)
		self.intervention.prepare(device=device)
		if self._step2 is not None:
			targets = []
			if 'd' in self._step2:
				targets.append(self.model.decoder)
			if 'c' in self._step2:
				targets.append(self.model.classifier)
			self.intervention_optim.setup(self.model.encoder)
			self.intervention_optim.targets = targets
		else:
			self.intervention_optim.setup(self.intervention)
			self.intervention_optim.targets = [self.model]
		self.intervention_optim.prepare(device=device)
		return self
	
	
	def setup(self, src: 'AbstractDataset', *, device: str = None):
		out = super().setup(src, device=device)
		print(self.intervention)
		return out


	def gadgetry(self):
		yield self.intervention
		yield from super().gadgetry()


	def learn(self, batch: Batch) -> bool:
		self._optimizer.step(batch)
		if self._clear_cache:
			batch.clear_cache()
		self.intervention_optim.step(batch)

		for e in self._events.values():
			e.step(batch)

		reporter = self.reporter
		if reporter is not None:
			reporter.step(batch)

		return self._terminate_fit(batch)


# from omnilearn.compute import PytorchOptimizer

# @fig.modifier('jumpy')
# class Jumpy(Machine, PytorchOptimizer):
@fig.component('avoid-adam')
class AvoidAdam(Adam):
	"""Jumps over undesirable parameters in `targets` during optimization"""
	def __init__(self, *args, targets: Union[Model, Iterable[Model]] = None, **kwargs):
		if isinstance(targets, Model):
			targets = [targets]
		super().__init__(*args, **kwargs)
		self.targets = targets

	def step(self, batch):
		if self.targets is not None:
			for target in self.targets:
				for param in target.parameters():
					param.requires_grad = False
		
		out = super().step(batch)

		if self.targets is not None:
			for target in self.targets:
				for param in target.parameters():
					param.requires_grad = True

		return out

