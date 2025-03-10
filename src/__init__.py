from . import util
# from .op import train, embed
from . import models
from . import dataset
from . import op
from . import interventions
from . import trainers
from . import helpers
from . import viz
from . import baselines

import omnifig as _fig


@_fig.script('test', description='Test import this project')
def _testme(cfg: _fig.Configuration):
    cfg.print(f'Importing code worked!')
