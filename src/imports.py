from typing import Union, List, Dict, Any, Tuple, Optional, Type, Iterable, Iterator, Self, Callable
import sys, os, shutil
from pathlib import Path
import subprocess, copy
import omnifig as fig
from omnibelt import pformat, where_am_i, colorize, unspecified_argument
from omnibelt import load_json, save_json, load_yaml, save_yaml
import random
from collections import Counter
import json
import h5py as hf
from tabulate import tabulate
from tqdm import tqdm 
# from omniply import Scope, Selection
# from omniply import AbstractGadget
from omniply.gears.errors import GearGrabError
from omniply.apps import DictGadget
from omnilearn import *
from omnilearn.op import *
from omnilearn import autoreg
# from omnilearn import Machine as MachineBase, Trainer as TrainerBase

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
from torch import nn
from torch.nn import functional as F

# from omniply.apps.mechanisms import Mechanism as _Mechanism


# @fig.component('mechanism')
# class Mechanism(fig.Configurable, _Mechanism):
# 	def __init__(self, content: Union[AbstractGadget, list[AbstractGadget], dict[str, AbstractGadget]],
# 				 apply: dict[str, str] | list[str] = None,
# 				 select: dict[str, str] | list[str] = None, **kwargs):
# 		gadgets = content
# 		if isinstance(content, dict):
# 			gadgets = list(content.values())
# 		if not isinstance(gadgets, (list, tuple)):
# 			gadgets = [gadgets]
# 		super().__init__(content=gadgets, apply=apply, select=select, **kwargs)
# 		self._content = content


