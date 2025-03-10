from .imports import *
from omnilearn.op import Dataset as _Dataset
from omnilearn.abstract import AbstractEvaluatableDataset
from .pretrained import CLIP

from torchvision.datasets import MNIST as Torchvision_MNIST


class Dataset(_Dataset, AbstractEvaluatableDataset):
    _ds_name = None
    _split_sizes = None
    _split_seed = 1000000007
    def __init__(self, *, dataroot: str = None, split: str = None, eval_split: Union[int, float] = None, **kwargs):
        super().__init__(**kwargs)
        self._dataroot = Path(dataroot)
        if not self._dataroot.exists():
            raise FileNotFoundError(f'Dataroot does not exist: {self._dataroot}')
        self._split = split
        if self._split_sizes is not None:
            assert split in self._split_sizes, f'Invalid split: {split} from {self._split_sizes.keys()}'
        self._size = None if self._split_sizes is None else self._split_sizes.get(split)
        self._original_size = self._size
        self._eval_split = eval_split
        self._selection = None
        if eval_split is not None:
            this_is_eval = abs(eval_split) != eval_split
            eval_split = abs(eval_split)
            if isinstance(eval_split, float):
                assert 0 < eval_split < 1, 'eval_split must be in (0, 1)'
                eval_split = int(eval_split * self._size)
            else:
                assert isinstance(eval_split, int) and 0 < eval_split < self._size, 'eval_split must be an integer in (0, size)'
            self._size = eval_split if this_is_eval else self._size - eval_split

    def _select_indices(self, N: int, split: Union[float, int]):
        this_is_eval = abs(split) != split
        split = abs(split)
        if isinstance(split, float):
            split = int(split * N)
        order = np.random.RandomState(self._split_seed).permutation(N)
        return order[:split] if this_is_eval else order[split:]

    def __repr__(self):
        base = super().__repr__()
        pre, post = base.split('(', 1)
        return f'{pre}[{self.size}]({post}'

    def as_eval(self):
        other = self._as_eval()
        other.gauge_apply(self._gauge)
        return other

    def _as_eval(self, *, split=unspecified_argument, dataroot=unspecified_argument, **kwargs) -> 'Dataset':
        if split is unspecified_argument:
            split = self._split
        if dataroot is unspecified_argument:
            dataroot = self._dataroot
        assert self._eval_split is not None, 'No eval split specified'
        return self.__class__(split=split, dataroot=dataroot, eval_split=-self._eval_split, **kwargs)


    def load(self, *, device: Optional[str] = None) -> Self:
        if self._selection is None and self._eval_split is not None:
            self._selection = self._select_indices(self._original_size, self._eval_split)
        return super().load(device=device)

    @tool('index')
    def get_indices(self, index: np.ndarray[int]) -> np.ndarray[int]:
        if self._selection is not None:
            return self._selection[index]
        return index

    @property
    def dataroot(self) -> Path:
        return self._dataroot


    @property
    def split(self) -> str:
        return self._split
    

    @property
    def size(self) -> int:
        return self._size
    

    @property
    def name(self) -> str:
        return f'{self._ds_name}-{self._split}' if self._split != 'train' else self._ds_name



@fig.component('mnist')
class MNIST(Dataset):
    _ds_name = 'MNIST'
    _val_split = 6000
    @property
    def _split_sizes(self):
        return {'train': 60000-self._val_split, 'val': self._val_split, 'test': 10000} if self._val_split is not None \
            else {'train': 60000, 'test': 10000}

    def __init__(self, split: str = 'train', *, download: bool = True, dataroot='/data/felix/mnist', **kwargs):
        super().__init__(split=split, dataroot=dataroot, eval_split=None, **kwargs)
        self._download = download
        self._dataset = None
        self._image_data = None
        self._label_data = None

    def _as_eval(self, **kwargs) -> 'MNIST':
        assert self._split == 'train', 'Only train split can be converted to eval'
        return self.__class__(split='val', dataroot=self.dataroot, **kwargs)


    def load(self, *, device: str = None) -> Self:
        if self._dataset is None:
            self._dataset = Torchvision_MNIST(self.dataroot, train=self._split != 'test', download=self._download)
            self._image_data = self._dataset.data.view(-1, 1, 28, 28)
            self._label_data = self._dataset.targets
            if self._split != 'test' and self._val_split is not None:
                if self._split == 'train':
                    self._image_data = self._image_data[self._val_split:]
                    self._label_data = self._label_data[self._val_split:]
                else:
                    self._image_data = self._image_data[:self._val_split]
                    self._label_data = self._label_data[:self._val_split]
            if device is not None:
                self._image_data = self._image_data.to(device)
                self._label_data = self._label_data.to(device)
        return super().load(device=device)


    @tool('image')
    def get_images(self, index: np.ndarray) -> torch.Tensor:
        '''returns int8 tensor of shape (N, 28, 28)'''
        return self._image_data[torch.from_numpy(index)]
    @get_images.space
    def image_space(self) -> spaces.Pixels:
        return spaces.Pixels(1, 28, 28, as_bytes=True)
    

    @tool('label')
    def get_labels(self, index: np.ndarray) -> torch.Tensor:
        return self._label_data[torch.from_numpy(index)]
    @get_labels.space
    def label_space(self) -> spaces.Categorical:
        return spaces.Categorical(10)



class Shapes3D(Dataset):
    def __init__(self, split='train', *, dataroot='/data/felix/3dshapes', **kwargs):
        super().__init__(split=split, dataroot=dataroot, **kwargs)
        self._hf_file = None

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._hf_file is None:
            self._hf_file = hf.File(self.dataroot.joinpath('3dshapes.h5'), 'r')
        return super().load(device=device)



class ImageNetBase(Dataset):
    _ds_name = 'ImageNet'
    _split_sizes = {'train': 1281167, 'val': 50000}
    def __init__(self, split='train', *, dataroot='/data/felix/imagenet', **kwargs):
        super().__init__(split=split, dataroot=dataroot, **kwargs)
        self._label_names = None
        self._device = None


    def load(self, *, device: Optional[str] = None) -> Self:
        if self._label_names is None:
            label_names = json.load(self.dataroot.joinpath('classes.json').open())
            label_names = {int(k): v for k, v in label_names.items()}
            self._label_names = label_names
        self._device = device
        return self


    @space('label')
    def label_space(self) -> spaces.Categorical:
        return spaces.Categorical(self._label_names)


    @tool('label_name')
    def get_label_names(self, label: Iterable[int]) -> List[str]:
        if isinstance(label, torch.Tensor):
            label = label.tolist()
        return [self._label_names[l] for l in label]
    


@fig.component('imagenet')
class ImageNet(ImageNetBase):
    def __init__(self, split='train', *, resize=224, **kwargs):
        if isinstance(resize, int):
            resize = (resize, resize)
        super().__init__(split=split, **kwargs)
        if resize[0] != resize[1]:
            raise NotImplemented('Non-square resize not implemented')
        self._resize = resize

        self._img_paths = None

        self._img_transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._img_paths is None:
            dataroot = self.dataroot
            paths = list(dataroot.glob(f'{self._split}/**/*.JPEG'))
            self._img_paths = paths

            labels =  dict(line.split() for line in dataroot.joinpath(f'{self._split}_label').read_text().splitlines())

            key = lambda p: p.name if self._split == 'val' else f'{p.parent.name}/{p.name}'
            labels = np.array([int(labels[key(p)]) for p in paths])
            self._labels = labels
        return super().load(device=device)


    @property
    def name(self) -> str:
        return f'imagenet-{self._split}{"-" + str(self._resize[0]) if self._resize[0] != 224 else ""}'


    @tool('image_path')
    def get_image_paths(self, index: Iterable[int]) -> List[str]:
        return [str(self._img_paths[i]) for i in index]


    @tool('image_loc')
    def get_image_locs(self, image_path: Iterable[int]) -> List[str]:
        # get path relative to root
        return [str(Path(p).relative_to(self.dataroot)) for p in image_path]


    @tool('image')
    def load_images(self, image_path: Iterable[str]) -> torch.Tensor:
        return torch.stack([self._img_transforms(Image.open(p)) for p in image_path]).to(self._device)
    @load_images.space
    def image_space(self) -> spaces.Pixels:
        return spaces.Pixels(3, *self._resize)


    @tool('label')
    def get_labels(self, index: Iterable[int]) -> torch.Tensor:
        return torch.from_numpy(self._labels[index]).to(self._device)
    


@fig.component('imagenet-clip')
class ImageNetCLIP(ImageNetBase):
    def __init__(self, vit='ViT-B/32', **kwargs):
        super().__init__(**kwargs)
        self._vit_model = CLIP.parse_vit_model_name(vit)
        self._hf_path = self.dataroot.joinpath('embeddings', f'imagenet-{self._split}-clip-{self._vit_model}.h5')
        if not self._hf_path.exists():
            raise FileNotFoundError(f'Could not find file: {self._hf_path}')

        self._hf_data = None
    

    @property
    def name(self) -> str:
        return f'imagenet-clip-{self._split}-{self._vit_model}'
    

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._hf_data is None:
            self._hf_data = hf.File(self._hf_path, 'r')
        return super().load(device=device)


    @tool('image_path')
    def get_image_paths(self, index: np.ndarray[int]) -> List[str]:
        names = self._hf_data['image_loc'][index].tolist()
        return [str(self.dataroot / name.decode()) for name in names]


    @tool('image')
    def load_images(self, image_path: Iterable[str]) -> List[Image.Image]:
        return [Image.open(p) for p in image_path]
    

    @tool('embedding')
    def get_embedding(self, index: np.ndarray[int]) -> torch.Tensor:
        try:
            samples = self._hf_data['embedding'][index]
        except TypeError:
            print(f'WARNING: index bulk load failed')
            samples = []
            for i in index:
                samples.append(self._hf_data['embedding'][i])
            samples = np.array(samples)
        return torch.from_numpy(samples).to(self._device)
    @get_embedding.space
    def embedding_space(self) -> spaces.Vector:
        return spaces.Vector(512)


    @tool('label')
    def get_labels(self, index: np.ndarray[int]) -> torch.Tensor:
        try:
            samples = self._hf_data['label'][index]
        except TypeError:
            print(f'WARNING: index bulk load failed')
            samples = []
            for i in index:
                samples.append(self._hf_data['label'][i])
            samples = np.array(samples)
        return torch.from_numpy(samples).to(self._device)
    


class COCOBase(Dataset):
    # _ds_name = 'COCO'
    _split_sizes = {'train': 118287, 'val': 5000, 'test': 40670, 'unlabeled': 123403}
    def __init__(self, split='train', *, dataroot='/data2/COCO', annoroot=None, **kwargs):
        if annoroot is None:
            annoroot = Path(dataroot) / 'annotations'
        super().__init__(split=split, dataroot=dataroot, **kwargs)
        self._annoroot = Path(annoroot)
        self._device = None
        self._data_info = None
        self._transforms = None
        self._class_order = None
        self._class_names = None

    
    def load(self, *, device: Optional[str] = None) -> Self:
        if self._data_info is None:
            assert self.dataroot.joinpath(f'{self._split}2017').exists(), f'Data root does not exist: {self.dataroot.joinpath(f"{self._split}2017")}'
            assert self._annoroot.exists(), f'Annotation root does not exist: {self._annoroot}'
            info = json.load(self._annoroot.joinpath(f'instances_{self._split}2017.json').open())
            self._data_info = info
            image_ids = {img['id'] for img in info['images']}
            anns = {}
            for ann in info['annotations']:
                anns.setdefault(ann['image_id'], []).append(ann)
            assert all(k in image_ids for k in anns.keys()), 'Not all annotations have corresponding images'
            # assert self.size == len(info['images']), f'Expected {self.size} images, got {len(anns["images"])}'
            self._annotations = anns
            self._label_names = {cat['id']: cat['name'] for cat in info['categories']}
            self._super_categories = {cat['id']: cat['supercategory'] for cat in info['categories']}
            
            order = list(self._label_names.keys())
            names = [self._label_names[k] for k in order]
            self._class_order = np.array(order)
            self._class_names = np.array(names)
        self._device = device
        return super().load(device=device)


    @tool('annotations')
    def get_annotations(self, image_id: Iterable[str]) -> List[List[Dict[str, Any]]]:
        return [self._annotations.get(i, []) for i in image_id]
    
    @tool('categories')
    def get_categories(self, annotations: Iterable[List[Dict[str, Any]]]) -> List[str]:
        return [[self._label_names[ann['category_id']] for ann in anns] for anns in annotations]

    
    @tool('class_count')
    def count_classes(self, annotations: Iterable[List[Dict[str, Any]]]) -> torch.Tensor:
        batch = []
        for anns in annotations:
            if len(anns):
                batch.append(sum([self._class_order == ann['category_id'] for ann in anns]))
            else:
                batch.append(np.zeros(len(self._class_order)))
        return torch.from_numpy(np.stack(batch)).to(self._device)
    @count_classes.space
    def class_count_space(self) -> spaces.Vector:
        return spaces.Vector(len(self._class_order), dtype=torch.int)
    
    @tool('class_presence')
    def get_class_presence(self, class_count: Iterable[torch.Tensor]) -> torch.Tensor:
        return class_count.gt(0)
    @get_class_presence.space
    def class_presence_space(self) -> spaces.BooleanCategorical:
        return spaces.BooleanCategorical(self._class_names.tolist())



@fig.component('raw-coco')
class RawCOCO(COCOBase):
    _ds_name = 'rawcoco'

    @tool('image_path')
    def get_image_paths(self, index: Iterable[int]) -> List[str]:
        img_paths = [self.dataroot.joinpath(f'{self._split}2017', self._data_info['images'][i]['file_name']) for i in index]
        return [str(p) for p in img_paths]

    @tool('rawimage')
    def load_images(self, image_path: Iterable[str]) -> List[Image.Image]:
        return [Image.open(p) for p in image_path]
    
    @tool('image_id')
    def get_image_id(self, index: Iterable[str]) -> List[int]:
        return [self._data_info['images'][i]['id'] for i in index]

    @tool('rawimage_height')
    def get_rawimage_height(self, index: Iterable[str]) -> np.ndarray[int]:
        return np.array([self._data_info['images'][i]['height'] for i in index])
    
    @tool('rawimage_width')
    def get_rawimage_width(self, index: Iterable[str]) -> np.ndarray[int]:
        return np.array([self._data_info['images'][i]['width'] for i in index])
    


@fig.component('raw-coco-captions')
class RawCOCOCaptions(COCOBase):
    _split_sizes = {'train': 591753, 'val': 25014}
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._caption_data = None

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._caption_data is None:
            info = json.load(self._annoroot.joinpath(f'captions_{self._split}2017.json').open())
            self._caption_data = info['annotations']
        return super().load(device=device)

    @tool('caption')
    def get_captions(self, index: Iterable[int]) -> List[str]:
        return [self._caption_data[i]['caption'] for i in index]
    
    @tool('caption_id')
    def get_caption_id(self, index: Iterable[int]) -> List[int]:
        return [self._caption_data[i]['id'] for i in index]
    
    @tool('image_id')
    def get_caption_image_id(self, index: Iterable[int]) -> List[int]:
        return [self._caption_data[i]['image_id'] for i in index]



class H5_Dataset(_Dataset):
    def __init__(self, *args, preload: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._preload = preload
        self._hf_file = None
        self._hf_data = None

    def _as_eval(self, **kwargs) -> 'Dataset':
        return super()._as_eval(preload=self._preload, **kwargs)

    def _load_hf_samples(self, key: str, index: np.ndarray[int], device: Optional[str] = None) -> torch.Tensor:
        if self._hf_data is not None and key in self._hf_data:
            return self._hf_data[key][torch.from_numpy(index)]
        try:
            samples = self._hf_file[key][index]
        except TypeError:
            # print(f'WARNING: index bulk load failed')
            samples = []
            for i in index:
                samples.append(self._hf_file[key][i])
            samples = np.array(samples)
        return torch.as_tensor(samples, device=device)

    def _load_hf_file(self, path: Path) -> hf.File:
        if not path.exists():
            raise FileNotFoundError(f'Could not find file: {path}')
        return hf.File(path, 'r')



_COCO_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush']
    


@fig.component('coco-text')
class COCOCaptions(H5_Dataset, Dataset):
    _ds_name = 'cococap'
    _split_sizes = {'train': 591753, 'val': 25014}
    _label_class_order = _COCO_classes

    def __init__(self, split='train', *, dataroot='/data/felix/coco', embedding='bert-cls', **kwargs):
        super().__init__(split=split, dataroot=dataroot, **kwargs)
        self._hf_path = Path(dataroot).joinpath('embeddings', f'{split}-{embedding}.h5')

    def _as_eval(self, **kwargs) -> 'Dataset':
        embedding = self._hf_path.stem.split('-', 1)[1]
        other = super()._as_eval(embedding=embedding, **kwargs)
        assert other._hf_path == self._hf_path, 'Eval dataset does not have the same embedding'
        return other
        

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._hf_file is None:
            self._hf_file = self._load_hf_file(self._hf_path)
            shape = self._hf_file['embedding'].shape
            # assert shape == (self.size, 768), f'Expected {(self.size, 768)} samples, got {shape}'
            if self._preload:
                data = {}
                data['embedding'] = torch.as_tensor(self._hf_file['embedding'][()], device=device)
                data['class_presence'] = torch.as_tensor(self._hf_file['class_presence'][()], device=device)
                self._hf_data = data
            self._device = device
        return super().load(device=device)
    
    def _select_indices(self, N: int, split: Union[float, int]):
        imgs_ids = self._hf_file['image_id'][()]
        uniq = np.unique(imgs_ids)
        this_is_eval = abs(split) != split
        split = abs(split)
        if isinstance(split, float):
            split = int(split * len(uniq))
        all_inds = np.arange(N)
        order = np.random.RandomState(self._split_seed).permutation(len(uniq))
        picks = uniq[order[:split]] if this_is_eval else uniq[order[split:]]
        picks.sort()
        inds = np.array([i for i in all_inds if (ind := np.searchsorted(picks, imgs_ids[i])) != len(picks) 
                                                and picks[ind] == imgs_ids[i]])
        self._size = len(inds)
        return inds

    @tool('embedding')
    def get_text_embedding(self, index: np.ndarray[int]) -> torch.Tensor:
        return self._load_hf_samples('embedding', index, device=self._device)
    @get_text_embedding.space
    def embedding_space(self) -> spaces.Vector:
        return spaces.Vector(768)

    @tool('label')
    def get_labels(self, index: np.ndarray[int]) -> torch.Tensor:
        return self._load_hf_samples('class_presence', index, device=self._device)
    @get_labels.space
    def label_space(self) -> spaces.BooleanCategorical:
        return spaces.BooleanCategorical(self._label_class_order)

    @tool('image_id')
    def get_image_id(self, index: np.ndarray[int]) -> np.ndarray[int]:
        return self._load_hf_samples('image_id', index).numpy()
    
    @tool('caption_id')
    def get_caption_id(self, index: np.ndarray[int]) -> np.ndarray[int]:
        return self._load_hf_samples('caption_id', index).numpy()
    
    @tool('caption')
    def get_captions(self, index: np.ndarray[int]) -> List[str]:
        return [self._hf_file['caption'][i].decode() for i in index]


@fig.component('coco')
class COCO(COCOCaptions):
    _ds_name = 'coco'
    def __init__(self,  split='train', *, force_size: int = None, image_dataset='vit_base_patch32_224', gap=None, **kwargs):
        if gap is None:
            gap = {}
        gap['embedding'] = 'text_features'
        super().__init__(split=split, gap=gap, **kwargs)
        self._image_dataset = image_dataset
        self._image_indices = None
        self._force_size = force_size

    @property
    def size(self) -> int:
        return super().size if self._force_size is None else self._force_size

    def _as_eval(self, **kwargs) -> 'Dataset':
        return super()._as_eval(image_dataset=self._image_dataset, **kwargs)
    
    def load(self, *, device: Optional[str] = None) -> Self:
        super().load(device=device)

        if self._image_dataset is not None:
            if isinstance(self._image_dataset, str):
                self._image_dataset = SimpleCOCO(split=self._split, name=self._image_dataset, dataroot=self.dataroot)
            
            self._image_dataset.load(device=device)

            image_ids = self._image_dataset.get_image_id(np.arange(self._image_dataset.size))

            cap_image_ids = self._hf_file['image_id'][()]

            index_map = {img_id: i for i, img_id in enumerate(image_ids)}
            self._image_indices = np.array([index_map[img_id] for img_id in cap_image_ids])

        if self._force_size is not None and (self._eval_split is None or self._eval_split > 0):
            self._size = self._force_size

        return self

    @tool('image')
    def load_images(self, index: np.ndarray[int]) -> torch.Tensor:
        assert self._image_dataset is not None, 'Image dataset not loaded'
        return self._image_dataset.load_images(self._image_indices[index])
    
    @tool('image_features')
    def get_image_features(self, index: np.ndarray[int]) -> torch.Tensor:
        assert self._image_dataset is not None, 'Image dataset not loaded'
        return self._image_dataset.get_embedding(self._image_indices[index])
    @get_image_features.space
    def image_features_space(self) -> spaces.Vector:
        return spaces.Vector(768)
        


@fig.component('simple-coco')
class SimpleCOCO(H5_Dataset, Dataset):
    _ds_name = 'coco'
    _split_sizes = {'train': 118287, 'val': 5000}
    _label_class_order = _COCO_classes

    def __init__(self, *args, name='vit_base_patch32_224', demean: Union[float, bool, None] = True, preload_images: bool = False, dataroot = '/data/felix/coco', **kwargs):
        super().__init__(*args, dataroot=dataroot, **kwargs)
        self._hf_path = self.dataroot.joinpath('embeddings', f'{self._split}-{name}.h5')
        self._demean = demean
        self._preload_images = preload_images

    def _as_eval(self, **kwargs) -> 'Dataset':
        name = self._hf_path.stem.split('-')[1]
        return super()._as_eval(name=name, demean=self._demean, preload_images=self._preload_images, **kwargs)

    def load(self, *, device: Optional[str] = None) -> Self:
        if self._hf_file is None:
            self._hf_file = self._load_hf_file(self._hf_path)
            shape = self._hf_file['embedding'].shape
            # assert shape == (self.size, 768), f'Expected {(self.size, 768)} samples, got {shape}'
            if self._preload:
                data = {}
                if self._preload_images:
                    data['image'] = torch.as_tensor(self._hf_file['image'][()], device=device)
                else:
                    data['embedding'] = torch.as_tensor(self._hf_file['embedding'][()], device=device)
                data['class_presence'] = torch.as_tensor(self._hf_file['class_presence'][()], device=device)
                self._hf_data = data
            self._device = device
        return super().load(device=device)
    
    @tool('image')
    def load_images(self, index: np.ndarray[int]) -> torch.Tensor:
        raw = self._load_hf_samples('image', index, device=self._device)
        if self._demean:
            raw = raw.float()
            raw.div_(255).sub_(0.5).div_(0.5)
            if not isinstance(self._demean, bool):
                raw.div_(self._demean)
        return raw

    @tool('image_id')
    def get_image_id(self, index: np.ndarray[int]) -> np.ndarray[int]:
        return self._load_hf_samples('image_id', index).numpy()

    @tool('embedding')
    def get_embedding(self, index: np.ndarray[int]) -> torch.Tensor:
        return self._load_hf_samples('embedding', index, device=self._device)
    @get_embedding.space
    def embedding_space(self) -> spaces.Vector:
        return spaces.Vector(768)

    @tool('instances')
    def get_instances(self, index: np.ndarray[int]) -> torch.Tensor:
        return self._load_hf_samples('class_count', index, device=self._device)
    @get_instances.space
    def instances_space(self) -> spaces.Vector:
        return spaces.Vector(len(self._label_class_order), dtype=torch.int)


    @tool('label')
    def get_labels(self, index: np.ndarray[int]) -> torch.Tensor:
        return self._load_hf_samples('class_presence', index, device=self._device)
    @get_labels.space
    def label_space(self) -> spaces.BooleanCategorical:
        return spaces.BooleanCategorical(self._label_class_order)
    


@fig.component('preload-coco')
class PreloadCOCO(COCOBase):
    def __init__(self, *args, label_key: str, resize=248, crop=224, 
                 shortcircuit=False, preload_batch_size=None,
                 demean: Union[float, bool, None]=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._demean = demean
        self._demo_load = shortcircuit
        self._label_key = label_key
        self._preload_batch_size = preload_batch_size
        self._resize = resize
        self._crop = crop
        self._imgs = None
        self._labels = None
    
    
    def load(self, *, device = None, ds_settings={}):
        out = super().load(device=device)
        if self._imgs is None:
            base = RawCOCO(dataroot=self.dataroot, split=self.split, annoroot=self._annoroot, **ds_settings)
            base.prepare(device=device).mechanize()
            self._label_space = base.mechanics()[self._label_key]
            tfm = PrepareRawImage(resize=self._resize, crop=self._crop).prepare(device=device)

            imgs = []
            labels = []

            for batch in base.iterate(self._preload_batch_size or max(100, base.size // 100), pbar_desc='Preloading images + labels'):
                imgs.append(tfm.prepare_image(batch['rawimage']))
                labels.append(batch[self._label_key])
                if self._demo_load:
                    print(f'WARNING: Short-circuiting preload for demo')
                    break
            
            self._imgs = torch.cat(imgs)
            self._labels = torch.cat(labels)
            if self._size != self._imgs.size(0):
                print(f'WARNING: Expected {self._size} images, got {self._imgs.size(0)}')
            self._size = self._imgs.size(0)

        return out
    

    @tool('image')
    def load_image(self, index: Iterable[int]) -> torch.Tensor:
        imgs = self._imgs[torch.as_tensor(index)]
        if self._demean:
            imgs = imgs.float()
            imgs.div_(255).sub_(0.5).div_(0.5)
            if not isinstance(self._demean, bool):
                imgs.div_(self._demean)
        return imgs
    @load_image.space
    def image_space(self) -> spaces.Pixels:
        return spaces.Pixels(3, self._crop, self._crop, as_bytes=not self._demean,
                             lower=-1*float(self._demean) if self._demean else 0,
                             upper=1*float(self._demean) if self._demean else 255)
    

    @tool('label')
    def get_labels(self, index: Iterable[int]) -> torch.Tensor:
        return self._labels[torch.as_tensor(index)]#.to(self._device)
    @get_labels.space
    def label_space(self) -> spaces.BooleanCategorical:
        return self._label_space
    

coco_to_audioset = {
    'person': ['Human sounds', 'Human voice', 'Speech'],
    'bicycle': ['Bicycle bell'],
    'car': ['Car', 'Vehicle horn, car horn, honking'],
    'motorcycle': ['Motorcycle'],
    'airplane': ['Aircraft', 'Jet engine', 'Fixed-wing aircraft, airplane'],
    'bus': ['Bus'],
    'train': ['Train', 'Train whistle', 'Train horn'],
    'truck': ['Truck'],
    'boat': ['Boat, Water vehicle', 'Motorboat'],
    'traffic light': [],  # ['Traffic noise, roadway noise']
    'fire hydrant': [],  # ['Water tap, faucet', 'Gurgling']
    'stop sign': [],  # ['Traffic noise, roadway noise']
    'parking meter': [],  # ['Coins dropping', 'Metallic sounds']
    'bench': [],  # ['Outdoor ambiance', 'Furniture creak']
    'bird': ['Bird', 'Bird vocalization, bird call, bird song'],
    'cat': ['Cat', 'Meow', 'Purr'],
    'dog': ['Dog', 'Bark', 'Howl'],
    'horse': ['Horse', 'Neigh, whinny'],
    'sheep': ['Sheep', 'Bleat'],
    'cow': ['Cattle, bovinae', 'Moo'],
    'elephant': [],  # ['Roaring cats (lions, tigers)', 'Wild animals']
    'bear': [],  # ['Wild animals', 'Roaring']
    'zebra': [],  # ['Wild animals', 'Running']
    'giraffe': [],  # ['Wild animals']
    'backpack': [],  # ['Zipper (clothing)', 'Fabric rustling']
    'umbrella': [],  # ['Rustling leaves', 'Wind']
    'handbag': [],  # ['Zipper (clothing)', 'Fabric rustling']
    'tie': [],  # ['Fabric rustling']
    'suitcase': [],  # ['Zipper (clothing)', 'Packing tape']
    'frisbee': [],  # ['Whoosh, swoosh, swish']
    'skis': [],  # ['Sliding on snow', 'Whoosh']
    'snowboard': [],  # ['Sliding on snow']
    'sports ball': ['Basketball bounce'],
    'kite': [],  # ['Wind', 'Flapping']
    'baseball bat': ['Sports impact sounds'],
    'baseball glove': [],  # ['Sports impact sounds']
    'skateboard': ['Skateboard'],
    'surfboard': [],  # ['Waves, surf']
    'tennis racket': [],  # ['Sports impact']
    'bottle': [],  # ['Glass shatter', 'Liquid pouring']
    'wine glass': [],  # ['Glass shatter']
    'cup': [],  # ['Glass shatter', 'Liquid pouring']
    'fork': [],  # ['Cutlery, silverware']
    'knife': [],  # ['Cutlery, silverware']
    'spoon': [],  # ['Cutlery, silverware']
    'bowl': [],  # ['Dishes, pots, and pans']
    'banana': [],  # ['Chewing, mastication']
    'apple': [],  # ['Chewing, mastication']
    'sandwich': [],  # ['Chewing, mastication']
    'orange': [],  # ['Chewing, mastication']
    'broccoli': [],  # ['Chewing, mastication']
    'carrot': [],  # ['Chewing, mastication']
    'hot dog': [],  # ['Chewing, mastication']
    'pizza': [],  # ['Chewing, mastication']
    'donut': [],  # ['Chewing, mastication']
    'cake': [],  # ['Chewing, mastication']
    'chair': [],  # ['Furniture creak']
    'couch': [],  # ['Furniture creak']
    'potted plant': [],  # ['Outdoor ambiance', 'Rustling leaves']
    'bed': [],  # ['Furniture creak']
    'dining table': [],  # ['Dishes, pots, and pans']
    'toilet': ['Toilet flush'],
    'tv': ['Television'],
    'laptop': ['Typing', 'Computer keyboard'],
    'mouse': ['Mouse'],
    'remote': [],  # ['Electronic device beep']
    'keyboard': ['Computer keyboard', 'Typing'],
    'cell phone': ['Telephone bell ringing'],
    'microwave': ['Microwave oven'],
    'oven': [],  # ['Domestic sounds']
    'toaster': [],  # ['Kitchen appliance']
    'sink': ['Sink (filling or washing)'],
    'refrigerator': [],  # ['Domestic sounds']
    'book': [],  # ['Page turn']
    'clock': ['Clock', 'Tick-tock'],
    'vase': [],  # ['Glass shatter']
    'scissors': ['Scissors'],
    'teddy bear': [],  # ['Toys']
    'hair drier': ['Hair dryer'],
    'toothbrush': ['Toothbrush', 'Electric toothbrush']
}


