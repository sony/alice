from .imports import *
from omniply import GrabError
from omnilearn import autoreg#, VizMechanism, VizBatch
# from .trainers import Trainer
from .util import set_default_device



@fig.script('train')
def train(cfg: fig.Configuration):
    device = set_default_device(cfg.pull('device', None, silent=True))
    cfg.push('device', str(device))

    record_step = cfg.pull('record-step', False)
    if record_step:
        Dataset._Batch = VizBatch
        # Trainer._Batch = VizBatch
        fig.component('mechanism')(VizMechanism)
        cfg.push('event.monitor.freqs', {})

    dataset: Dataset = cfg.pull('dataset')

    cfg.push('trainer._type', 'trainer', overwrite=False, silent=True)
    cfg.push('planner._type', 'planner', overwrite=False, silent=True)
    cfg.push('reporter._type', 'reporter', overwrite=False, silent=True)
    trainer: Trainer = cfg.pull('trainer')

    if record_step:
        targets = cfg.pull('grab', None)
        if isinstance(targets, str):
            targets = [targets]

        system = trainer.setup(dataset)
        batch = next(trainer.loop(60, system=system))
        try:
            if targets is None:
                trainer.learn(batch)
            else:
                for target in targets:
                    batch.grab(target)
            # batch['probe']
            # batch['rec_accuracy']

        except GrabError:
            pass
        
        print()
        print(batch.report(**cfg.pull('report-settings', {})))
    else:
        trainer.fit(dataset)
    
    return trainer



@fig.script('collect')
def collect(cfg: fig.Configuration):
    device = set_default_device(cfg.pull('device', None, silent=True))
    cfg.push('device', str(device))

    record_step = cfg.pull('record-step', False)
    if record_step:
        Dataset._Batch = VizBatch
        # Trainer._Batch = VizBatch
        fig.component('mechanism')(VizMechanism)
        
    show_pbar = cfg.pull('pbar', True)

    dataset: Dataset = cfg.pull('dataset')

    env: dict[str, Machine] = cfg.pull('env', {})
    assert 'dataset' not in env, 'dataset is a reserved name for the dataset'

    path = cfg.pull('path', f'{dataset.name}-{"_".join(getattr(v, "name", k) for k,v in env.items())}.h5')
    path = pformat(path, dataset=dataset, **env)
    assert path.endswith('.h5'), f'path must end with .h5, got {path}'
    path = Path(path).expanduser().absolute()
    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    products = cfg.pull('products') # gizmo -> name
    if isinstance(products, str):
        products = [products]
    if isinstance(products, (list, tuple)):
        products = {product: product for product in products}
    assert len(products), 'No products specified'
    
    # if path.exists() and not cfg.pull('overwrite', False):
    #     raise FileExistsError(f'File already exists: {path}')
    storage = {} if record_step else hf.File(path, 'a')

    verifications = {name for name in storage.keys() if name in products}
    todo = [gizmo for gizmo, name in products.items() if name not in verifications]

    batch_size = cfg.pulls('batch-size', 'bs', default=max(100,dataset.suggest_batch_size(target_iterations=1000, prefer_power_of_two=False)))

    # print(f'Components: ', ', '.join(env.keys()))

    system = Structured(dataset, *env.values())
    system.mechanize()
    dataset.prepare(device=device)
    for e in env.values():
        e.prepare(device=device)

    tbl = []
    for name, machine in env.items():
        tbl.append([name, str(machine)])
    print(f'Machines:')
    print(tabulate(tbl, headers=['Name', 'Component']))

    print(f'{dataset.size} samples with batch size {batch_size}')
    print(f'Products:')
    print(tabulate([[gizmo, name, name in verifications, system.gives(gizmo)] 
                    for gizmo, name in products.items()], 
                    headers=['Product', 'Saved as', 'Already Exists', 'Available']))
    print(f'Verifications: {verifications}')
    
    print(f'Collecting {dataset.name} with {len(env)} machine/s to {path}')

    if record_step:
        targets = cfg.pull('grab')
        if isinstance(targets, str):
            targets = [targets]
        
        batch = dataset.batch(batch_size).extend(env.values())
        
        try:
            for target in targets:
                batch.grab(target)
        except GrabError:
            pass
        
        print()
        print(batch.report(**cfg.pull('report-settings', {})))
        return

    def format_raw_data(raw):
        if isinstance(raw, torch.Tensor):
            raw = raw.cpu().numpy()
        if isinstance(raw, list):
            raw = np.array(raw)
        return raw

    with torch.no_grad():
        for batch in dataset.iterate(batch_size, *env.values(), shuffle=False, show_pbar=show_pbar, count_samples=True):
            if len(verifications):
                for verification in verifications:
                    current = format_raw_data(batch[verification])
                    expected = storage[verification][batch['index']]
                    assert np.allclose(current, expected), f'Verification failed for {verification}'
                verifications.clear()
            for product in todo:
                index = batch['index']
                raw = batch[product]
                data = format_raw_data(raw)
                assert len(data) == batch.size, f'Expected len({product}) == {batch.size}, got {len(data)}'
                name = products[product]
                if name not in storage:
                    shape = (dataset.size,) + data.shape[1:]
                    dtype = data.dtype
                    if np.issubdtype(dtype, np.str_):
                        dtype = hf.string_dtype(encoding='utf-8')
                    storage.create_dataset(name, shape, dtype=dtype)
                storage[name][index] = data

    storage.close()

    print(f'Collection of {dataset.size} samples complete. Stored to {path}')


