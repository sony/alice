from .imports import *
from .errors import NoFreeGPUError

def repo_root():
    return Path(__file__).parent.parent

def data_root():
    path = Path(os.getenv('DATAROOT', repo_root() / 'data'))
    if not path.exists():
        path.mkdir(parents=True)
    return path


def get_free_memory_from_nvidia_smi():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )
    # Parse the output
    free_memory = [int(x) for x in result.stdout.strip().split('\n')]
    return free_memory

def get_utilization_from_nvidia_smi():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )
    # Parse the output
    utilization = [int(x) for x in result.stdout.strip().split('\n')]
    return utilization

@fig.autocomponent('best-device')
def get_best_gpu_device(memory=20, free=False, silent=False) -> 'torch.device':
    '''
    This function selects the best GPU device based on the memory and utilization.
    :param memory: Minimum memory threshold in GB
    :param free: If True, only consider GPUs with >80 GB free memory and 0% utilization
    :param silent: If True, don't print the selected device
    :return: The best GPU device

    ref: https://chatgpt.com/c/670e0ae8-7850-8005-bd53-cd1d85795915
    '''
    env = where_am_i()
    import torch
    if torch.cuda.is_available():
        # Get GPU utilization and free memory from nvidia-smi
        utilization = get_utilization_from_nvidia_smi()
        free_memory = get_free_memory_from_nvidia_smi()
        
        # Convert memory threshold from GB to MB
        memory_threshold_mb = memory * 1024
        
        if free:
            # We're looking for a GPU with more than 80 GB free and 0% utilization
            for i in range(len(free_memory)):
                if free_memory[i] > 80 * 1024 and utilization[i] == 0:
                    if not silent:
                        print(f"Selecting CUDA device {i} with {free_memory[i]} MiB free memory and {utilization[i]}% utilization")
                    return torch.device(f"cuda:{i}")
            # If no such GPU is found, raise an exception
            raise NoFreeGPUError("No GPU with >80 GB free and 0% utilization is available.")
        
        # Normal selection logic: find GPUs that meet the memory threshold and sort by utilization
        candidates = [(i, free_memory[i], utilization[i]) for i in range(len(free_memory)) if free_memory[i] >= memory_threshold_mb]
        
        if candidates:
            # Sort by utilization (lower is better)
            best_device = sorted(candidates, key=lambda x: x[2])[0][0]
            if not silent:
                print(f"Selecting CUDA device {best_device} with {free_memory[best_device]} MiB free memory and {utilization[best_device]}% utilization")
            return torch.device(f"cuda:{best_device}")
        else:
            # No device meets the memory requirement, pick the one with the most free memory
            best_device = free_memory.index(max(free_memory))
            if not silent:
                print(f"Selecting CUDA device {best_device} with {free_memory[best_device]} MiB free memory and {utilization[best_device]}% utilization")
            return torch.device(f"cuda:{best_device}")
    else:
        if not silent:
            print("No CUDA devices available, using CPU")
        return torch.device("cpu")


def set_default_device(device: Union[None, int, str, 'torch.device'] = None, **kwargs) -> 'torch.device':
    import torch
    if device is None:
        device = get_best_gpu_device(**kwargs)
    if isinstance(device, int):
        device = torch.device(f"cuda:{device}")
    elif isinstance(device, str):
        device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


















