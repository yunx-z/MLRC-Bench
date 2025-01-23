""" Helper functions to use in the ingestion program. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import torch
from typing import List, Any, Iterator


def get_torch_gpu_environment() -> List[str]:
    """ Retrieve all the information regarding the GPU environment.

    Returns:
        List[str]: Information of the GPU environment.
    """
    env_info = list()
    env_info.append(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        env_info.append(f"Cuda version: {torch.version.cuda}")
        env_info.append(f"cuDNN version: {torch.backends.cudnn.version()}")
        env_info.append("Number of available GPUs: "
            + f"{torch.cuda.device_count()}")
        env_info.append("Current GPU name: " +
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        env_info.append("Number of available GPUs: 0")
    
    return env_info


def cycle(steps: int,
          iterable: Any) -> Iterator[Any]:
    """ Creates a cycle of the specified number of steps using the specified 
    iterable.

    Args:
        steps (int): Steps of the cycle.
        iterable (Any): Any iterable. In the ingestion program it is used when
            batch data format is selected for training.

    Yields:
        Iterator[Any]: The output of the iterable.
    """
    c_steps = -1
    while True:
        for x in iterable:
            c_steps += 1
            if c_steps == steps:
                break
            yield x
        if c_steps == steps:
            break
            