import os
import random
import typing

import numpy as np
import torch

DEVICE = torch.device("cuda")
DTYPE = torch.float32
DTYPE_EVAL = torch.float64
EVAL_BATCH_SIZE = 1024


def setup_seeds(
    seed: int,
    deterministic_algorithms: bool = True,
    benchmark_algorithms: bool = False,
):
    # Globally fix seeds in case manual seeding is missing somewhere
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic_algorithms:
        # Enable deterministic (GPU) operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        if benchmark_algorithms:
            raise ValueError("Benchmarking should not be enabled under deterministic algorithms")

    # NB: benchmarking significantly improves training speed in general,
    #  but can reduce performance if things like input shapes change a lot!
    torch.backends.cudnn.benchmark = benchmark_algorithms


def get_setting_seed(
    global_seed: int,
    shadow_model_idx: typing.Optional[int],
    num_shadow: int,
) -> int:
    return global_seed * (num_shadow + 1) + (0 if shadow_model_idx is None else shadow_model_idx + 1)
