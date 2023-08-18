# deep learning libraries
import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures

# other libraries
import os
import random
from typing import Literal


def load_data(
    dataset_name: Literal["Cora", "CiteSeer", "PubMed"], save_path: str
) -> InMemoryDataset:
    # get dataset
    dataset: InMemoryDataset = torch_geometric.datasets.Planetoid(
        root=save_path, name=dataset_name, transform=NormalizeFeatures()
    )

    return dataset


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior

    Parameters
    ----------
    seed : int
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
