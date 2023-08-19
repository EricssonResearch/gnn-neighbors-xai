# deep lerning libraries
import torch
from torchmetrics.classification import MulticlassAccuracy
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import InMemoryDataset

# other libraries
import os
from tqdm.auto import tqdm
from typing import Literal

# own modules
from src.utils import set_seed, load_data
from src.train.models import GCN, GAT

# set seed and define device
set_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define static variables
DATA_PATH = "data"
SAVE_PATH = "models"


def main() -> None:
    # define variables
    dataset_name: Literal["Cora", "CiteSeer", "PubMed"] = "Cora"
    model_name: Literal["gcn", "gat"] = "gcn"
    self_loops: bool = True

    # define hyperparameters
    lr: float = 1e-3
    epochs: int = 1000

    # define name and tensorboard writer
    name: str = f"{dataset_name}_{model_name}_{self_loops}"
    writer: SummaryWriter = SummaryWriter(f"runs/{name}")

    # define dataset
    dataset: InMemoryDataset = load_data(dataset_name, f"{DATA_PATH}/{dataset_name}")

    # define model
    model: torch.nn.Module
    if model_name == "gcn":
        model = GCN(dataset.num_features, dataset.num_classes, self_loops).to(device)
    elif model_name == "gat":
        model = GAT(dataset.num_features, dataset.num_classes, self_loops).to(device)
    else:
        raise ValueError("Invalid model_name")

    # define loss
    loss: torch.nn.Module = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # define metrics
    accuracy: torch.nn.Module = MulticlassAccuracy(dataset.num_classes).to(device)

    # pass elements to correct device
    x: torch.Tensor = dataset[0].x.float().to(device)
    edge_index: torch.Tensor = dataset[0].edge_index.long().to(device)
    y: torch.Tensor = dataset[0].y.long().to(device)
    train_mask: torch.Tensor = dataset[0].train_mask.to(device)
    val_mask: torch.Tensor = dataset[0].val_mask.to(device)

    # iter over epochs
    epoch: int
    for epoch in tqdm(range(epochs)):
        # activate train mode
        model.train()

        # compute outputs and loss
        outputs: torch.Tensor = model(x, edge_index)
        loss_value = loss(outputs[train_mask, :], y[train_mask])

        # optimize
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # writer on tensorboard
        writer.add_scalar("loss/train", loss_value.item(), epoch)
        writer.add_scalar(
            "accuracy/train",
            accuracy(outputs[train_mask, :], y[train_mask]).item(),
            epoch,
        )

        # activate eval mode
        model.eval()

        # decativate the gradient
        with torch.no_grad():
            # compute outputs
            outputs = model(x, edge_index)

            # write on tensorboard
            writer.add_scalar(
                "accuracy/val",
                accuracy(outputs[val_mask, :], y[val_mask]).item(),
                epoch,
            )

    # create dirs to save model
    if not os.path.exists(f"{SAVE_PATH}/{name}"):
        os.makedirs(f"{SAVE_PATH}/{name}")

    # save model
    model = model.cpu()
    torch.save(model.state_dict(), f"{SAVE_PATH}/{name}/state_dict.pt")


if __name__ == "__main__":
    main()
