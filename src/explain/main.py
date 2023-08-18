# deep learning libraries
import torch
import pandas as pd
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import subgraph

# other libraries
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm.auto import tqdm
from typing import Literal, List, Dict, Type, Tuple

# own modules
from src.train.models import GCN, GAT
from src.utils import set_seed, load_data
from src.explain.methods import (
    Explainer,
    SaliencyMap,
    SmoothGrad,
    DeConvNet,
    GuidedBackprop,
    GNNExplainer,
    PGExplainer,
)


# set seed and device
set_seed(42)
torch.set_num_threads(8)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# static variables
DATA_PATH: str = "data"
LOAD_PATH: str = "models"
METHODS: Dict[str, Type[Explainer]] = {
    "Saliency Map": SaliencyMap,
    "Smoothgrad": SmoothGrad,
    "Deconvnet": DeConvNet,
    "Guided-Backprop": GuidedBackprop,
    "GNNExplainer": GNNExplainer,
    "PGExplainer": PGExplainer,
}
PERCENTAGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DATASETS_NAME: Tuple[Literal["Cora", "CiteSeer", "PubMed"], ...] = (
    "Cora",
    "CiteSeer",
    "PubMed",
)
MODEL_NAMES: Tuple[Literal["gcn", "gat"], ...] = ("gcn", "gat")
SELF_LOOPS: Tuple[bool, bool] = (True, False)


def main() -> None:
    # empty nohup file
    open("nohup.out", "w").close()

    # check device
    print(f"device: {device}")

    # define progress bar
    progress_bar = tqdm(
        range(
            len(SELF_LOOPS) * len(DATASETS_NAME) * len(MODEL_NAMES) * len(METHODS) * 2
        )
    )

    for self_loops in SELF_LOOPS:
        for dataset_name in DATASETS_NAME:
            for model_name in MODEL_NAMES:
                # define dataset
                dataset: InMemoryDataset = load_data(
                    dataset_name, f"{DATA_PATH}/{dataset_name}"
                )

                # load model
                model: torch.nn.Module
                if model_name == "gcn":
                    model = GCN(
                        dataset.num_features, dataset.num_classes, self_loops
                    ).to(device)
                elif model_name == "gat":
                    model = GAT(
                        dataset.num_features, dataset.num_classes, self_loops
                    ).to(device)
                else:
                    raise ValueError("Invalid model_name")
                model.load_state_dict(
                    torch.load(
                        f"{LOAD_PATH}/{dataset_name}_{model_name}_{self_loops}/state_dict.pt"
                    )
                )
                model.eval()

                # pass elements to correct device
                x: torch.Tensor = dataset[0].x.float().to(device)
                edge_index: torch.Tensor = dataset[0].edge_index.long().to(device)
                train_mask: torch.Tensor = dataset[0].train_mask.to(device)
                test_mask: torch.Tensor = dataset[0].test_mask.to(device)
                node_ids: torch.Tensor = torch.arange(x.shape[0]).to(device)
                train_node_ids: torch.Tensor = node_ids[train_mask]
                test_node_ids: torch.Tensor = node_ids[test_mask]

                # compute original outputs
                original_outputs: torch.Tensor = model(x, edge_index).argmax(dim=1)
                original_outputs_probs: torch.Tensor = torch.gather(
                    F.softmax(model(x, edge_index), dim=1),
                    1,
                    original_outputs.unsqueeze(1),
                )

                # init aucs
                aucs: List[List[object]] = []
                aucs_probs: List[List[object]] = []
                last_loyalties: List[List[object]] = []
                last_loyalties_probs: List[List[object]] = []

                # iter over different methods
                method_name: str
                method: Type[Explainer]
                for method_name, method in METHODS.items():
                    for metric in ["loyalty", "inverse loyalty"]:
                        # define explainer
                        explainer: Explainer
                        if method == PGExplainer:
                            explainer = method(model, x, edge_index, train_node_ids)
                        else:
                            explainer = method(model)

                        # define loyalties
                        loyalties: List[float] = [0.0 for _ in PERCENTAGES]
                        loyalties_probs: List[float] = [0.0 for _ in PERCENTAGES]

                        # iter over node ids
                        node_id: int
                        for node_id in test_node_ids.tolist():
                            # compute feature map
                            feature_map: torch.Tensor = explainer.explain(
                                x, edge_index, node_id
                            )

                            # important nodes
                            important_nodes: torch.Tensor = feature_map[feature_map > 0]
                            important_ids: torch.Tensor = node_ids[feature_map > 0]

                            # delete itself
                            important_nodes = important_nodes[important_ids != node_id]
                            important_ids = important_ids[important_ids != node_id]

                            # sort important ids
                            ordered_important_ids: torch.Tensor
                            if metric == "loyalty":
                                _, ordered_important_ids = torch.sort(
                                    important_nodes, descending=True
                                )
                            elif metric == "inverse loyalty":
                                _, ordered_important_ids = torch.sort(
                                    important_nodes, descending=False
                                )

                            # iter over percentages
                            i: int
                            for i in range(len(PERCENTAGES)):
                                # compute filtered important ids
                                filtered_important_ids: torch.Tensor
                                filtered_important_ids = ordered_important_ids[
                                    : int(len(ordered_important_ids) * PERCENTAGES[i])
                                ]

                                # compute filtered ids
                                filtered_ids: torch.Tensor = important_ids[
                                    filtered_important_ids
                                ]

                                # construct keep ids tensor
                                keep_mask: torch.Tensor = torch.ones_like(node_ids)
                                keep_mask[filtered_ids] = 0
                                keep_ids: torch.Tensor = node_ids[
                                    keep_mask == 1
                                ].clone()

                                # filter edges
                                filtered_edge_index: torch.Tensor
                                filtered_edge_index, _ = subgraph(keep_ids, edge_index)

                                # compute outputs
                                outputs: torch.Tensor = model(
                                    x, filtered_edge_index
                                ).argmax(dim=1)
                                outputs_probs: torch.Tensor = torch.gather(
                                    F.softmax(model(x, filtered_edge_index), dim=1),
                                    1,
                                    original_outputs.unsqueeze(1),
                                )

                                # update loyalty and loyalty probabilities
                                if outputs[node_id] == original_outputs[node_id]:
                                    loyalties[i] += 1

                                loyalties_probs[i] += (
                                    torch.abs(
                                        outputs_probs[node_id]
                                        - original_outputs_probs[node_id]
                                    ).item()
                                    / original_outputs_probs[node_id].item()
                                )

                        # scale loyalties
                        for i in range(len(loyalties)):
                            loyalties[i] /= test_node_ids.shape[0]
                            loyalties_probs[i] /= test_node_ids.shape[0]

                        # compute auc
                        auc: float = 0.0
                        for i in range(1, len(loyalties)):
                            auc += (
                                min(loyalties[i - 1], loyalties[i]) * 0.1
                                + (
                                    max(loyalties[i - 1], loyalties[i])
                                    - min(loyalties[i - 1], loyalties[i])
                                )
                                / 2
                                * 0.1
                            )

                        # compute auc probs
                        auc_prob: float = 0.0
                        for i in range(1, len(loyalties_probs)):
                            auc_prob += (
                                min(loyalties_probs[i - 1], loyalties_probs[i]) * 0.1
                                + (
                                    max(loyalties_probs[i - 1], loyalties_probs[i])
                                    - min(loyalties_probs[i - 1], loyalties_probs[i])
                                )
                                / 2
                                * 0.1
                            )

                        # compute last loyalties
                        last_loyalty: float = loyalties[-1]
                        last_loyalty_probs: float = loyalties_probs[-1]

                        # append to lists
                        aucs.append([method_name, metric, auc])
                        aucs_probs.append([method_name, metric, auc_prob])
                        last_loyalties.append([method_name, metric, last_loyalty])
                        last_loyalties_probs.append(
                            [method_name, metric, last_loyalty_probs]
                        )

                        # create dir if it doesn't exist
                        if not os.path.isdir(f"results/graphs/{method_name}/{metric}"):
                            os.makedirs(f"results/graphs/{method_name}/{metric}")

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        plt.plot(PERCENTAGES, loyalties)
                        plt.xlabel("percentage of deleted neighbors [%]")
                        plt.ylabel(f"{metric} [%]")
                        plt.grid()
                        if not self_loops:
                            plt.ylim([-0.03, 1.03])
                        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                        plt.savefig(
                            f"results/graphs/{method_name}/{metric}/"
                            f"{dataset_name}_{model_name}_{self_loops}.pdf"
                        )

                        # create dir if it doesn't exist
                        if not os.path.isdir(
                            f"results/graphs/{method_name}/{metric} probs"
                        ):
                            os.makedirs(f"results/graphs/{method_name}/{metric} probs")

                        # create and save graph for probabilities
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        plt.plot(PERCENTAGES, loyalties_probs)
                        plt.xlabel("percentage of deleted neighbors [%]")
                        plt.ylabel(f"{metric} probabilities [%]")
                        plt.grid()
                        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
                        plt.savefig(
                            f"results/graphs/{method_name}/{metric} probs/"
                            f"{dataset_name}_{model_name}_{self_loops}.pdf"
                        )
                        plt.close()

                        # update progress bar
                        progress_bar.update()

                # define loyalties
                loyalty: float = 0.0
                loyalty_probs: float = 0.0

                # iter over node ids
                for node_id in test_node_ids.tolist():
                    # compute outputs
                    outputs = model(x, edge_index[:, :0]).argmax(dim=1)
                    outputs_probs = torch.gather(
                        F.softmax(model(x, edge_index[:, :0]), dim=1),
                        1,
                        original_outputs.unsqueeze(1),
                    )

                    # update loyalty and loyalty probabilities
                    if outputs[node_id] == original_outputs[node_id]:
                        loyalty += 1

                    loyalty_probs += (
                        torch.abs(
                            outputs_probs[node_id] - original_outputs_probs[node_id]
                        ).item()
                        / original_outputs_probs[node_id].item()
                    )

                # scale
                loyalty /= test_node_ids.shape[0]
                loyalty_probs /= test_node_ids.shape[0]

                # append to last loyalties lists
                last_loyalties.append(["without neighbors", "-", last_loyalty])
                last_loyalties_probs.append(
                    ["without neighbors", "-", last_loyalty_probs]
                )

                # create directory if it doesn't exist
                if not os.path.isdir("results/aucs"):
                    os.makedirs("results/aucs")

                # create directory if it doesn't exist
                if not os.path.isdir("results/aucs_probs"):
                    os.makedirs("results/aucs_probs")

                # create directory if it doesn't exist
                if not os.path.isdir("results/last_loyalties"):
                    os.makedirs("results/last_loyalties")

                # create directory if it doesn't exist
                if not os.path.isdir("results/last_loyalties_probs"):
                    os.makedirs("results/last_loyalties_probs")

                # save lists into dataframes
                pd.DataFrame(aucs, columns=["method_name", "metric", "auc"]).to_csv(
                    f"results/aucs/{dataset_name}_{model_name}_{self_loops}.csv"
                )
                pd.DataFrame(
                    aucs_probs, columns=["method_name", "metric", "auc"]
                ).to_csv(
                    f"results/aucs_probs/{dataset_name}_{model_name}_{self_loops}.csv"
                )
                pd.DataFrame(
                    last_loyalties, columns=["method_name", "metric", "auc"]
                ).to_csv(
                    f"results/last_loyalties/{dataset_name}_{model_name}_{self_loops}.csv"
                )
                pd.DataFrame(
                    last_loyalties_probs, columns=["method_name", "metric", "auc"]
                ).to_csv(
                    f"results/last_loyalties_probs/{dataset_name}_{model_name}_{self_loops}.csv"
                )


if __name__ == "__main__":
    main()
