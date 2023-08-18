# deep learning libraries
import torch
import torch_geometric
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle

# other libraries
import copy
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Callable
from abc import ABC, abstractmethod

# own modules
from src.explain.algorithms import PGExplainerAlgorithm


class Explainer(ABC):
    def __init__(self, model: torch.nn.Module, *args):
        """
        Constructor for Explainer class

        Args:
            model: pytorch model
        """

        self.model = model

    @abstractmethod
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        pass


class DeConvNet(Explainer):
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        # call super class constructor
        super().__init__(copy.deepcopy(model))

        # register hooks
        self.register_hooks()

    def register_hooks(self) -> None:
        # define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> Tuple[torch.Tensor]:
            # compute
            new_grad_out: torch.Tensor = F.relu(grad_out[0])

            return (new_grad_out,)

        # define hooks variables
        backward_hook: Callable = backward_hook_fn

        # get modules
        modules: List[Tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(backward_hook)

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> Optional[torch.Tensor]:
        # forward pass
        inputs = x.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores[node_id].backward()

        return inputs.grad

    # overriding
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute saliency maps
        gradients: Optional[torch.Tensor] = self._compute_gradients(
            x, edge_index, node_id
        )
        if gradients is None:
            raise RuntimeError("Error in gradient computation")
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class GuidedBackprop(Explainer):
    @torch.no_grad()
    def __init__(self, model: torch.nn.Module) -> None:
        # call super class constructor
        super().__init__(copy.deepcopy(model))

        # init activations maps of model
        self.activation_maps: List[torch.Tensor] = []

        # register hooks
        self.register_hooks()

    def register_hooks(self) -> None:
        # define forward hook
        def forward_hook_fn(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            self.activation_maps.append(output)

        # define backward hook
        def backward_hook_fn(
            module: torch.nn.Module, grad_in: torch.Tensor, grad_out: torch.Tensor
        ) -> Tuple[torch.Tensor]:
            # create forward pass
            forward_grad: torch.Tensor = self.activation_maps.pop()
            forward_grad[forward_grad > 0] = 1

            # compute
            new_grad_out: torch.Tensor = F.relu(grad_out[0]) * forward_grad

            return (new_grad_out,)

        # define hooks variables
        forward_hook: Callable = forward_hook_fn
        backward_hook: Callable = backward_hook_fn

        # get modules
        modules: List[Tuple[str, torch.nn.Module]] = list(self.model.named_children())

        # register hooks in relus
        module: torch.nn.Module
        for _, module in modules:
            if isinstance(module, torch.nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> Optional[torch.Tensor]:
        # forward pass
        inputs = x.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores[node_id].backward()

        return inputs.grad

    # overriding
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute saliency maps
        gradients: Optional[torch.Tensor] = self._compute_gradients(
            x, edge_index, node_id
        )
        if gradients is None:
            raise RuntimeError("Error in gradient computation")
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class SaliencyMap(Explainer):
    def __init__(self, model: torch.nn.Module) -> None:
        # call super class constructor
        super().__init__(model)

    @torch.enable_grad()
    def _compute_gradients(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> Optional[torch.Tensor]:
        # forward pass
        inputs = x.clone()
        inputs.requires_grad_(True)
        outputs = self.model(inputs, edge_index)
        max_scores = torch.amax(outputs, dim=1)

        # clear previous gradients and backward pass
        self.model.zero_grad()
        max_scores[node_id].backward()

        return inputs.grad

    # overriding
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute saliency maps
        gradients: Optional[torch.Tensor] = self._compute_gradients(
            x, edge_index, node_id
        )
        if gradients is None:
            raise RuntimeError("Error in gradient computation")
        feature_maps = torch.mean(torch.abs(gradients), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class SmoothGrad(Explainer):
    """
    This class creates smoothgrad saliency map visualizations. This class inherits from SaliencyMap class

    Attributes:
        model for classifying images
        threshold for masking part of saliency map
        noise level for the creation of smoothgrad visualizations
        sample size for creation of noise duplicates
    """

    def __init__(
        self, model: torch.nn.Module, noise_level: float = 0.2, sample_size: int = 50
    ):
        """
        Constructor of SmoothGradSaliencyMap class

        Args:
            model for classifying images
            noise level for the creation of smoothgrad visualizations. Default value: 0.2
            sample size for creation of noise duplicates. Default value: 50
        """

        # set noise level and sample size
        self.model = model
        self.noise_level = noise_level
        self.sample_size = sample_size

    # overriding super class method
    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        """
        This method computes smoothgrad saliency maps

        Args:
            batch of images. Dimensions: [bath size, channels, height, width]

        Returns:
            batch of saliency maps. Dimensions: [batch size, height, width]
        """

        # compute inputs with noise
        min_ = torch.amin(x)
        max_ = torch.amax(x)
        std = (
            (max_ - min_)
            * self.noise_level
            * torch.ones(self.sample_size, *x.size()).to(x.device)
        )
        noise = torch.normal(mean=0, std=std)
        inputs = x.clone().unsqueeze(0)
        inputs = inputs + noise

        # create gradients tensor
        gradients = torch.zeros_like(inputs)

        # compute gradients for each noise batch
        for i in range(inputs.size(0)):
            # clone batch
            inputs_batch = inputs[i].clone()

            # pass the noise batch through the model
            with torch.enable_grad():
                inputs_batch.requires_grad_()
                outputs = self.model(inputs_batch, edge_index)
                max_scores = torch.amax(outputs, dim=1)

                # compute gradients
                self.model.zero_grad()
                max_scores[node_id].backward()
                if inputs_batch.grad is None:
                    raise RuntimeError("Error in gradient computation")
                gradients[i] = inputs_batch.grad

        # create saliency maps
        feature_maps = torch.mean(gradients, dim=0) / self.sample_size
        feature_maps = torch.amax(torch.abs(feature_maps), dim=1)

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class GNNExplainer(Explainer):
    def __init__(self, model: torch.nn.Module):
        """
        Constructor for Explainer class

        Args:
            model: pytorch model
        """

        # call super class constructor
        super().__init__(model)

        # define explainer and targets
        self.explainer = torch_geometric.explain.Explainer(
            self.model,
            torch_geometric.explain.GNNExplainer(),
            explanation_type="model",
            node_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    # overriding method
    @torch.enable_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute feature maps
        explanation = self.explainer(x, edge_index, index=node_id)
        feature_maps = explanation.node_mask[:, 0]

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps


class PGExplainer(Explainer):
    @torch.no_grad()
    def __init__(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        train_node_ids: torch.Tensor,
    ):
        """
        Constructor for Explainer class

        Args:
            model: pytorch model
        """

        # call super class constructor
        super().__init__(model)

        # define explainer and targets
        self.explainer = torch_geometric.explain.Explainer(
            self.model,
            PGExplainerAlgorithm(30),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

        # pass mlp to x device
        self.explainer.algorithm.mlp = self.explainer.algorithm.mlp.to(x.device)

        # compute targets
        self.targets = model(x, edge_index)

        # train
        with torch.enable_grad():
            for epoch in range(30):
                for node_id in train_node_ids:
                    self.explainer.algorithm.train(
                        epoch,
                        model,
                        x,
                        edge_index,
                        target=self.targets,
                        index=node_id.unsqueeze(0),
                    )

    # overriding method
    @torch.enable_grad()
    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_id: int
    ) -> torch.Tensor:
        # compute feature maps
        explanation = self.explainer(x, edge_index, index=node_id, target=self.targets)
        edge_mask: torch.Tensor = explanation.edge_mask

        feature_maps: torch.Tensor = torch.zeros_like(x[:, 0])
        feature_maps[edge_index[0, edge_mask != 0]] = edge_mask[edge_mask != 0]

        # normalize
        min_ = torch.amin(feature_maps)
        max_ = torch.amax(feature_maps)
        feature_maps = (feature_maps - min_) / (max_ - min_)

        return feature_maps
