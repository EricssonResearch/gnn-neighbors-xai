# deep learning libraries
import torch
import torch_geometric
import torch.nn.functional


class GCN(torch.nn.Module):
    """
    This class defines a model with two GCN layers

    Attributes:
        conv1: first GNN
        relu: non linearity function
        dropout: dropout layer
        conv2: second GNN
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        self_loops: bool,
        hidden_channels: int = 64,
        prob_drop: float = 0.5,
    ) -> None:
        """
        This method is the constructor for the GCN class

        Args:
            input_size: number of node features
            hidden_channels: size between the layers
            output_size: number of possible classes
        """

        # call superclass constructor
        super().__init__()

        # define layers
        self.conv1: torch_geometric.nn.MessagePassing = torch_geometric.nn.GCNConv(
            input_size, hidden_channels, add_self_loops=self_loops
        )
        self.relu: torch.nn.Module = torch.nn.ReLU()
        self.dropout: torch.nn.Module = torch.nn.Dropout(prob_drop)
        self.conv2: torch_geometric.nn.MessagePassing = torch_geometric.nn.GCNConv(
            hidden_channels, output_size, add_self_loops=self_loops
        )

    def forward(self, inputs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass

        Args:
            inputs: node matrix. Dimensions: [batch size, input size]
            edge_index: edge index tensor that represents the adj matrix. Dimensions: [2, number of edges]

        Returns:
            predictions of the classes. Dimensions: [batch size, output size]
        """

        # compute outputs
        outputs: torch.Tensor = self.conv1(inputs, edge_index)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs, edge_index)

        return outputs


class GAT(torch.nn.Module):
    """
    This class defines a model with two GAT layers

    Attributes:
        conv1: first GNN
        relu: non linearity function
        dropout: dropout layer
        conv2: second GNN
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        self_loops: bool,
        hidden_channels: int = 64,
        prob_drop: float = 0.5,
    ) -> None:
        """
        This method is the constructor for the GCN class

        Args:
            input_size: number of node features
            hidden_channels: size between the layers
            output_size: number of possible classes
        """

        # call superclass constructor
        super().__init__()

        # define layers
        self.conv1: torch_geometric.nn.MessagePassing = torch_geometric.nn.GATv2Conv(
            input_size, hidden_channels, add_self_loops=self_loops
        )
        self.relu: torch.nn.Module = torch.nn.ReLU()
        self.dropout: torch.nn.Module = torch.nn.Dropout(prob_drop)
        self.conv2: torch_geometric.nn.MessagePassing = torch_geometric.nn.GATv2Conv(
            hidden_channels, output_size, add_self_loops=self_loops
        )

    def forward(self, inputs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        This method defines the forward pass

        Args:
            inputs: node matrix. Dimensions: [batch size, input size]
            edge_index: edge index tensor that represents the adj matrix. Dimensions: [2, number of edges]

        Returns:
            predictions of the classes. Dimensions: [batch size, output size]
        """

        # compute outputs
        outputs: torch.Tensor = self.conv1(inputs, edge_index)
        outputs = self.relu(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs, edge_index)

        return outputs
