import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0, heads=4
    ):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        # Initialize the first layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
            )
        )
        # Initialize the hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
        # Initialize the output layer
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
            )
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # The forward pass remains similar to the SAGE example, but adapted for GATConv's requirements.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)
                x = x.flatten(1)
        return x.log_softmax(dim=-1)
