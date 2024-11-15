import torch
import torch_scatter
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool.glob import global_mean_pool


class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_filters,
        node_filters,
        fc_in_dim,
        fc_out_dim,
        fc_con_list=None,
        fc_hidden_layers=(64, 64),
        unpack_output_func=None,
    ):
        super(GNN, self).__init__()

        self.gnn_model = FlowGNN(
            input_dim,
            output_dim,
            edge_filters,
            node_filters,
            fc_in_dim,
            fc_out_dim,
            fc_con_list=fc_con_list,
            fc_hidden_layers=fc_hidden_layers,
        )

        self.unpack_output_func = unpack_output_func

    def forward(self, arg, buff=False):
        if buff:
            return self.buffer_data
        else:
            if isinstance(arg, Data):
                return self.gnn_model(arg)
            else:
                edges = self.mesh.gen_edges(arg)
                arg = Data(nodes=arg, edge_index=edges)
                return self.gnn_model(arg)

    def infer_model(self, mesh):
        output = self.gnn_model(mesh)
        self.buffer_data = self.unpack_output_func(mesh, output)


class FlowGNN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        edge_filters,
        node_filters,
        fc_in_dim,
        fc_out_dim,
        fc_con_list=None,
        fc_hidden_layers=(128, 128),
    ):
        super().__init__()

        self.edge_filters = edge_filters
        self.node_filters = node_filters
        self.fc_in_dim = fc_in_dim
        self.fc_out_dim = fc_out_dim

        self.fc_con_list = fc_con_list
        self.fc_hidden_layers = fc_hidden_layers

        self.geom_in_dim = input_dim
        self.out_dim = output_dim

        self.gcnn_layers_list = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()

        for i, (ef, nf) in enumerate(zip(self.edge_filters, self.node_filters)):
            fc_con = False
            if i in self.fc_con_list:
                fc_con = True
                self.fc_layers_list.append(
                    FlowGNN_fc_block(self.fc_out_dim, self.fc_hidden_layers)
                )
            self.gcnn_layers_list.append(
                FlowGNN_conv_block(ef, nf, fc_con=fc_con, idx=i)
            )

        self.decoder = nn.LazyLinear(self.out_dim)

    def forward(self, data):
        x = data.nodes
        edge_attr = torch.mean(x[torch.transpose(data.edge_index, 0, 1)], 1)

        x_outs = {}  # nodes
        edge_outs = {}  # edges
        skip_info = x[:, : self.geom_in_dim]
        fc_out = None
        fc_count = 0

        for i, layer in enumerate(self.gcnn_layers_list):
            if layer.idx in self.fc_con_list:
                graph_pool = global_mean_pool(x, data.batch)
                graph_pool = graph_pool[data.batch]

                if fc_out is None:
                    fc_out = self.fc_layers_list[fc_count](graph_pool)
                else:
                    fc_out = self.fc_layers_list[fc_count](
                        torch.cat([fc_out, graph_pool], 1)
                    )

                fc_count += 1

            x, edge_attr = layer(x, data.edge_index, edge_attr, fc_out, skip_info)

            x_outs[i] = x
            edge_outs[i] = edge_attr

        pred = self.decoder(x)

        return pred


class ProcessorLayer(MessagePassing):
    def __init__(self, edge_feats, node_feats, hidden_state, idx=0, selu=False):
        super().__init__()

        self.name = "processor"
        self.idx = idx
        activation = nn.GELU()

        self.edge_mlp = nn.Sequential(
            nn.LazyLinear(hidden_state),
            activation,
            nn.LazyLinear(edge_feats),
            activation,
        )

        self.node_mlp = nn.Sequential(
            nn.LazyLinear(hidden_state),
            activation,
            nn.LazyLinear(node_feats),
            activation,
        )

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        updated_edges = torch.cat(
            [torch.div(x_i + x_j, 2), torch.abs(x_i - x_j) / 2, edge_attr], 1
        )
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce="mean"
        )
        return out, updated_edges


class SmoothingLayer(MessagePassing):
    def __init__(self, idx=0):
        super().__init__()

        self.name = "smoothing"
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):
        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_nodes, out_edges

    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j) / 2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = torch_scatter.scatter(
            updated_edges, edge_index[0, :], dim=node_dim, reduce="mean"
        )
        return out, updated_edges


class FlowGNN_conv_block(nn.Module):
    def __init__(self, edge_dims, node_dims, hidden_size=128, fc_con=False, idx=0):
        super().__init__()
        self.conv = ProcessorLayer(edge_dims, node_dims, hidden_size)
        self.smooth = SmoothingLayer()

        self.fc_con = fc_con
        self.idx = idx

    def forward(self, node_attr, edge_idx, edge_attr, fc_con=None, skip_info=None):
        if skip_info is not None:
            node_attr = torch.cat([node_attr, skip_info], 1)

        if self.fc_con and fc_con is not None:
            node_attr = torch.cat([node_attr, fc_con], 1)

        node_attr, edge_attr = self.conv(node_attr, edge_idx, edge_attr)
        node_attr, edge_attr = self.smooth(node_attr, edge_idx, edge_attr)

        return node_attr, edge_attr


class FlowGNN_fc_block(nn.Module):
    def __init__(self, out_dim, hidden_layers):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_layers = hidden_layers
        self.layers = nn.ModuleList()

        for hidden_dim in self.hidden_layers:
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.GELU())
        self.layers.append(nn.LazyLinear(self.out_dim))
        self.layers.append(nn.GELU())

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
