
import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import subgraph, to_edge_index, to_torch_coo_tensor
from torch_geometric.data import Data

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger
import random
import copy
from tqdm import tqdm
import ipdb


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

def get_node_index_by_year_range(data, start_year, end_year, dataset_name):
    if dataset_name == "ogbn-arxiv":
        node_year = data.node_year.squeeze()
    if dataset_name == "ogbn-mag":
        node_year = data.node_year.squeeze()
    return torch.logical_and(node_year >= start_year, node_year < end_year).nonzero().squeeze()

def get_subgraph_by_year_range(data, start_year, end_year, dataset_name):
    node_idx = get_node_index_by_year_range(data, start_year, end_year, dataset_name)
    return data.subgraph(node_idx)

@torch.no_grad()
def test_distinguish(model, data1, data2, year1, year2,  evaluator, dataset_name):
    model.eval()
    result = {}
    out = model(data1.x, data1.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    acc = evaluator.eval({
        'y_true': data1.y,
        'y_pred': y_pred,
    })['acc']
    
    result['data1_acc'] = acc

    node_idx1 = get_node_index_by_year_range(data2, 0, year1+1, dataset_name)
    node_idx2 = get_node_index_by_year_range(data2, year1+1, year2+1, dataset_name) 

    out = model(data2.x, data2.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    acc = evaluator.eval({
        'y_true': data2.y[node_idx1],
        'y_pred': y_pred[node_idx1],
    })['acc']
    result['data2_old_node_acc'] = acc

    if year1 != year2:
        acc = evaluator.eval({
          'y_true': data2.y[node_idx2],
          'y_pred': y_pred[node_idx2],
        })['acc']
        result['data2_new_node_acc'] = acc
    else:
        result['data2_new_node_acc'] = float('nan')
    return result

def process_data(data, logger, dataset_name):
    years = [2017, 2018, 2019]
    subdata_dict = {}

    for year in years:
        subdata_dict[year] = get_subgraph_by_year_range(data, 0, year + 1, dataset_name)
        logger.save_results(f"{year} {subdata_dict[year]}")
    return years, subdata_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=100)
    # parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--method',type=str, default="SAGE")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset', type = str, required=True)

    args = parser.parse_args()
    logger = Logger(args.runs, args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name= args.dataset,
                                     transform=T.ToUndirected())
                                     #transform=T.ToSparseTensor())

    data_ = dataset[0]
    if args.dataset == 'ogbn-mag':
        data = Data(
            num_nodes=data_.num_nodes_dict['paper'],
            edge_index=data_.edge_index_dict[('paper', 'cites', 'paper')],
            x=data_.x_dict['paper'],
            node_year=data_.node_year['paper'],
            y=data_.y_dict['paper'],
        )

    else:
        data = data_

    data = data.to(device)
    years, subdata_dict = process_data(data, logger, args.dataset)


    if args.method == "SAGE":
        model = SAGE(data.num_features, args.hidden_channels,
                     2, args.num_layers,
                     args.dropout).to(device)
    elif args.method == "GCN":
        model = GCN(data.num_features, args.hidden_channels,
                    2, args.num_layers,
                    args.dropout).to(device)
    else:
        print("error")
        return False

    evaluator = Evaluator(name=args.dataset)
    logger = Logger(args.runs, args, f"{args.dataset}" + "shift"  + "_" + f"{args.method}")

    # training
    
    for year in years:
        logger.save_results(f'year {years[0]} vs {year}')
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data1 = copy.deepcopy(subdata_dict[years[0]])
        data2 = copy.deepcopy(subdata_dict[year])
        data1.y = torch.zeros_like(data1.y)
        data2.y = torch.ones_like(data2.y)
        train_idx1 = torch.arange(data1.num_nodes).to(device)
        train_idx2 = torch.arange(data2.num_nodes).to(device)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data1, train_idx1, optimizer)
            loss += train(model, data2, train_idx2, optimizer)
            result = test_distinguish(model, data1, data2, years[0], year,  evaluator, args.dataset)

            if epoch % args.log_steps == 0:
                logger.save_results(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'old graph: {100 * result["data1_acc"]:.2f}%, '
                      f'new graph old nodes: {100 * result["data2_old_node_acc"]:.2f}% '
                      f'old nodes: {50 * (result["data1_acc"] + result["data2_old_node_acc"]):.2f}%'
                      f'new nodes: {100 * result["data2_new_node_acc"]:.2f}%')

if __name__ == "__main__":
    main()
