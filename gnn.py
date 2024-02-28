
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
    out = model(data.x, data.edge_index)[train_idx]  #change from adj_t to edge_index
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def get_node_index_by_year_range(data, start_year, end_year):
    node_year = data.node_year.squeeze()
    return torch.logical_and(node_year >= start_year, node_year < end_year).nonzero().squeeze()

def get_subgraph_by_year_range(data, start_year, end_year):
    node_idx = get_node_index_by_year_range(data, start_year, end_year)
    return data.subgraph(node_idx)
    
def test_by_year(model, data, start_year, last_year, evaluator):
    model.eval()

    out = model(data.x, data.edge_index)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = {}
    for year in range(start_year, last_year+1):
        if year == start_year:
            node_idx = get_node_index_by_year_range(data, 0, start_year+1)
        else:
            node_idx = get_node_index_by_year_range(data, year, year+1)
        if node_idx.numel() == 0:
            continue
        acc = evaluator.eval({
            'y_true': data.y[node_idx].view(-1,1),
            'y_pred': y_pred[node_idx].view(-1,1),
            })['acc']
        results[year] = acc
    return results

def process_data(data, logger):
    years = [2017, 2018, 2019]
    subdata_dict = {}

    for year in years:
        subdata_dict[year] = get_subgraph_by_year_range(data, 0, year + 1)

        logger.save_results(f"{year} {subdata_dict[year]}")
    return years, subdata_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--method',type=str, default="SAGE")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--dataset', type = str, required=True)
    args = parser.parse_args()
    logger = Logger(args.runs, args)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(device)
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
    years, subdata_dict = process_data(data, logger)

    train_data = subdata_dict[years[0]]
    split_idx = {}
    last_year_node_index = get_node_index_by_year_range(train_data, years[0], years[0]+1).tolist()
    previous_year_node_index = get_node_index_by_year_range(train_data, 0, years[0]).tolist()
    random.shuffle(last_year_node_index)
    test_cnt = int(0.2 * len(last_year_node_index))
    split_idx['test'] = split_idx['valid'] = torch.LongTensor(last_year_node_index[:test_cnt]).to(device)
    split_idx['train'] = torch.LongTensor(previous_year_node_index + last_year_node_index[test_cnt:]).to(device)
    logger.save_results(f"{split_idx['train'].size()}, {split_idx['test'].size()}")
    train_idx = split_idx['train']


    if args.method == "SAGE":
        model = SAGE(data.num_features, args.hidden_channels,
                     dataset.num_classes, args.num_layers,
                     args.dropout).to(device)
    elif args.method == "GCN":
        model = GCN(data.num_features, args.hidden_channels,
                    dataset.num_classes, args.num_layers,
                    args.dropout).to(device)
    else:
        print("error")
        return False

    evaluator = Evaluator(name=args.dataset)
    

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in tqdm(range(1, 1 + args.epochs), desc=f'Run {run + 1}/{args.runs}'):
            loss = train(model, train_data, train_idx, optimizer)
            result = test(model, train_data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                logger.save_results(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {100 * train_acc:.2f}%, Valid: {100 * valid_acc:.2f}%, Test: {100 * test_acc:.2f}%')
    
        logger.print_statistics(run)
        for year in years:
            results = test_by_year(model, subdata_dict[year], years[0], year, evaluator)
            logger.save_results(f'{year}\n{results}')
        
        for p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]:
            new_node_index = get_node_index_by_year_range(data, years[0]+1, years[-1]+1).tolist()
            previous_node_index = get_node_index_by_year_range(data, 0, years[0]+1).tolist()
            l = int(p*len(new_node_index))
            logger.save_results(f'p={p}, len={l}')

            node_index = torch.LongTensor(previous_node_index + new_node_index[:l]).to(device)
            subdata = data.subgraph(node_index)
            results = test_by_year(model, subdata, years[0], years[-1], evaluator)
            logger.save_results(str(results))

    logger.print_statistics()

if __name__ == "__main__":
    main()
