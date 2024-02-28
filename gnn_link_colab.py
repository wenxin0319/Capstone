import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import pickle

from logger import Logger
import ipdb
import random
from tqdm import tqdm
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import numpy as np

num_neg_samples = 100000

def sparse2edge_index(sparse_matrix):
    row, col, _ = sparse_matrix.t().coo()
    return torch.stack([row, col], dim=0)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size, device):

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(device)

    total_loss = total_examples = 0
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    #Data(num_nodes=235868, edge_index=[2, 331592], x=[235868, 128], edge_weight=[331592], edge_year=[331592, 1])

    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True)):
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        # print('train loss: ', loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples

        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test_by_year(model, predictor, data, evaluator, batch_size, device, start_year, last_year):
    model.eval()
    predictor.eval()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    edge_year = data.edge_year.squeeze()
    h = model(data.x, data.edge_index)

    results = {}
    for year in tqdm(range(start_year, last_year + 1)):
        if year == start_year:
            test_edge_idx = torch.logical_and(edge_year > 0, edge_year <= year).nonzero().squeeze()
        else:
            test_edge_idx = torch.logical_and(year <= edge_year, edge_year < year + 1).nonzero().squeeze()

        data = data.cpu()
        test_data = data.edge_subgraph(test_edge_idx)
        edge_num = test_data.edge_index[0].size(0)
        print(f"test_data_nodes_num = {test_data.num_nodes},edge_num ={edge_num}, want to sample {int(edge_num * 3)} neg")

        pos_test_edge = test_data.edge_index.t().to(device)
        neg_test_edge = negative_sampling(test_data.edge_index, num_nodes = test_data.num_nodes, num_neg_samples = int(edge_num * 3)).t().to(device)
        print("finish generate negative_samples")

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        test_hits = {}
        print(pos_test_pred.shape, neg_test_pred.shape)
        for K in [10, 50, 100]:
            evaluator.K = K
            test_hits[K] = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[year] = test_hits
    return results

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    model.eval()
    predictor.eval()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    h = model(data.x, data.edge_index)

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(data.x, data.edge_index)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] =  (train_hits, valid_hits, test_hits)

    return results


def get_train_test_subgraph(data, start_year, end_year, threshold):
    edge_year = data.edge_year.squeeze()
    rand_val = torch.rand(edge_year.shape)
    train_edge_idx = torch.logical_or(
        edge_year < start_year,
        torch.logical_and((edge_year >= start_year) & (edge_year < end_year), rand_val < threshold)
    ).nonzero().squeeze()

    test_edge_idx = torch.logical_and(
        (edge_year >= start_year) & (edge_year < end_year),
        rand_val > threshold
    ).nonzero().squeeze()

    return train_edge_idx, test_edge_idx

def get_subgraph(data, start_year, end_year):
    edge_year = data.edge_year.squeeze()
    select_idx = torch.logical_and(edge_year >= start_year, edge_year < end_year).nonzero().squeeze()
    return select_idx




def process_data(data, logger):
    years = [2014,2015,2016, 2017]
    subdata_dict = {}

    for year in years:
        subdata_dict[year] = get_subgraph(data, 0, year + 1)
        # cur_data = data.edge_subgraph(subdata_dict[year])
        # print(cur_data, cur_data.edge_year.max(), cur_data.edge_year.min())
        # del cur_data
    train_edge_idx, test_edge_idx = get_train_test_subgraph(data, years[0],years[1], 0.8)
    return years, subdata_dict, train_edge_idx, test_edge_idx

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--method', type=str, default="SAGE")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    print(args)
    logger = Logger(args.runs, args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-collab')  # year 1963-2017
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    # data = T.ToSparseTensor()(data)
    years, subdata_dict, train_edge_idx, test_edge_idx = process_data(data, logger)
    train_data = data.edge_subgraph(train_edge_idx)
    #Data(num_nodes=235868, edge_index=[2, 331689], x=[235868, 128], edge_weight=[331689], edge_year=[331689, 1])

    test_data = data.edge_subgraph(test_edge_idx)
    #Data(num_nodes=235868, edge_index=[2, 7679], x=[235868, 128], edge_weight=[7679], edge_year=[7679, 1])

    split_edge = {}
    split_edge["train"] = {
        'edge': train_data.edge_index.t(),
        'weight': train_data.edge_weight,
        'year': train_data.edge_year.view(-1)
    }
    split_edge["valid"] = split_edge["test"] = {
        'edge': test_data.edge_index.t(),
        'weight': test_data.edge_weight,
        'year': test_data.edge_year.view(-1),
        'edge_neg': negative_sampling(test_data.edge_index, num_neg_samples = num_neg_samples).t()
    }

    print("train_data_shape: ", split_edge["train"]["edge"].shape)
    print("test_data_shape: ", split_edge["test"]["edge"].shape)
    # train_data_shape: torch.Size([331775, 2])
    # test_data_shape: torch.Size([7593, 2])

    # data = T.ToSparseTensor()(data)
    # data.full_adj_t = data.adj_t

    data = data.to(device)

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in tqdm(range(1, 1 + args.epochs), desc=f'Run {run + 1}/{args.runs}'):
            loss = train(model, predictor, train_data, split_edge, optimizer,
                         args.batch_size, device)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator, args.batch_size, device)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

        for year in years:
            results = test_by_year(model, predictor, data.cpu().edge_subgraph(subdata_dict[year]), evaluator, args.batch_size // 4,
                                   device, years[0],year)
            print(f'{year}\n{results}')

        for p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]:
            new_edge_index = get_subgraph(data, years[0] + 1, years[-1] + 1).tolist()
            previous_edge_index = get_subgraph(data, 0, years[0] + 1).tolist()
            l = int(p * len(new_edge_index))
            print(f'p={p}, len={l}')

            edge_index = torch.LongTensor(previous_edge_index +  new_edge_index[:l])
            subdata = data.cpu().edge_subgraph(edge_index)
            results = test_by_year(model, predictor,subdata, evaluator, args.batch_size // 4, device, years[0], years[-1])
            print(str(results))

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()