import argparse

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_scipy_sparse_matrix
import pickle

from logger import Logger
import ipdb
import random
from tqdm import tqdm
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import numpy as np
import pickle

num_neg_samples = 100


def sparse2edge_index(sparse_matrix):
    row, col, _ = sparse_matrix.t().coo()
    return torch.stack([row, col], dim=0)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

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

    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)

    total_loss = total_examples = 0
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    for perm in tqdm(DataLoader(range(source_edge.size(0)), batch_size,
                                shuffle=True)):
        optimizer.zero_grad()

        h = model(data.x, data.edge_index)

        src, dst = source_edge[perm], target_edge[perm]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        print('train loss: ', loss.item())
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, train_data, test_data, split_edge, evaluator, batch_size, device):
    predictor.eval()

    def test_split(split):
        if split == 'eval_train':
            data = train_data
        else:
            data = test_data

        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        h = model(data.x, data.edge_index)

        source_ = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in tqdm(DataLoader(range(source_.size(0)), batch_size,
                                    shuffle=True)):
            src, dst = source_[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source_ = source_.view(-1, 1).repeat(1, num_neg_samples).view(-1)
        target_neg = target_neg.view(-1)
        for perm in tqdm(DataLoader(range(source_.size(0)), batch_size,
                                    shuffle=True)):
            src, dst_neg = source_[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, num_neg_samples)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


@torch.no_grad()
def test_by_year(model, predictor, data, evaluator, batch_size, device, start_year, last_year):
    predictor.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    node_year = data.node_year.squeeze()
    h = model(data.x, data.edge_index)
    data = data.cpu()
    results = {}

    for year in tqdm(range(start_year, last_year + 1)):
        if year == start_year:
            test_node_idx = torch.logical_and(node_year > 0, node_year <= year).nonzero().squeeze()
            node_num = test_node_idx.size(0)
            assert node_num <= data.num_nodes
        else:
            test_node_idx = torch.logical_and(year <= node_year, node_year < year + 1).nonzero().squeeze()
            node_num = test_node_idx.size(0)
        test_data = data.subgraph(test_node_idx)
        source_ = test_data.edge_index[0].to(device)
        target = test_data.edge_index[1].to(device)
        edge_num = test_data.edge_index[0].size(0)
        print(f"test_data_nodes_num = {node_num},edge_num ={edge_num}")

        target_neg = negative_sampling(test_data.edge_index, num_nodes=node_num,
                                       num_neg_samples=num_neg_samples * node_num).to(device)
        print("finish generate negative_samples")
        pos_preds = []
        for perm in tqdm(DataLoader(range(source_.size(0)), batch_size,
                                    shuffle=True)):
            src, dst = source_[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source_ = source_.view(-1, 1).repeat(1, num_neg_samples).view(-1)
        target_neg = target_neg.view(-1)
        for perm in tqdm(DataLoader(range(source_.size(0)), batch_size,
                                    shuffle=True)):
            src, dst_neg = source_[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, num_neg_samples)

        mrr = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

        results[year] = mrr
        del test_data
    return results


def get_train_test_subgraph(data, start_year, end_year, threshold):
    node_year = data.node_year.squeeze()
    rand_val = torch.rand(node_year.shape)
    train_node_idx = torch.logical_or(
        node_year <= start_year,
        torch.logical_and((node_year >= start_year) & (node_year < end_year), rand_val < threshold)
    ).nonzero().squeeze()

    test_node_idx = torch.logical_and(
        (node_year >= start_year) & (node_year < end_year),
        rand_val > threshold
    ).nonzero().squeeze()

    return train_node_idx, test_node_idx


def get_subgraph(data, start_year, end_year):
    node_year = data.node_year.squeeze()
    select_idx = torch.logical_and(node_year >= start_year, node_year < end_year).nonzero().squeeze()
    return select_idx


def process_data(data, logger):
    years = [2016, 2017, 2018, 2019]
    subdata_dict = {}

    for year in years:
        subdata_dict[year] = get_subgraph(data, 0, year + 1)
        # cur_data = data.subgraph(subdata_dict[year])
        # print(cur_data, cur_data.node_year.max(), cur_data.node_year.min())
        # del cur_data

    # Data(num_nodes=2373639, x=[2373639, 128], node_year=[2373639, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 41671790]) tensor(2014) tensor(1901)
    # Data(num_nodes=2487977, x=[2487977, 128], node_year=[2487977, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 45302554]) tensor(2015) tensor(1901)
    # Data(num_nodes=2604211, x=[2604211, 128], node_year=[2604211, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 49099608]) tensor(2016) tensor(1901)
    # Data(num_nodes=2715233, x=[2715233, 128], node_year=[2715233, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 52993186]) tensor(2017) tensor(1901)
    # Data(num_nodes=2819372, x=[2819372, 128], node_year=[2819372, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 56870792]) tensor(2018) tensor(1901)
    # Data(num_nodes=2927963, x=[2927963, 128], node_year=[2927963, 1], adj_t=[2927963, 2927963, nnz=60703760], edge_index=[2, 60703760]) tensor(2019) tensor(1901)
    # train_data:  2604211 49099608
    # test_data:  23242 5296
    train_node_idx, test_node_idx = get_train_test_subgraph(data, years[0], years[1], 0.8)
    return years, subdata_dict, train_node_idx, test_node_idx


def get_node_index_by_year_range(data, start_year, end_year):
    node_year = data.node_year.squeeze()
    return torch.logical_and(node_year >= start_year, node_year < end_year).nonzero().squeeze()


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    # parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--method', type=str, default="SAGE")
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()
    print(args)
    logger = Logger(args.runs, args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = PygLinkPropPredDataset(name=args.dataset,
    #                                  transform=T.ToSparseTensor())
    # data = dataset[0]  # node_year 1901 - 2019

    # data.adj_t = data.adj_t.to_symmetric()
    # data.edge_index = sparse2edge_index(data.adj_t)

    # with open('data.pickle', 'wb') as f:
    #     pickle.dump(data, f)

    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    years, subdata_dict, train_node_idx, test_node_idx = process_data(data, logger)
    train_data = data.subgraph(train_node_idx)
    test_data = data.subgraph(test_node_idx)

    print("train_data: ", train_data.num_nodes, int(train_data.edge_index.shape[1]))
    print("test_data: ", test_data.num_nodes, int(test_data.edge_index.shape[1]))

    with open('train_data.pickle', 'wb') as f:
        pickle.dump(train_data, f)

        # with open('train_data.pickle', 'rb') as f:
    #     train_data = pickle.load(f)

    with open('test_data.pickle', 'wb') as f:
        pickle.dump(test_data, f)

    # with open('test_data.pickle', 'rb') as f:
    #     test_data = pickle.load(f)

    split_edge = {}
    test_num = test_data.edge_index.shape[-1]

    split_edge["train"] = {
        'source_node': train_data.edge_index[0],
        'target_node': train_data.edge_index[1],
    }
    split_edge['eval_train'] = {
        'source_node': train_data.edge_index[0][:test_num],
        'target_node': train_data.edge_index[1][:test_num],
        'target_node_neg': negative_sampling(train_data.edge_index[:, :test_num], num_nodes=train_data.num_nodes,
                                             num_neg_samples=num_neg_samples * train_data.num_nodes)

    }
    print("finish eval train negative sample")
    split_edge["valid"] = split_edge["test"] = {
        'source_node': test_data.edge_index[0][:test_num],
        'target_node': test_data.edge_index[1][:test_num],
        'target_node_neg': negative_sampling(test_data.edge_index[:, :test_num], num_nodes=test_data.num_nodes,
                                             num_neg_samples=num_neg_samples * test_data.num_nodes)
    }
    print("finish valid train negative sample")

    with open('split_edge.pickle', 'wb') as f:
        pickle.dump(split_edge, f)

    # with open('split_edge.pickle', 'rb') as f:
    #     split_edge = pickle.load(f)

    if args.method == "SAGE":
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
        data.edge_index = sparse2edge_index(data.adj_t)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name=args.dataset)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in tqdm(range(1, 1 + args.epochs), desc=f'Run {run + 1}/{args.runs}'):
            loss = train(model, predictor, train_data, split_edge, optimizer,
                         args.batch_size, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, train_data, test_data, split_edge, evaluator,
                              args.batch_size, device)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')

        # logger.print_statistics(run)
        for year in years:
            results = test_by_year(model, predictor, data.subgraph(subdata_dict[year]), evaluator, args.batch_size,
                                   device, years[0],
                                   year)
            print(f'{year}\n{results}')

        for p in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1]:
            new_node_index = get_node_index_by_year_range(data, years[0] + 1, years[-1] + 1).tolist()
            previous_node_index = get_node_index_by_year_range(data, 0, years[0] + 1).tolist()
            l = int(p * len(new_node_index))
            print(f'p={p}, len={l}')

            node_index = torch.LongTensor(previous_node_index + new_node_index[:l]).to(device)
            subdata = data.subgraph(node_index)
            results = test_by_year(model, predictor, subdata, evaluator, args.batch_size, device, years[0], years[-1])
            print(str(results))
    logger.print_statistics()


if __name__ == "__main__":
    main()
