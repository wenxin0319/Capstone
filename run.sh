python gnn_tgb_new.py --device 4 --dataset tgbn-trade
python test.py --device 6 --dataset tgbn-trade

python test_x_shift_tgb.py --dataset tgbn-trade

python gnn.py --dataset ogbn-mag --method SAGE
python gnn.py --dataset ogbn-arxiv --method SAGE
python test_x_shift.py --dataset ogbn-mag --method SAGE
python test_x_shift.py --dataset ogbn-arxiv --method SAGE
python test_x_shift_new.py --dataset ogbn-mag --method SAGE --device 0

CUDA_VISIBLE_DEVICES=6 python gnn_link_colab.py --dataset ogbl-collab --method GCN --epochs 400  | tee colab_gcn.log
CUDA_VISIBLE_DEVICES=7 python gnn_link_colab.py --dataset ogbl-collab --method SAGE --epochs 400 | tee colab_sage.log

CUDA_VISIBLE_DEVICES=7 python gnn_link_cite.py --dataset ogbl-citation2 --method SAGE --epochs 50  | tee cite_sage.log
CUDA_VISIBLE_DEVICES=6 python gnn_link_cite.py --dataset ogbl-citation2 --method GCN --epochs 50 | tee cite_gcn.log
CUDA_VISIBLE_DEVICES=5 python gnn_link_cite_pos.py --dataset ogbl-citation2 --method GCN --epochs 50 | tee cite_gcn_pos.log

