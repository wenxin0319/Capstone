# Observation of Distribution Shift on Dynamic Graphs
## Introduction
This repository contains code for the observation of distribution shift on dynamic graphs. The primary datasets used in this research are extracted from the Open Graph Benchmark (OGB). For instance, you can use the following command to extract the citation-v2 dataset:

```
wget https://snap.stanford.edu/ogb/data/linkproppred/citation-v2.zip
unzip citation-v2.zip -d datasets/ogbl_citation2
```

## Environment
To replicate the experiments, ensure you have the following environment set up:

Python version 3.9.0 (or any compatible version of Python 3.9)
Required packages can be installed using the following instructions:
```
pip install https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp39-cp39-linux_x86_64.whl#sha256=94b60ae7562ae732554ae8744123b33d46e659c3251a5a58c7269c12e838868b
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.4.0
pip install ipdb
pip install tqdm
pip install ogb==1.3.6
pip install tgb
```
Note: Installing torch using the provided URL ensures compatibility with tgb and ogb.

Running the Code
To run the code, refer to the run.sh script. Make sure to set the CUDA_VISIBLE_DEVICES variable before executing the command to ensure proper GPU usage. Avoid passing the device as a command-line argument (args.device) as it might not always be correct.

For example:
```
CUDA_VISIBLE_DEVICES=0 python your_script.py
```
Replace your_script.py with the actual Python script you intend to run.

## Additional Notes
Make sure to update the paths or any configurations according to your setup.
For further assistance or inquiries, feel free to contact [Wenxin Cheng] at [cwx0319@gmail.com].
Enjoy exploring the observation of distribution shift on dynamic graphs!