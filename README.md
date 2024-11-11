# Pessimistic Adversarially Regularized Learning for Graph Embedding (ADMA 2023)

## PGAE/PVGAE

PGAE is the code for our paper "Pessimistic Adversarially Regularized Learning for Graph Embedding", which is published in ADMA 2023. 

<p align="center">
  <img src="./model.jpg" width="50%" />
</p>

## Citation

```
@inproceedings{li2023pessimistic,
  title={Pessimistic Adversarially Regularized Learning for Graph Embedding},
  author={Li, Mengyao and Song, Yinghao and Yan, Long and Feng, Hanbin and Song, Yulun and Li, Yang and Wang, Gongju},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={338--351},
  year={2023},
  organization={Springer}
}
```

## Data

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid

## Overview
Here we provide an implementation of PAGAE/PAGAEpo in PyTorch, along with a minimal execution example (on the Cora dataset). The repository is organised as follows:
- `pgae/data/` contains the necessary dataset files;
- `pgae/layers.py` contains the implementation of a GCN layer;
- `pgae/utils.py` contains the necessary processing function.
- `pgae/model.py` contains the implementation of a GAE model, discriminator model and mutual information estimator model.
- `pgae/optimizer.py` contains the implementation of the reconstruction loss.

## Usage
- run "./pgae/run.py"
