# SMHGC

This is the code of paper: SiMilarity-Enhanced Homophily for Multi-View Heterophilous Graph Clustering

# Requirements

- 'requirements.txt'

# Datasets

Datasets are included in `./data/`.

## Raw data

|  Dataset  | #Clusters | #Nodes |       #Features       |                           Graphs                            |              HR              |
| :-------: | :-------: | :----: | :-------------------: | :---------------------------------------------------------: | :--------------------------: |
|    ACM    |     3     |  3025  |    1830 <br />1830    |            $\mathcal{G}^1$ <br />$\mathcal{G}^2$            |       0.82 <br />0.64        |
|   DBLP    |     4     |  4057  | 334<br />334<br />334 | $\mathcal{G}^1$ <br />$\mathcal{G}^2$ <br />$\mathcal{G}^3$ | 0.80 <br /> 0.67 <br /> 0.32 |
|   Texas   |     5     |  183   |    1703<br />1703     |            $\mathcal{G}^1$<br />$\mathcal{G}^2$             |        0.09<br />0.09        |
| Chameleon |     5     | 22777  |    2325<br />2325     |            $\mathcal{G}^1$<br />$\mathcal{G}^2$             |        0.23<br />0.23        |

# Test SMHGC

```python
# Test SMHGC on ACM dataset
python SMHGC.py --dataset 'acm' --train 0 --use_cuda True --cuda_device 0

# Test SMHGC on DBLP dataset
python SMHGC.py --dataset 'dblp' --train 0 --use_cuda True --cuda_device 0

# Test SMHGC on Texas dataset
python SMHGC.py --dataset 'texas' --train 0 --use_cuda True --cuda_device 0

# Test SMHGC on Chameleon dataset
python SMHGC.py --dataset 'chameleon' --train 0 --use_cuda True --cuda_device 0
```

# Train SMHGC

```python
# Train SMHGC on ACM dataset
python SMHGC.py --dataset 'acm' --train 1 --use_cuda True --cuda_device 0

# Train SMHGC on DBLP dataset
python SMHGC.py --dataset 'dblp' --train 1 --use_cuda True --cuda_device 0

# Train SMHGC on Texas dataset
python SMHGC.py --dataset 'texas' --train 1 --use_cuda True --cuda_device 0

# Train SMHGC on Chameleon dataset
python SMHGC.py --dataset 'chameleon' --train 1 --use_cuda True --cuda_device 0
```

**Parameters**: More parameters and descriptions can be found in the script and paper.

# Results of SMHGC

|           |      NMI%      |      ARI%      |      ACC%      |      F1%       |
| :-------: | :------------: | :------------: | :------------: | :------------: |
|    ACM    | 81.1 $\pm$ 4.1 | 83.2 $\pm$ 5.2 | 93.9 $\pm$ 2.0 | 93.9 $\pm$ 2.0 |
|   DBLP    | 76.2 $\pm$ 0.8 | 81.9 $\pm$ 0.2 | 92.4 $\pm$ 0.2 | 91.8 $\pm$ 0.2 |
|   Texas   | 41.8 $\pm$ 1.1 | 46.9 $\pm$ 3.2 | 71.3 $\pm$ 0.8 | 49.8 $\pm$ 2.3 |
| Chameleon | 20.0 $\pm$ 1.3 | 15.1 $\pm$ 1.0 | 42.1 $\pm$ 0.8 | 41.3 $\pm$ 0.9 |

