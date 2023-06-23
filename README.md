# Spatial-Temporal and Text Representation Learning (STaTRL)

### Applied Intelligence 2022. IF=5.019 JCR分区（Q2）

### This repository is easily used for comparison of STaTRL without Text Representation Learning. 

| dataset | # Users | # POI |
|---------|---------|---------|
| Yelp2020     | 28038     |# 15745 |
| Yelp    | 30887     |# 18995 |
| Gowalla     | 18737     |# 32510 |
| Foursquare     | 7642     |# 28483 |

### Dependencies
* Python 3.7.6
* [Anaconda](https://www.anaconda.com/) 4.8.2 contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.7.1.

### Instructions
* step 1 check transformer/Constants.py
* step 2 cal_poi_pairwise_relation.
> python cal_poi_pairwise_relation.py
* step 3 training.
> python Main.py

### Note
* Since the best hyperparameter configuration takes a lot of memory, it hasn't been set to perform best, please tune it and get the result.
* If there is any problem, please contact to kaysenn@163.com.
* The full STaTRL model can be found in [STaTRL](https://github.com/wxf2445/STaTRL), our paper can be found in [Applied Intelligence](https://link.springer.com/content/pdf/10.1007/s10489-022-03858-w.pdf).
* If this repository helps you, please cite:
    @article{wang2023statrl,
      title={STaTRL: Spatial-temporal and text representation learning for POI recommendation},
      author={Wang, Xinfeng and Fukumoto, Fumiyo and Li, Jiyi and Yu, Dongjin and Sun, Xiaoxiao},
      journal={Applied Intelligence},
      volume={53},
      number={7},
      pages={8286--8301},
      year={2023},
      publisher={Springer}
    }
