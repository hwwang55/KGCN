# KGCN / KGCN-LS

This repository is the implementation of KGCN ([arXiv](https://arxiv.org/abs/1901.08907)) and KGCN-LS ([arXiv](http://arxiv.org/abs/1905.04413)):

> Knowledge Graph Convolutional Networks for Recommender Systems  
Hongwei Wang, Miao Zhao, Xing Xie, Wenjie Li, Minyi Guo.  
In Proceedings of The 2019 Web Conference (WWW 2019)

> Knowledge Graph Convolutional Networks for Recommender Systems with Label Smoothness Regularizations  
Hongwei Wang, Fuzheng Zhang, Mengdi Zhang, Jure Leskovec, Miao Zhao, Wenjie Li, Zhongyuan Wang.  
In Proceedings of The 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019)

![](https://github.com/hwwang55/KGCN/blob/master/framework.png)

KGCN is **K**nowledge **G**raph **C**onvolutional **N**etworks for recommender systems, which uses the technique of graph convolutional networks (GCN) to proces knowledge graphs for the purpose of recommendation.
KGCN-LS (**K**nowledge **G**raph **C**onvolutional **N**etworks with **L**abel **S**moothness regularization) further improves KGCN by adding a label smoothness regularizer in the loss function for more powerful and adaptive learning.


### Files in the folder

- `data/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
  - `restaurant/`
    - `Dianping-Food.zip`: containing the final rating file and the final KG file;
- `src/`: implementations of KGCN and KGCN-LS.




### Running the code
- Movie  
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  Download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ mv ml-20m/ratings.csv data/movie/
  $ cd src
  $ python preprocess.py --dataset movie
  $ python main.py (settings for KGCN and KGCN-LS are different. Plase carefully choose the corresponding code blocks in main.py)
  ```
- Music
  - ```
    $ cd src
    $ python preprocess.py --dataset music
    ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Last.FM;
    
  - ```
    $ python main.py (settings for KGCN and KGCN-LS are different. Plase carefully choose the corresponding code blocks in main.py)
    ```
- Restaurant  
  ```
  $ cd data/restaurant
  $ unzip Dianping-Food.zip
  ```
  - open `src/main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Dianping-Food;
    
  - ```
    $ python main.py (settings for KGCN and KGCN-LS are different. Plase carefully choose the corresponding code blocks in main.py)
    ```
