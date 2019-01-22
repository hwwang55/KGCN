# KGCN / KGCN-LS

This repository is the implementation of KGCN and KGCN-LS.


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
