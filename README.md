# KGCN

This repository is the implementation of KGCN.


### Files in the folder

- `data/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Last.FM;
- `src/`: implementations of KGCN.




### Running the code
- Movie  
  (The raw rating file of MovieLens-20M is too large to be contained in this repository.
  You should download the dataset first.)
  ```
  $ wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
  $ unzip ml-20m.zip
  $ cp ml-20m/ratings.csv data/movie/
  $ cd src
  $ python preprocess.py --dataset movie
  $ python main.py
  ```
- Music
  - ```
    $ cd src
    $ python preprocess.py --dataset music
    ```
  - open `main.py` file;
    
  - comment the code blocks of parameter settings for MovieLens-20M;
    
  - uncomment the code blocks of parameter settings for Last.FM;
    
  - ```
    $ python main.py
    ```