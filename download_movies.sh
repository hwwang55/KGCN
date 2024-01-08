#!/bin/bash

wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
mv ml-20m/ratings.csv data/movie/