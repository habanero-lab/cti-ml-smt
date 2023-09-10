#!/bin/bash

mkdir -p data

wget http://public.sdh.cloud/prog-eval-data/ipnsw/music.edges.gz -O data/music.edges.gz
wget http://public.sdh.cloud/prog-eval-data/ipnsw/database_music100.bin.gz -O data/database_music100.bin.gz
wget http://public.sdh.cloud/prog-eval-data/ipnsw/query_music100.bin.gz -O data/query_music100.bin.gz

gunzip data/*.gz

python prep.py \
    --inpath data/music.edges \
    --outpath data/music.graphs.pkl
