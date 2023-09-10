#!/bin/bash

mkdir -p data

wget http://public.sdh.cloud/prog-eval-data/sinkhorn_wmd/crawl-300d-2M.vec.zip -O data/crawl-300d-2M.vec.zip
wget http://public.sdh.cloud/prog-eval-data/sinkhorn_wmd/dbpedia.train.gz -O data/dbpedia.train.gz

unzip data/crawl-300d-2M.vec.zip -d data/
rm data/crawl-300d-2M.vec.zip
gunzip data/dbpedia.train.gz

python prep.py --outpath data/cache
