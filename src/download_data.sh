#!/bin/bash

kaggle datasets download -d fmena14/crowd-counting
python unzip.py --dataset=mall
kaggle datasets download -d tthien/shanghaitech-with-people-density-map
python unzip.py --dataset=shanghai
