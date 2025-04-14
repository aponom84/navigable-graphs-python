#!/usr/bin/sh

python test-generator.py -dim 32 -n 10000 -nq 100 -k 10 -v datasets/synthetic/10k_d32_data.txt -t datasets/synthetic/10k_d32_queries.txt -gt datasets/synthetic/10k_d32_gt.txt