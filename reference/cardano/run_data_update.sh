#!/bin/bash
# Script to run Cardano data update from cron
# Created: Tue Mar 11 21:24:55 MST 2025

# Change to project directory
cd /Users/colekimball/ztech/cardano

# Activate conda environment and run data update
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate cardano
/usr/local/anaconda3/envs/cardano/bin/python /Users/colekimball/ztech/cardano/cardano_data.py --weeks 1
