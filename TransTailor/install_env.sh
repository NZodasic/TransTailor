#!/usr/bin/bash

cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

bash Anaconda3-2024.02-1-Linux-x86_64.sh

conda env create -f tensorml.yml
