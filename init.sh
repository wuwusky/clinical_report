#! /bin/bash

conda create -n zh
conda activate zh
conda install --yes --file requirements.txt 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  -r ./requirements_pip.txt
