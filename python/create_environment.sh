#!/bin/bash

conda create -n albatross python=3.6
source activate albatross
pip install -U -r ./requirements.txt
