#!/bin/bash -x
set -e

SCRIPT_DIRECTORY=`dirname $0`
ENVIRONMENT_NAME=albatross_python_env
ENVIRONMENT_DIR=$SCRIPT_DIRECTORY/$ENVIRONMENT_NAME

cd $SCRIPT_DIRECTORY/../doc;

if ! type "conda" > /dev/null; then
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    hash -r
    # Useful for debugging any issues with conda
fi

conda config --set always_yes yes --set changeps1 no
#conda update -q conda
#conda info -a

# Create a new virtual env if one doesn't exist
if [ ! -f $ENVIRONMENT_DIR/bin/python ]; then
    conda env create  --prefix ./$ENVIRONMENT_NAME --file ../ci/albatross_docs_environment.yml
fi

source activate `realpath ./$ENVIRONMENT_NAME`

echo `python --version`
echo `which python`
make html

