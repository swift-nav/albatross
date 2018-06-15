#!/bin/bash -x
set -e

SCRIPT_DIRECTORY=`dirname $0`
ENVIRONMENT_NAME=albatross_python_env
ENVIRONMENT_DIR=$SCRIPT_DIRECTORY/$ENVIRONMENT_NAME
ACTIVATE_SCRIPT=my_project/bin/activate

cd $SCRIPT_DIRECTORY;

# Create a new virtual env if one doesn't exist
if [ ! -f $ENVIRONMENT_DIR/bin/activate ]; then
    python3 -m venv $ENVIRONMENT_NAME
fi

source $ENVIRONMENT_NAME/bin/activate


echo `python --version`
pip install -r requirements.txt
make html

