# README

These tutorials were designed with the goal of building intution and introduce some of the tips and tricks you encounter when using GPs. These tutorials can be thought of as the homework for a mini course in Gaussian processes. They consist of python notebooks which intentionally include gaps in the code. Fill in the sections marked `YOUR CODE HERE`. The first few tutorials might take several hours to work through, as they progress they should take less effort to complete.

## Setting up the environment

### Setup using conda

1. Check that you have [conda](https://docs.conda.io/en/latest/) installed. 
To do so, run 
```sh
conda list
```
2. Run 
```sh
./create_environment
```
to generate the environment and install the dependencies. 

1. `cd` into the [python](../python) directory.

### Setup using virtualenv

1. Check that [virtualenv](https://github.com/pyenv/pyenv) is installed, if not then run 
```sh
pip install virtualenv
```
to install.
2. Generate a new environment named `albatross` (or any other name of your choice).
```sh
virtualenv albatross
```
3. Activate the new environment
```sh
source albatross/bin/activate
```
4. Install all the requirements
```sh
pip install -r requirements.txt
```

## Starting the notebooks

1. Make sure the `albatross` virtual environment is activated.
2. `cd` into to the [tutorials](../tutorials) directory. 
3. Run 
```sh
jupyter notebook
```
to start up the IPython notebooks in your browser. 

