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


## Launching the tutorials using Docker

Docker provides an isolated environment with all necessary dependencies pre-installed, making it the easiest way to get started with the Albatross tutorials without worrying about local Python environment conflicts.

### Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/) on your system
2. Ensure Docker is running by checking:

   ```sh
   docker --version
   ```

### Step-by-step Docker setup

1. **Pull the Jupyter Data Science notebook image**

   ```sh
   docker pull jupyter/datascience-notebook:latest
   ```

   This image contains Jupyter Lab with popular data science libraries (numpy, pandas, matplotlib, scipy, etc.) pre-installed.

2. **Launch the Docker container**

   ```sh
   docker run --rm -ti -p 8888:8888 -v /path/to/albatross:/home/jovyan jupyter/datascience-notebook:latest
   ```

   **Command breakdown:**
   - `--rm`: Remove the residuals of the container after stopping it.
   - `-ti`: Run in interactive mode with a TTY
   - `-p 8888:8888`: Map port 8888 from container to host
   - `-v /path/to/albatross:/home/jovyan`: Mount the albatross directory into the container's `work` directory. Note: `jovyan` is the main user folder, and its contents will be shown in the Jupyter lab file browser when the service is launched.
   - The container will start and display a URL with a token

3. **Access Jupyter Lab**

   Copy the URL from the terminal output (it will look like `http://127.0.0.1:8888/lab?token=...`) and paste it into your browser, or simply go to [http://127.0.0.1:8888/lab](http://127.0.0.1:8888/lab) and enter the token when prompted.

4. **Install Albatross-specific requirements**

   Once Jupyter Lab is open, click on the "Terminal" tile in the launcher to open a terminal, then run:

   ```bash
   pip install -r python/requirements.txt
   ```

   This installs additional packages required for the tutorials:
   - `emcee`: MCMC sampling library
   - `gpflow`: Gaussian Process library
   - `pyproj`: Cartographic projections
   - `seaborn`: Statistical visualization
   - And other dependencies

5. **Start working with tutorials**

   Navigate to the `tutorials/` folder in the file browser and open any of the available notebooks:
   - `tutorial_1_one_dimension.ipynb` - Introduction to 1D Gaussian Processes
   - `tutorial_2_maximum_likelihood_estimation.ipynb` - Parameter estimation
   - `tutorial_3_one_dimension_sparse.ipynb` - Sparse GPs for scalability
   - `tutorial_4_kalman_fliter_equivalent.ipynb` - Time series applications
   - `tutorial_5_evaluating_uncertainty.ipynb` - Uncertainty quantification

### Troubleshooting

- **Port 8888 already in use**: If you get a port binding error, either stop the existing service using port 8888 or use a different port:

  ```sh
  docker run --rm -ti -p 8889:8888 -v `pwd`:/home/jovyan jupyter/datascience-notebook:latest
  ```

  Then access via [http://127.0.0.1:8889/lab](http://127.0.0.1:8889/lab)

- **Permission issues**: On Linux/macOS, you might need to add your user to the docker group or run with `sudo`

- **Volume mounting issues**: Ensure you're running the command from the albatross root directory

### Stopping the container

To stop the container, press `Ctrl+C` in the terminal where Docker is running, or run:

```sh
docker ps  # Find the container ID
docker stop <container_id>
```
