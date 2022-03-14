# Data-driven reconstruction of Gravitational Lenses using Recurrent Inference Machine II

by
Alexandre Adam,
Laurence Perreault-Levasseur,
Yashar Hezaveh

> This is the main implementation of the code for the paper, 
> including the RIM, VAE, preprocessing of COSMOS and IllustrisTNG, 
> simulation of gravitational lenses similar to high quality HST images 
> and post-processing of the results.

<img src="https://raw.githubusercontent.com/AlexandreAdam/Censai/barebone/.github/images/large_result_grid.png" alt="" style="height: 1000px; width:1000px;"/>

## Abstract
> Modeling strong gravitational lenses in order to quantify the distortions of the background sources
> and reconstruct the mass density in the foreground lens has traditionally been a major computational
> challenge. As the quality of gravitational lens images increases, the task of fully exploiting the infor-
> mation they contain becomes computationally and algorithmically more difficult. In this work, we use
> a neural network based on the Recurrent Inference Machine (RIM) to simultaneously reconstruct an
> undistorted image of the background source and the lens mass density distribution as pixelated maps.
> The method we present iteratively reconstructs the model parameters (the source and density map
> pixels) by learning the process of optimization of their likelihood given the data using the physical
> model (a ray tracing simulation), regularized by a prior implicitly learnt by the neural network through
> its training data. When compared to more traditional parametric models, the method we propose is
> significantly more expressive and can reconstruct complex mass distribution, which we demonstrate
> by using realistic lensing galaxies taken from the hydrodynamical IllustrisTNG simulation .
> Fill out the sections below with the information for your paper.


## Software implementation
The source code for the neural networks, physical models and data preprocessing 
can be found in the directory `censai`.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/AlexandreAdam/Censai.git


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create


## Reproducing the results

Before running any code you must activate the conda environment:

    source activate ENVIRONMENT_NAME

or, if you're on Windows:

    activate ENVIRONMENT_NAME

This will enable the environment for your current terminal session.

Another way of exploring the code results is to execute the Jupyter notebooks
individually.
To do this, you must first start the notebook server by going into the
repository top level and running:

    jupyter lab

This will start the server and open your default web browser to the Jupyter
interface. In the page, go into the `code/notebooks` folder and select the
notebook that you wish to view/run. You will need to install jupyterlab:

    pip install jupyterlab


## License

All source code is made available under a MIT License. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE` for the full license text.

