# MNIST Diffusion Project

This repository contains software that can be used to generate synthetic images of handwritten digits.
This is achieved using diffusion models built around convolutional neural networks,
trained on the MNIST handwritten digit dataset consisting of 50,000 28x28 monochrome images.

## Description

The main source code for the project is found in the `src` directory,
which in turn contains some scripts used to run various training and analysis procedures,
as well as a package `diffusiontools` which contains various modules used to instantiate, train, and inspect various diffusion models investigated.
The software implements a standard Gaussian denoising scheme, as well as a more unconventional scheme which involves swapping pixels to degrade the images.
This is described in more detail in the report in the `report` directory.
Moreover, the code is documented in the `docs` folder, which contains both a PDF and HTML version of the codebase documentation.

## How to use the project

In order to reproduce the code results it is advisable to reproduce the development environment by running the following terminal commands:

`python -m venv M2ENV`

`source M2ENV/bin/activate` (for Linux and macOS)

`pip install --upgrade pip` (optional but recommended)

`pip install -r requirements.txt`

Then any of the three scripts found in `src` can be run via `python src/script.py`.
Additionally, the behaviour of the main training script `run.py` can be controlled via the `config.ini` file in the root of the repository.
Simply change the parameters there as desired, and then run

`python src/run.py config.ini`

Outputs from running the scripts should be produced in the `data` directory;
when running the final analysis script it is recommended to save the terminal output via

`python src/final_analysis.py > data/analysis/final_analysis.txt`
