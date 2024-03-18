"""!@mainpage MNIST Diffusion Project
    @brief Software used to investigate the use of generative diffusion
    modelling to recreate realistic monochrome images of handwritten digits,
    using the classical MNIST dataset as training data. The software utilises
    PyTorch to implement the neural network models and the training procedure.
    @section Modules
    @subsection models
    This module contains code used to instantiate standard image-to-image
    convolutional neural networks, as well as a @ref DDPM class, with methods
    that enable training the image-to-image model according to the diffusion
    paradigm, and similarly performing diffusion sampling.
    @subsection train
    This module contains the @ref ddpm_train function, which implements the
    diffusion training process, and saves the model parameters and samples
    periodically, according to specified training parameters.
    @author Created by J. Hughes on 06/12/2023.
"""
