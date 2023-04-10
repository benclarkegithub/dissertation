# Coarse-to-fine Perceptual Decomposition With Deep Generative Models

Dissertation project for [Artificial Intelligence (MSc)](https://www.ed.ac.uk/studying/postgraduate/degrees?id=107&r=site/view) at [The University of Edinburgh](https://www.ed.ac.uk/informatics).

Supervisor: [Siddharth N.](https://homepages.inf.ed.ac.uk/snaraya3/)

One of the fundamental research goals of artificial intelligence is knowledge representation and compression.
At the heart of this are deep generative models (DGMs), models which approximate highly complicated and intractable probability distributions.
However, training DGMs to organise latent variables hierarchically, variables which represent high-level information about the data, remains an open problem.
This project investigates three novel approaches to coarse to fine perceptual decomposition.

See `Experiments` for an example of how the architectures, methods, datasets, and evaluation suite interact.

## Introduction

This project aims to explore lossy compression techniques using deep generative models (DGMs). DGMs are powerful tools used to model complex, high-dimensional probability distributions, and are able to estimate the likelihood of, represent, and generate data. The goal of this project is to investigate hierarchical compression, a technique that involves compressing data from high to low levels, also known as coarse-to-fine compression.

The motivation behind hierarchical compression is to discard information that is conceptually redundant and not essential for maintaining the perceptual quality of the image, while still retaining its most important features. This research area is relatively unexplored, and it is rare to find DGMs that function with a variable number of hidden components, also known as latent representations.

The project will address this gap in knowledge by studying how DGMs can represent data to compress it hierarchically. This involves understanding how the model can learn to extract and represent the most salient features of an image at multiple levels of abstraction. The results of this research could have significant implications for a variety of fields in artificial intelligence and data analysis.

In addition, the project will explore the interrelated topics of linear factor models and representation learning. Linear factor models, such as principal component analysis (PCA), have been used to efficiently represent data by exploiting correlations between different dimensions. This allows for high-dimensional data to be represented by low-dimensional data and is more effective the more correlated the dimensions are. For example, PCA has been used to represent 3D human body shapes by exploiting correlations between different features such as fat/thin and tall/short.

Representation learning involves learning representations of the data that are useful for subsequent processing, such as classification or clustering. DGMs have been shown to be effective at learning representations that capture the underlying structure of the data. By studying how DGMs can be used for hierarchical compression, the project aims to contribute to the field of representation learning and develop techniques that can be applied to a wide range of problems in data analysis and artificial intelligence.

## Background & Related Work

### Variational Autoencoders

This project uses the Variational Autoencoder (VAE) \cite{VAE, Introduction} as the deep generative model. The VAE is a type of deep learning model that assumes independently and identically distributed data (i.i.d.) $\mathbf{x}$ is generated by some random process $\mathbf{x} \sim p_{\theta^*}(\mathbf{x} \mid \mathbf{z})$, involving a continuous random variable $\mathbf{z}$, called a latent variable, which represents high-level components in the data.

VAEs offer an efficient solution to approximate three important quantities:

1. The maximum likelihood estimate of the parameters $\theta$, allowing artificial data to be generated by sampling $\mathbf{x} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$.
2. The posterior inference of the latent variable $\mathbf{z}$ given data $\mathbf{x}$, $p_\theta(\mathbf{z} \mid \mathbf{x})$, which is useful for knowledge representation.
3. The marginal probability of the variable $\mathbf{x}$, $p_\theta(\mathbf{x})$, which can be used to determine the likelihood of the data.

The VAE jointly learns a deep latent variable model (DLVM) $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z}) p_\theta(\mathbf{z})$, called the decoder or generative model, and a corresponding inference model $p_\theta(\mathbf{z} \mid \mathbf{x})$. VAEs solve a key problem with DLVMs, where the marginal $p_\theta(\mathbf{x})$ is intractable due to the integral $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z})\ d\mathbf{x}$. VAEs introduce an approximate inference model $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p_\theta(\mathbf{z} \mid \mathbf{x})$, called the encoder or recognition model, which enables the marginal $p_\theta(\mathbf{x})$ to be estimated and optimized by $p_\theta(\mathbf{x}) = p_\theta(\mathbf{x}, \mathbf{z}) / q_\phi(\mathbf{z} \mid \mathbf{x})$.

In practice, a VAE consists of an encoder that transforms an input $\mathbf{x}$ into a latent code $\mathbf{z}$, a decoder that maps the latent code back to an output $\mathbf{x'}$, and a loss function that compares $\mathbf{x}$ and $\mathbf{x'}$. During training, VAEs optimize a lower bound on the marginal likelihood of the data, called the Evidence Lower Bound (ELBO), using stochastic gradient descent. The ELBO has two components, the reconstruction term and the KL divergence (respectively):
```math
    \mathcal{L} (\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{KL}(q_\phi(\mathbf{z} \mid \mathbf{x}) \mid\mid p_\theta(\mathbf{z}))
```

In summary, VAEs provide a powerful tool for modeling complex high-dimensional data by learning a lower-dimensional representation that captures the salient features of the data.
