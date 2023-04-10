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
