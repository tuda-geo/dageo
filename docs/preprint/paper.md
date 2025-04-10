---
title: "dageo: Data Assimilation in Geosciences"
tags:
  - python
  - geophysics
  - data assimilation
  - forecasting
  - reservoir engineering
authors:
 - name: Dieter Werthmüller
   orcid: 0000-0002-8575-2484
   affiliation: "1, 2"
 - name: Gabriel Serrao Seabra
   orcid: 0009-0002-0558-8117
   affiliation: "1, 3"
 - name: Femke C. Vossepoel
   orcid: 0000-0002-3391-6651
   affiliation: 1
affiliations:
 - name: TU Delft, NL
   index: 1
 - name: ETH Zurich, CH
   index: 2
 - name: Petroleo Brasileiro S.A. (Petrobras), BR
   index: 3
date: 5 March 2025
bibliography: paper.bib
---

# Summary

Data Assimilation combines computer models with real-world measurements to
improve estimates and forecasts of dynamical systems such as oceans,
atmosphere, and subsurface reservoirs. The Python package `dageo` is a tool to
apply data assimilation in geoscience applications. Currently, it encompasses
the Ensemble Smoother with Multiple Data Assimilation (ESMDA) method [@esmda]
and provides tools for reservoir engineering applications. The package includes
localization to help with relatively small ensembles, Gaussian random field
generation for generating heterogeneous parameter fields, and integration
capabilities with external simulators.

An additional feature of `dageo` is a two-dimensional single-phase reservoir
simulator that models pressure changes over time and well behavior for both
injection and production scenarios. This simulator is particularly useful for
educational purposes, providing a practical platform for students and
researchers to learn and experiment with data assimilation concepts and
techniques. The software features an online documentation, with examples that
guide users through learning ESMDA concepts, testing new ideas, and applying
methods to real-world problems.


# ESMDA

ESMDA is the first data assimilation method implemented in `dageo`. However,
`dageo` is general enough so that other
data assimilation methods can and will be added easily at a later stage. While
ESMDA is theoretically straightforward, practical implementation requires
careful handling of matrix operations, ensemble management, and ensuring
numerical stability. The algorithm works by iteratively updating an ensemble of
model parameters to match observed data following

$$
z_j^a = z_j^f + C_\text{ZD}^f \left(C_\text{DD}^f + \alpha C_\text{D}
\right)^{-1}\left(d_{\text{uc},j} - d_j^f \right) \ ,
$$

where $z^a$ represents the updated (analysis) parameters, $z^f$ the prior
(forecast) parameters, and the $C$ terms represent various covariance matrices
for the data and the model parameters (subscripts D and Z, respectively). The
ESMDA coefficient (or inflation factor) is denoted by $\alpha$, the predicted
data vector, which is obtained by applying the observation operator to the
model output, is $d^f$ and $d_{\text{uc}}$ represents the perturbed observations
[@burgers] for the j-th ensemble member, generated by adding random noise to
the original observations for each iteration, as proposed in the original ESMDA
method. Note that we assume to have an identity observation operator that
translates the model state to the equivalent of the observations, so it is
omitted in the equation (for more details in this regard see @evensen). The
equation is evaluated for $i$ steps, where $i$ is typically a low number
between 4 to 10. The $\alpha$ can change in each step, as long as $\sum_i
\frac{1}{\alpha_i} = 1$. Common are either constant values or series of
decreasing values. Note that while this explanation describes the parameter
estimation problem, it could also be used to estimate the state estimation or
both. The algorithm's implementation in `dageo` includes optimizations for
efficient computation of the covariance matrix and allows to easily parallelize
the forward model.


# Key Features and Applications

Existing implementations often lack documentation and informative examples,
creating barriers for unexperienced users of data assimilation methods.
These challenges are addressed in `dageo`
through several key innovations: it provides a robust, tested ESMDA
implementation alongside a built-in, simple reservoir simulator, while offering
and showcasing in the gallery, as a key feature, integration capabilities with
external simulators. The gallery contains an example of this integration with
the \emph{open Delft Advanced Research Terra Simulator} `open-DARTS`
[@opendarts], a state-of-the-art, open-source reservoir simulation framework
developed at TU Delft. It demonstrates how `dageo` can be used with
industry-standard simulators while maintaining its user-friendly interface. The
code itself is light, building upon NumPy arrays [@NumPy] and sparse matrices
provided by SciPy [@SciPy], as only dependencies.

While other ESMDA implementations exist, e.g., `pyesmda` [@pyesmda], `dageo`
distinguishes itself through comprehensive documentation and examples, the
integration of a simple but practical reservoir simulator, the implementation
of advanced features like localization techniques for parameter updates,
Gaussian random field generation for realistic permeability modeling, and a
focus on ease of use, making it suitable for educational applications. This
makes `dageo` a unique and valuable tool for both research and teaching. The
software has been used in several research projects, including reservoir
characterization studies at TU Delft, integration with the open-DARTS simulator
for geothermal applications, and educational workshops on data assimilation
techniques [e.g., @saifullin; @seabra]. These applications highlight the
software's versatility and its ability to address a wide range of challenges in
reservoir engineering and geoscience.


# Acknowledgements

This work was supported by the [Delphi
Consortium](https://www.delphi-consortium.com). The authors thank Dr. D.V.
Voskov for his insights on implementing a reservoir simulation.


# References

