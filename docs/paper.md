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
date: 31 December 2024
bibliography: paper.bib
---

# Summary

Data Assimilation (DA) combines computer models with real-world measurements to
improve predictions. The Python package `dageo` is a tool to apply DA in
geoscience applications. Currently, it implements the Ensemble Smoother with
Multiple Data Assimilation (ESMDA) method [@esmda] and provides tools for
reservoir engineering applications. The package includes localization for
refined updates, gaussian random field generation for realistic permeability
modeling, and integration capabilities with external simulators.

An additional feature of `dageo` is an educational, two-dimensional
single-phase reservoir simulator that models pressure changes over time and
well behavior for both injection and production scenarios. This simulator is
particularly useful for educational purposes, providing a practical platform
for students and researchers to learn and experiment with DA concepts and
techniques. The software is well documented, with examples that guide users
through learning ESMDA concepts, testing new ideas, and applying methods to
real-world problems.


# ESMDA

ESMDA is the first implemented method, out of the current need of the authors.
However, `dageo` is general enough so that other DA methods can and will be
added easily at a later stage. While ESMDA is theoretically straightforward,
practical implementation requires careful handling of matrix operations,
ensemble management, and numerical stability. The algorithm works by
iteratively updating an ensemble of model parameters to match observed data,

$$
z_j^a = z_j^f + C_\text{MD}^f \left(C_\text{DD}^f + \alpha C_\text{D}
\right)^{-1}\left(d_{\text{uc},j} - d_j^f \right) \ ,
$$

where $z^a$ represents the updated (analysis) parameters, $z^f$ the prior
(forecast) parameters, and the $C$ terms represent various covariance matrices
for the data and the model parameters (subscripts D and M, respectively). The
ESMDA coefficient (or inflation factor) is denoted by $\alpha$, and the
predicted and perturbed data vectors by $d^f$ and $d_{\text{uc}}$,
respectively. The equation is evaluated for $i$ data assimilation steps, where
$i$ is typically a low number between 4 to 10. The $\alpha$ can change in each
step, as long as $\sum_i \frac{1}{\alpha_i} = 1$. Common are either constant
values or series of decreasing values. The algorithm's implementation in
`dageo` includes optimizations for computational efficiency and numerical
stability.


# Key Features and Applications

Existing implementations often lack documentation and informative examples,
creating barriers for newcomers. These challenges are addressed in `dageo`
through several key innovations: it provides a robust, tested ESMDA
implementation alongside a built-in, simple reservoir simulator, while
offering, as a key feature, integration capabilities with external simulators.
The gallery contains an example of this integration with the \emph{open Delft
Advanced Research Terra Simulator} `open-DARTS` [@opendarts], a
state-of-the-art, open-source reservoir simulation framework developed at TU
Delft. It demonstrates how `dageo` can be used with industry-standard
simulators while maintaining its user-friendly interface. The code itself is
light, building upon NumPy arrays [@NumPy] and sparse matrices provided by
SciPy [@SciPy], as only dependencies.

While other ESMDA implementations exist, e.g., `pyesmda` [@pyesmda], `dageo`
distinguishes itself through comprehensive documentation and examples, the
integration of a simple but practical reservoir simulator, the implementation
of advanced features like localization techniques for parameter updates,
gaussian random field generation for realistic permeability modeling, and a
focus on educational applications. This makes `dageo` a unique and valuable
tool for both research and teaching. The software has been used in several
research projects, including reservoir characterization studies at TU Delft,
integration with the DARTS simulator for geothermal applications, and
educational workshops on data assimilation techniques. These applications
highlight the software's versatility and its ability to address a wide range of
challenges in reservoir engineering and geoscience.


# Acknowledgements

This work was supported by the [Delphi
Consortium](https://www.delphi-consortium.com). The authors thank Dr. D.V.
Voskov for his insights on implementing a reservoir simulation.


# References

