---
title: "resmda: ES-MDA with a simple 2D reservoir modeller"
tags:
  - python
  - reservoir engineering
  - geophysics
  - data assimilation
  - forecasting
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
 - name: Petrobras, GABRIEL TODO, BR
   index: 3
date: 31 December 2024
bibliography: paper.bib
---

# Summary

Data Assimilation (DA) combines computer models with real-world measurements to improve predictions. The `resmda` package implements the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) method in Python, providing tools for reservoir engineering applications. The package includes localization for refined updates, gaussian random field generation for realistic permeability modeling, and integration capabilities with external simulators. At its core, `resmda` features an educational 2D single-phase reservoir simulator that models pressure changes and well behavior, making it valuable for both research and teaching.

The package also includes a educational 2D single phase reservoir simulator, which models pressure changes over time, and well behavior for both injection and production scenarios. This simulator is particularly useful for educational purposes, providing a practical platform for students and researchers to learn and experiment with data assimilation concepts. The software is well-documented, with extensive examples that guide users through learning ES-MDA concepts, testing new ideas, and applying methods to real-world problems. This makes `resmda` a resource for researchers, students, and practitioners in geoscience and reservoir engineering who need to understand and apply data assimilation techniques.

# Statement of Need

While ES-MDA is theoretically straightforward, practical implementation requires careful handling of matrix operations, ensemble management, and numerical stability. Existing implementations often lack documentation and educational examples, creating barriers for newcomers. `resmda` addresses these challenges through several key innovations: it provides a robust, tested ES-MDA implementation alongside a built-in educational reservoir simulator, while offering integration capabilities with external simulators like open-DARTS. The package includes advanced features such as localization and permeability field generation, all supported by comprehensive documentation and examples.

Most available tools either focus solely on the algorithm or are tightly coupled to specific simulators. In contrast, `resmda` offers a complete ES-MDA implementation, a built-in reservoir simulator, integration capabilities with external simulators, and advanced features like localization and efficient permeability field generation. The ES-MDA algorithm works by iteratively updating an ensemble of model parameters to match observed data, using the following equation:

$$
m_j^a = m_j^f + C_\text{MD}^f \left(C_\text{DD}^f + \alpha C_\text{D}
\right)^{-1}\left(d_{\text{uc},j} - d_j^f \right)
$$

where $m_j^a$ represents the updated (analysis) parameters, $m_j^f$ the prior (forecast) parameters, and the $C$ terms represent various covariance matrices. The algorithm's implementation in `resmda` includes optimizations for computational efficiency and numerical stability.

# Key Features

A key feature of `resmda` is its ability to integrate with external simulators. A prime example is its integration with open-DARTS (open Delft Advanced Research Terra Simulator) [@voskov2024open], a state-of-the-art open-source reservoir simulation framework developed at TU Delft. This integration is documented in the package examples, showing how `resmda` can be used with industry-standard simulators while maintaining its user-friendly interface. 

The software implements several features to enhance its functionality. These include localization techniques for parameter updates, gaussian random field generation for realistic permeability modeling, .... INCLUDE HERE MORE FEATURES.  

# Research Applications

The software has been used in several research projects, including reservoir characterization studies at TU Delft, integration with the DARTS simulator for geothermal applications, and educational workshops on data assimilation techniques. These applications highlight the software's versatility and its ability to address a wide range of challenges in reservoir engineering and geoscience.

# Comparison with Existing Software - NEEDS TO BE ENHANCED

While other ES-MDA implementations exist, such as pyesmda and esmda, `resmda` distinguishes itself through comprehensive documentation and examples, the integration of a simple but practical reservoir simulator, the implementation of advanced features like localization, and a focus on educational applications. This makes `resmda` a unique and valuable tool for both research and teaching.

# Acknowledgements

This work was supported by the [Delphi Consortium](https://www.delphi-consortium.com). The authors thank Dr. D.V. Voskov for his insights on reservoir simulation implementation.

# References


