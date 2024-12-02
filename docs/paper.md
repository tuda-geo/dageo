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

Data Assimilation (DA) methods aim to combine numerical models with observed
data. Originating in weather prediction, it found widespread application in
other disciplines, such as reservoir modelling.

many different variants

many methods are expensive

es-mda is a simple, "cheap" variant

The code `resmda` provides a simple, two-dimensional (2D) reservoir simulator
and a straight-forward implementation of the basic *Ensemble Smoother with
Multiple Data Assimilation* (ES-MDA) algorithm as presented by @esmda.


- TODO «Begin your paper with a summary of the high-level functionality of your
  software for a non-specialist reader. Avoid jargon in this section.»
- Further details: «...we require that authors include in the paper some
  sentences that explain the software functionality and domain of use to a
  non-specialist reader. We also require that authors explain the research
  applications of the software. The paper should be between 250-1000 words.»
- «A Statement of need section that clearly illustrates the research purpose of
  the software and places it in the context of related work.»
- «A list of key references, including to other software addressing related
  needs. Note that the references should include full names of venues, e.g.,
  journals and conferences, not abbreviations only understood in the context of
  a specific discipline.»
- «Mention (if applicable) a representative set of past or ongoing research
  projects using the software and recent scholarly publications enabled by it.»
- «Acknowledgement of any financial support.»


This new Python package, now available via pip and conda, brings simplicity and versatility to ensemble-based data assimilation. 


# Statement of need

(Why resmda?)

[Go here for background](https://tuda-geo.github.io/resmda/manual/about.html)

resmda can serve as an Ensemble Smoother Multiple Data Assimilation (ES-MDA)
engine for any forward modeling or simulation code. Plus, we’ve added a rich
gallery of examples to help users get started, including:

- ES-MDA: A step-by-step introduction
- 2D Reservoir Models: From basic to complex cases with channelized systems
- Geothermal Case Study: Integration with external codes (DARTS)
- Localization: Applying ES-MDA with localized adjustments

The project reflects not just a functional tool but also the importance of
examples in showcasing its use. Fingers crossed for acceptance!




Need: while es-mda is in theory very easy to implement (few lines of codes),
its correct implementation and application is often tricky. This is where
`resmda` comes in handy, as it has on one side an implementation of ES-MDA, but
equally important it has examples of its usage than can be adapted to personal
needs.



$$
    m_j^a = m_j^f + C_\text{MD}^f \left(C_\text{DD}^f + \alpha C_\text{D}
   \right)^{-1}\left(d_{\text{uc},j} - d_j^f \right) \qquad \text{(1)}
$$


The code is written in Python using NumPy and SciPy [@NumPy;@SciPy].

# Citations

Other codes provide ES-MDA implementations, e.g.,
[pyesmda](https://pypi.org/project/pyesmda),
[esmda](https://github.com/rodrigoext/esmda),
[iterative_ensemble_smoother](https://github.com/equinor/iterative_ensemble_smoother),
or
[genES-MDA](https://www.sciencedirect.com/science/article/abs/pii/S0098300422001601).

In comparison, `resmda` offers, in addition to ES-MDA, a toy reservoir modeller
for quick computations, out-of-the-box integration with an industry-standard
reservoir modeller, DARTS, and the possibility for localization.



# Acknowledgements

This code was developed with funding from the [Delphi
Consortium](https://www.delphi-consortium.com).

Gabriel/Femke: Anything else to acknowledge?


# References
