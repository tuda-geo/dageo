About
=====

A simple 2D reservoir simulator and a straight-forward implementation of the
basic *Ensemble Smoother with Multiple Data Assimilation* (ESMDA) algorithm.

.. _esmda:

ESMDA
-----


In the following an introduction to the Ensemble Smoother with Multiple Data
Assimilation (ESMDA) algorithm following [EmRe13]_:

In history-matching problems, it is common to consider solely the
parameter-estimation problem and thereby neglecting model uncertainties. Thus,
unlike with the ensemble Kalman filter (EnKF), the parameters and states are
always consistent (Thulin et al., 2007). This fact helps to explain the better
data matches obtained by ESMDA compared to EnKF. The analyzed vector of model
parameters :math:`z^a` is given in that case by

.. math::
    z_j^a = z_j^f + C_\text{MD}^f \left(C_\text{DD}^f + \alpha C_\text{D}
   \right)^{-1}\left(d_{\text{uc},j} - d_j^f \right) \qquad \text{(1)}

for ensembles :math:`j=1, 2, \dots, N_e`. Here,

- :math:`^a`: analysis;
- :math:`^f`: forecast;
- :math:`z^f`: prior vector of model parameters (:math:`N_m`);
- :math:`C_\text{MD}^f`: cross-covariance matrix between :math:`z^f` and
  :math:`d^f` (:math:`N_m \times N_d`);
- :math:`C_\text{DD}^f`:  auto-covariance matrix of predicted data
  (:math:`N_d \times N_d`);
- :math:`C_\text{D}`: covariance matrix of observed data measurement errors
  (:math:`N_d \times N_d`);
- :math:`\alpha`: ESMDA coefficient;
- :math:`d_\text{uc}` : vector of perturbed data, obtained from the
  vector of observed data, :math:`d_\text{obs}` (:math:`N_d`);
- :math:`d^f`: vector of predicted data (:math:`N_d`).

The prior vector of model parameters, :math:`z^f_j`, can in reality be
:math:`j` possible models :math:`z^f` given from an analyst (e.g., the
geologist). In theoretical tests, these are usually created by perturbing the
prior :math:`z^f` by, e.g., adding random Gaussian noise.

The ESMDA algorithm follows [EmRe13]_:

1. Choose the number of data assimilations, :math:`N_a`, and the coefficients
   :math:`\alpha_i` for :math:`i = 1, \dots, N_a`.
2. For :math:`i = 1` to :math:`N_a`:

   1. Run the ensemble from time zero.
   2. For each ensemble member, perturb the observation vector using
      :math:`d_\text{uc} = d_\text{obs} + \sqrt{\alpha_i} C_\text{D}^{1/2}
      z_d`, where :math:`z_d \sim \mathcal{N}(0,I_{N_d})`.
   3. Update the ensemble using Eq. (1) with :math:`\alpha_i`.

The difficulty is the inversion of the large (:math:`N_d \times N_d`) matrix
:math:`C=C_\text{DD}^f + \alpha C_\text{D}`, which is often poorly conditioned
and poorly scaled. How to compute this inverse is one of the main differences
between different ESMDA implementations.

Also note that in the ESMDA algorithm, every time we repeat the data
assimilation, we re-sample the vector of perturbed observations, i.e., we
recompute :math:`d_\text{uc} \sim \mathcal{N}(d_\text{obs}, \alpha_i
C_\text{D})`. This procedure tends to reduce sampling problems caused by
matching outliers that may be generated when perturbing the observations.

One potential difficultly with the proposed MDA procedure is that :math:`N_a`
and the coefficients :math:`\alpha_i`'s need to be selected prior to the data
assimilation. The simplest choice for :math:`\alpha` is :math:`\alpha_i = N_a`
for all :math:`i`. However, intuitively we expect that choosing
:math:`\alpha_i` in a decreasing order can improve the performance of the
method. In this case, we start assimilating data with a large value of
:math:`\alpha`, which reduces the magnitude of the initial updates; then, we
gradually decrease :math:`\alpha`.


Reservoir Model
---------------

The implemented small 2D Reservoir Simulator was created by following the
course material of **AESM304A - Flow and Simulation of Subsurface processes**
at Delft University of Technology (TUD); this particular part was taught by Dr.
D.V. Voskov, https://orcid.org/0000-0002-5399-1755.
