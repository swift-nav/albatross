#################################################
Iterative GP Details
#################################################

.. _iterative-gp-implementation:


----------------
Introduction
----------------

This page details the algorithms used in the CGGP system for doing GP inference using iterative methods (specifically, conjugate gradient (CG) and friends).

The core calculations needed for GP inference are

 * Predictive means :math:`y_* = K_{*x} K_{xx}^-1 y`

 * Predictive covariances: :math:`\Sigma_{**} = K_{**} - K_{*x} K_{xx}^{-1} K_{x*}`

 * The special case of predictive variances, :math:`\mathtt{diag}(\Sigma_{**})`

In some cases, we may also be interested in :math:`\log\left|K_{xx}\right|` to allow us to compute in-sample training likelihood.  For now we assume the reader is rigorously training against cross-validated or held-out data and focus on the 3 inference calculations above.

-------
Methods
-------

This approach is drawn directly from 

   [1] GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration
   Jacob R. Gardner, Geoff Pleiss, David Bindel, Kilian Q. Weinberger and Andrew Gordon Wilson

   [2] Constant-Time Predictive Distributions for Gaussian Processes
   Geoff Pleiss, Jacob R. Gardner, Kilian Q. Weinberger and Andrew Gordon Wilson

A few important contrasts:

 * We are not running on GPUs at the moment.  This means our dense matrix * vector operations are less efficient than in the paper.

 * We do not provide an implementation of SKI [#ski-sparse]_.  This means we do not have the constant-fill W matrix from [2], and therefore we don't compute predictive uncertainties in constant time.  We still try to compute predictive uncertainties efficiently.

.. [#ski-sparse] You can define an SKI-like approximation using Albatross' Sparse GP model, but that model uses direct methods for linear algebra and won't do this same LOVE calculation.
   
However there are still potential speedups to be had from these methods.  Iterative solution may be faster if:

 - accuracy requirements are low

 - you have past solutions with which to initialise the iterative process

 - you have less stringent accuracy requirements for predictive uncertainties than for means

 - you have a relatively sparse covariance so that you can multiply vectors faster than a full dense matrix

   