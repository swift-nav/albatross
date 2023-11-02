###############
Crossvalidation
###############

.. _crossvalidation:

If you're using this package it's probably safe to say you're interested in building
a statitical model and plan on training with some data and making predictions
for another.  During the process of you'll invariable end up wondering how well the model work.  It could be that you have multiple
models and want to know which model will work best.  Or could simply be that you're curious how well your model will work in real life.

This is where `cross validation`_ steps in. The motivation is relatively simple:  Fitting your model and testing it on the same data is cheating.
Cross validation is an approach for getting a more representative estimate of a model's generalization error, or the
ability of your model to predict things it hasn't seen.  The process consists of spliting your data
into groups, holding one group out, fitting a model to the remaining data and predicting the
held out group, then repeating for all groups.  The result will be out of sample predictions for
all the data in your dataset which can be compared to the truth and used with your favorite metrics

There is a ``cross_validate()`` helper which is available for any model you build in albatross. You provide a function ``get_group`` which takes a single feature (``dataset.features[i]``) and returns the group it belongs too.

.. code-block:: c

  get_group = [](const FeatureType &f) -> GroupKey {
    return f.something;
  };  
  

Then you can ask for leave one group out predictions from your model, returned in the original order (which makes it easy to compare directly to the truth dataset),

.. code-block:: c

  // Get the cross validated marginal predictions in the original order;
  MarginalDistribution cv_pred = model.cross_validate().predict(dataset, get_group).marginal();


Or you can ask for predictions organized by group,

.. code-block:: c

  std::map<GroupKey, MarginalDistribution> cv_preds =
           model.cross_validate().predict(dataset, get_group).marginals();
  std::map<GroupKey, JointDistribution> cv_preds =
           model.cross_validate().predict(dataset, get_group).joints();


.. _`Gaussian Processes for Machine Learning`: http://gaussianprocess.org/gpml/chapters/RW.pdf

++++++++++++++++++++++++++++++++++++++++++++++++++++
Efficient Cross Validation with Gaussian Processes
++++++++++++++++++++++++++++++++++++++++++++++++++++

In `Gaussian Processes for Machine Learning`_ (Section 5.4.2) they describe an efficient way for making leave one out predictions, here we expand that same trick to enable making leave one group out predictions.

Consider the case where we have a set of observations, :math:`y`, and we would like to make leave one group out cross validated predictions and by groups we mean independent sets of one or more variables.

We start with our GP,

.. math::

    \mathbf{f} \sim \mathcal{N}\left(0, \Sigma \right)

Which we can then break into groups,

.. math::

    \begin{bmatrix} \mathbf{\hat{y}} \\ \mathbf{y_i} \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} 0 \\ 0 \end{bmatrix}, \begin{bmatrix}\hat{\Sigma} & S \\ S^T & C \end{bmatrix}\right)

Where we will be using a subset of observations, :math:`\hat{y}` to make predictions for a held out set of locations, :math:`x_i`.  We can do this directly using the Gaussian process predict formula,

.. math::

    [\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] \sim \mathcal{N}\left(S^T \hat{\Sigma}^{-1} \hat{y}, C - S^T \hat{\Sigma}^{-1} S\right)

But doing so would require computing :math:`\hat{\Sigma}^{-1}` for every group, :math:`i`, that we hold out.  So if we're doing leave one out with :math:`n` observations we have to do the :math:`\mathcal{O}(n^3)` inversion :math:`n` times leading to :math:`\mathcal{O}(n^4)` complexity which will quickly get infeasible.

However, in the process of fitting our GP we'll need to end up computing the inverse of the full covariance, :math:`\Sigma^{-1}` as well as what we've been calling the information vector, :math:`v = \Sigma^{-1} y`.  By using block inversion we get,

.. math::

    \Sigma^{-1} = \begin{bmatrix}
    \left(\hat{\Sigma} - S C^{-1} S^T\right)^{-1} & -\left(\hat{\Sigma} - S C^{-1} S^T\right)^{-1}SC^{-1} \\
    -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} & \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}\end{bmatrix}

And if we break up :math:`v` into :math:`[\hat{v} \hspace{8pt} v_i]` using the same partitioning as :math:`y` we see,

.. math::

    v_i & = \left[\Sigma^{-1} y\right]_i \\
    & = \begin{bmatrix}
    -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} & \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}
    \end{bmatrix} \begin{bmatrix} \hat{y} \\ y_i \end{bmatrix} \\
    & = -\left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} S^T \hat{\Sigma}^{-1} \hat{y} + \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1} y_i \\
    & = -A S^T \hat{\Sigma}^{-1} \hat{y} + A y_i

Where :math:`A = \left(C - S^T \hat{\Sigma}^{-1} S\right)^{-1}` is the lower right corner of :math:`\Sigma^{-1}` and :math:`A^{-1}` is the leave one out prediction covariance. Notice that if we multiply :math:`v_i` through by :math:`A^{-1}` we end up with,

.. math::

    A^{-1} v_i &= - S^T \hat{\Sigma}^{-1} \hat{y} + y_i \\
    &= -\mbox{E}[\mathbf{y_i}|\hat{y}] + y_i \\
    \mbox{E}[\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] &= y_i - A^{-1} v_i

We can then recover the leave one out predictions,

.. math::

    [\mathbf{y_i}|\mathbf{\hat{y}}=\hat{y}] \sim \mathcal{N}\left(y_i - A^{-1} v_i, A^{-1}\right)

+++++++++++++++++++++++++++++++++
Computing :math:`A`
+++++++++++++++++++++++++++++++++

Above we see that if we can compute :math:`A` then we can recover the leave one out predictions without ever directly computing :math:`\hat{\Sigma}^{-1}`.  Take the case of leave one observation out, in this case :math:`A` will be the last diagonal value of :math:`\Sigma^{-1}`.  When training a Gaussian process we'll often have a decomposition of :math:`\Sigma` laying around, typically :math:`\Sigma = LDL^T`.  To get the :math:`i^{th}` diagonal value of :math:`\Sigma^{-1}` we can first compute, :math:`q = D^{-1/2} L^{-1} e_i`, where :math:`e_i` is a vector of zeros with a one in element :math:`i`, then we find that :math:`\Sigma^{-1}_{ii} = q^T q`.  Since :math:`L` is lower triangular and :math:`D` is diagonal :math:`p` can be computed efficiently.

Similarly if we're making leave one group out predictions we can build an indexing matrix :math:`E_i` which consists of columns :math:`e_j` for each :math:`j` in group :math:`i`.  Then we find that,

.. math::

    A = Q^T Q

with

.. math::

    Q = D^{-1/2} L^{-1} E_i.

Where :math:`L^{-1} E_i` amounts to extracting columns of :math:`L^{-1}`.

.. _`cross validation`: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
.. _`scikit`: https://scikit-learn.org/stable/modules/cross_validation.html
