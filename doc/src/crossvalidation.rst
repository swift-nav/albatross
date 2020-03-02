###############
Crossvalidation
###############

.. _crossvalidation:

If you're using this package it's probably safe to say you're interested in building
a statitical model and plan on training with some data and making predictions
for another.  During the process of you'll invariable end up wondering how well the model work.  It could be that you have multiple
models and wnat to know which model will work best.  Or could simply be that you're curious how well your model will work in real life.

This is where `cross validation`_ steps in. The motivation is relatively simple:  Fitting your model and testing it on the same data is cheating.
Cross validation is an approach for getting a more representative estimate of a model's generalization error, or the
ability of your model to predict things it hasn't seen.  The process consists of spliting your data
into groups, holding one group out, fitting a model to the remaining data and predicting the
held out group, then repeating for all groups.  The result will be out of sample predictions for
all the data in your dataset which can be compared to the truth and used with your favorite metrics

+++++++++++++++++++
Example Usage
+++++++++++++++++++

.. code-block:: c

  // Get the cross validated marginal predictions in the original order;
  model.cross_validate().predict(dataset, get_group).marginal();
  // Get the cross validated marginal predictions by group;
  model.cross_validate().predict(dataset, get_group).marginals();


.. _`cross validation`: https://en.wikipedia.org/wiki/Cross-validation_(statistics)
.. _`scikit`: https://scikit-learn.org/stable/modules/cross_validation.html
