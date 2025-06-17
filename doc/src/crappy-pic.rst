#################################################
PIC math scratchpad
#################################################

.. _crappy-pic:

-------------------------------------------------------------------
Relationship between :math:`\sigma_\cancel{B}` and :math:`\sigma_f`
-------------------------------------------------------------------

It seems like what we need first of all is some way to get the PITC posterior variance based only on :math:`\cancel{B}` using only the full PITC posterior and calculations that scale only with :math:`|B|`.  I'm not totally finished working out how :math:`(\sigma^2_{*})^{\cancel{B} \cancel{B}}` fits into the whole posterior :math:`(\sigma^2_*)^{PIC}`

Observe that since :math:`\mathbf{\Lambda}` is a block-diagonal matrix, its inverse is too, so when it's in the middle of a quadratic form, the cross terms disappear.  As far as I can tell, this is the only place we can additively separate :math:`\mathbf{Q}_{\cancel{B} \cancel{B}}` and :math:`\mathbf{K}_{B B}` without explicit cross terms.  Start with the usual application of Woodbury's lemma to the PITC posterior covariance:

.. math::

            \newcommand{Lam}{\mathbf{\Lambda}}
            \newcommand{Kuu}{\mathbf{K}_{uu}}
            \newcommand{Kuf}{\mathbf{K}_{uf}}
            \newcommand{Kfu}{\mathbf{K}_{fu}}
            (\sigma_*^2) &= K_* - \mathbf{Q}_{* f} \left(\mathbf{K}_{fu} \mathbf{K}_{uu}^{-1} \mathbf{K}_{uf} + \mathbf{\Lambda} \right)^{-1} \mathbf{Q}_{f *} \\
            &= K_* - \mathbf{K}_{*u} \Kuu^{-1} \Kuf \left(\Kfu \Kuu^{-1} \Kuf + \Lam\right)^{-1} \Kfu \Kuu^{-1} \mathbf{K}_{u*} \\
            &= K_* - \mathbf{K}_{*u} \Kuu^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \left( \Kuu + \Kuf \Lam^{-1} \Kfu \right)^{-1} \mathbf{K}_{u*}

Now break up :math:`\Kuf` and :math:`\Lam`:

.. math::

            \Kuf &= \begin{bmatrix} \mathbf{K}_{u \cancel{B}} & \mathbf{K}_{u B} \end{bmatrix} \\
            \Lam &= \begin{bmatrix} \Lam_{\cancel{B}} & \mathbf{0} \\ \mathbf{0} & \Lam_{B} \end{bmatrix} \\

so that

.. math::

            \Kuf \Lam^{-1} \Kfu &= \begin{bmatrix} \mathbf{K}_{u \cancel{B}} & \mathbf{K}_{u B} \end{bmatrix}
            \begin{bmatrix} \Lam_{\cancel{B}}^{-1} & \mathbf{0} \\ \mathbf{0} & \Lam_{B}^{-1} \end{bmatrix}
            \begin{bmatrix} \mathbf{K}_{\cancel{B} u} \\ \mathbf{K}_{B u} \end{bmatrix} \\
            &= \mathbf{K}_{u \cancel{B}} \Lam_{\cancel{B}}^{-1} \mathbf{K}_{\cancel{B} u} + 
            \mathbf{K}_{u B} \Lam_{B}^{-1} \mathbf{K}_{B u} \\
            \mathbf{K}_{u \cancel{B}} \Lam_{\cancel{B}}^{-1} \mathbf{K}_{\cancel{B} u} &= \Kuf \Lam^{-1} \Kfu - \mathbf{K}_{u B} \Lam_{B}^{-1} \mathbf{K}_{B u}

This last one is the inverse term in the training-data dependent part of the PITC posterior covariance using only the training points in :math:`\cancel{B}`, which I think is what we want to combine with the dense posterior from the training points in :math:`B`:

.. math::
            (\sigma_*^2)^{\cancel{B} \cancel{B}} = K_* - \mathbf{K}_{*u} \Kuu^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \left( \Kuu + \mathbf{K}_{u \cancel{B}} \Lam_{\cancel{B}}^{-1} \mathbf{K}_{\cancel{B} u} \right)^{-1} \mathbf{K}_{u*}

which we can write in terms of only the full PITC prior and the prior for group :math:`B`:

.. math::
            (\sigma_*^2)^{\cancel{B} \cancel{B}} = K_* - \mathbf{K}_{*u} \Kuu^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \left( \Kuu + \Kuf \Lam^{-1} \Kfu - \mathbf{K}_{u B} \Lam_{B}^{-1} \mathbf{K}_{B u} \right)^{-1} \mathbf{K}_{u*}

If you set :math:`B = \emptyset`, removing the correction for the training points in block :math:`B` entirely, you have the original PITC posterior covariance based on the full set of training points :math:`f`, as expected.  Now we only have to compute :math:`\Kuf \Lam^{-1} \Kfu` (the term that scales with the full training set :math:`f`) once, and nothing that scales with :math:`|\cancel{B}|`.  This still involves an :math:`O(M^3)` decomposition for each group :math:`B`.  I have been assuming our group size :math:`|B|` is significantly smaller than the number of inducing points :math:`M`, so it's not great, but an interesting thing happens if we use the "downdate" form of Woodbury's lemma in the opposite direction to the usual way:

.. math::
            \left(A - B C^{-1} B^T \right)^{-1} &= A^{-1} + A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\

Letting :math:`\mathbf{A} = \Kuu + \Kuf \Lam^{-1} \Kfu` (playing the part of :math:`A` in Woodbury's lemma):

.. math::

            (\sigma_*^2)^{\cancel{B} \cancel{B}} &= K_* - \mathbf{K}_{*u} \Kuu^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \left( \mathbf{A}^{-1} + \mathbf{A}^{-1} \mathbf{K}_{u B} \left(\Lam_{B} - \mathbf{K}_{B u} \mathbf{A}^{-1} \mathbf{K}_{u B}\right)^{-1} \mathbf{K}_{B u} \mathbf{A}^{-1} \right) \mathbf{K}_{u*} \\
            &= K_* - \mathbf{K}_{*u} \Kuu^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \mathbf{A}^{-1} \mathbf{K}_{u*} + \mathbf{K}_{*u} \mathbf{A}^{-1} \mathbf{K}_{u B} \left(\Lam_{B} - \mathbf{K}_{B u} \mathbf{A}^{-1} \mathbf{K}_{u B}\right)^{-1} \mathbf{K}_{B u} \mathbf{A}^{-1} \mathbf{K}_{u*} \\

I find the use of Woodbury's lemma to downdate a bit uncomfortable -- does this work with real symmetric matrices?  But we would only need to decompose :math:`\mathbf{A}` (and also :math:`\Kuu`) at a cost of :math:`O(M^3)` once; the inversion to be calculated per group costs :math:`O(|B|^3)`.  I also don't immediately spot a slick QR decomposition way to do this per-group term; it's pretty symmetrical.  The only thing that jumps out is precomputing :math:`\mathbf{K}_{B u} \mathbf{A}^{-\frac{1}{2}}`.

----------------------------------------------
Derivation of Woodbury's Lemma for Differences
----------------------------------------------

This is specifically for symmetric matrices, to convince myself the downdate formula is legit.  We would like to compute a rank-:math:`k` downdate to the inverse of a matrix :math:`A`.  

.. math::
            \left(A - B C^{-1} B^T \right)^{-1} &= A^{-1} + A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\

To be extra clear, let's say :math:`A \in \mathbb{R}^{n \times n}`, :math:`B \in \mathbb{R}^{n \times k}`, :math:`C \in \mathbb{R}^{k \times k}`.  We put the relevant matrices into a block matrix :math:`M \in \mathbb{R}^{m \times m}` for :math:`m = n + k`, and label the conforming blocks of its inverse:

.. math::
            M &= \begin{pmatrix} A & B \\ B^T & C\end{pmatrix} \\
            M^{-1} &= \begin{pmatrix} W & X \\ X^T & Y\end{pmatrix} \\
            M M^{-1} &= \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \\
            M^{-1} M &= \begin{pmatrix} \mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}

The blocks of :math:`M M^{-1}` yield the following equations:
            
.. math::
            AW + BX^T &= \mathbf{I} \\
            AX + BY &= \mathbf{0} \\
            B^T W + CX^T &= \mathbf{0} \\
            B^T X + CY &= \mathbf{I} \\

Rearrange the middle two:

.. math::
            X &= -A^{-1} B Y \\
            X^T &= -C^{-1} B^T W

If we do the same for :math:`M^{-1} M`:

.. math::
            X &= -W B C^{-1} \\
            X^T &= -Y B^T A^{-1}

These blocks are equal:

.. math::
            C^{-1} B^T W &= Y B^T A^{-1} \\
            A^{-1} B Y &= W B C^{-1} \\

Now use the middle two equations from :math:`M M^{-1}` and plug them into the first and last equations irrespectively:

.. math::
            W - A^{-1} B C^{-1} B^T W &= A^{1} \\
            \left(I - A^{-1} B C^{-1} B^T\right) W &= A^{-1} \\
            -C^{-1} B^T A B Y + Y &= C^{-1} \\
            \left(I - C^{-1} B^T A^{-1} B\right)Y &= C^{-1} \\

Assuming :math:`\left(I - A^{-1} B C^{-1} B^T\right)` and :math:`\left(I - C^{-1} B^T A^{-1} B\right)` are invertible, rearrange:

.. math::
            W &= \left(I - A^{-1} B C^{-1} B^T\right)^{-1} A^{-1} \\
            &= \left(A - B C^{-1} B^T\right)^{-1} \\
            Y &= \left(I - C^{-1} B^T A^{-1} B\right)^{-1} C^{-1} \\
            &= \left(C - B^T A^{-1} B\right)^{-1}

Now use equality of the off-diagonal blocks from the two ways to multiply :math:`M` and :math:`M^{-1}` above (don't you wish Sphinx would number equations?) and substitute:

.. math::
            C^{-1} B^T \left(A - B C^{-1} B^T\right)^{-1} &= \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\
            \left(A - B C^{-1} B^T\right)^{-1} B C^{-1} &= A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} \\

Let's look just at the term :math:`\left(A - B C^{-1} B^T\right)` and right-multiply by :math:`-A^{-1}`:

.. math::
            \left(A - B C^{-1} B^T\right) \left(-A^{-1}\right) &= -\mathbf{I} + B C^{-1} B^T A^{-1} \\
            \mathbf{I} + \left(A - B C^{-1} B^T\right) \left(-A^{-1}\right) &= B C^{-1} B^T A^{-1}

Now let's return to the previous equation and right-multiply by :math:`B^T A^{-1}`:

.. math::
            \left(A - B C^{-1} B^T\right)^{-1} B C^{-1} B^T A^{-1} &= A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\

Substitute the previous result for :math:`B C^{-1} B^T A^{-1}`:

.. math::
            \left(A - B C^{-1} B^T\right)^{-1} \left( \mathbf{I} + \left(A - B C^{-1} B^T\right) \left(-A^{-1}\right) \right) &= A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\
            \left(A - B C^{-1} B^T\right)^{-1} - A^{-1} &= A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \\
            \left(A - B C^{-1} B^T \right)^{-1} &= A^{-1} + A^{-1} B \left(C - B^T A^{-1} B\right)^{-1} B^T A^{-1} \blacksquare

which is the difference form of Woodbury's lemma, or a rank-:math:`k` downdate of :math:`A^{-1}`.  The main assumption seems to be that :math:`\left(I - A^{-1} B C^{-1} B^T\right)` and :math:`\left(I - C^{-1} B^T A^{-1} B\right)` are invertible.

------------------------
Blockwise inversion of S
------------------------

I tried to invert :math:`\mathbf{S}` blockwise; it didn't work out but I'm leaving it in here just to look back at:

.. math::
            \newcommand{VV}{\mathbf{V}}
            \mathbf{U} &= \mathbf{Q}_{* \cancel{B}} \mathbf{S}^{-1}_{\cancel{B} B} \VV_{B *} \\
            &= \mathbf{Q}_{* \cancel{B}} \left( \mathbf{Q}_{\cancel{B} \cancel{B}}^{-1} \mathbf{Q}_{\cancel{B} B} \left(\mathbf{Q}_{B B} - \mathbf{Q}_{B \cancel{B}} \mathbf{Q}_{\cancel{B} \cancel{B}}^{-1} \mathbf{Q}_{\cancel{B} B}\right)^{-1} \right) \VV_{B *}
            

How are we going to avoid the :math:`O(|\cancel{B}|^3)` inversion for :math:`\mathbf{Q}_{\cancel{B} \cancel{B}}`?  Also we are going to get cross-terms with both :math:`\mathbf{Q}_{* \cancel{B}}` and :math:`\mathbf{Q}_{B B}` involved, but maybe they are OK because they are only :math:`O(|\cancel{B}| |B|)`?