# Analysis and data repository for Sparks, Liao, et al., NatComms (2020)

This repository contains code and links to dataset from Sparks, Liao, et al., NatComms (2020). This code has been refactored to improve reproducibility. Closed source and internal toolkits have been replaced with open-source equivalents.

### Analysis code for Latent Ensemble Recruitment

#### Prerequisites

* [Python](http://python.org) 3.7
* [numpy](http://www.scipy.org) >= 1.18.1
* [scikit-learn](http://scikit-learn.org) >= 0.22.1
* [seaborn](https://seaborn.pydata.org/) >= 0.10.1
* [matplotlib](https://matplotlib.org/) >= 3.1.3

#### Description
The LER model is written according to a scikit-learn interface. It is simply a modification of the LatentDirichletAllocation class provided in scikit-learn, which it subclasses.

To initialize the model:

```
from latent_ensemble_recruitment import LatentEnsembleRecruitment
ler = LatentEnsembleRecruitment(K=6, **kwargs)

```

The following initialization parameters are defined by the user

* _K_ : int, optional (default=6)
    Number of ensembles
* _method_ : 'bootstrap'|'lda' (default='bootstrap')
    Method to use for inferring cell recruitment. Currently only 
    'bootstrap' is implemented. If 'lda', assumes the data is already
    binarized and simply performs vanilla LDA.
* _n_bootstraps_ : int, optional (default=10000)
    Number of samples to take to construct null distribution
* _percentile_ : float, optional (default=95.)
    Percentile to threshold "recruited" cells
* **_lda_kwargs_ 
    Keyword arguments to pass to sklearn.decomposition.LatentDirichletAllocation

You can either provide your own binarized dataset, or you can provide a continuous-valued dataset which will be binarized to the _percentile_ of the baseline via a resampling procedure. 

To fit the model to data
```
ler.fit(X, y)
```

To view the learned ensembles
```
ler.ensembles_
```

To sample _T_ events in _N_ cells from the model distribution
```
X = ler.sample(self, N, T, y=None, std=1., mu_active=3., alpha=None, beta=None, coactivation_gain=1.)
```

See https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html for parameters and methods inherited from LDA.

#### Authors
*zhenrui DOT liao AT columbia DOT edu

### Access to the dataset:

The dataset will be made available soon

## License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

