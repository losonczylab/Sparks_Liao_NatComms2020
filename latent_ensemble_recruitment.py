import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

class LatentEnsembleRecruitment(LatentDirichletAllocation):
    def __init__(self, K=6, method='bootstrap', n_bootstraps=10000, percentile=99., **lda_kwargs):
        """Latent Ensemble Recruitment, based on Latent Dirichlet Allocation.
        Infers latent ensembles in a "responsiveness" dataset with the topic modeling analogies
        cell ~ "word", event ~ "document", ensemble ~ "topic".
        Unlike LDA, the "words" are unknown a priori and are also inferred.
        
        Inputs
        -------
        K : int, optional (default=6)
            Number of ensembles
        method : 'bootstrap'|'lda' (default='bootstrap')
            Method to use for inferring cell recruitment. Currently only 
            'bootstrap' is implemented. If 'lda', assumes the data is already
            binarized and simply performs vanilla LDA.
        n_bootstraps : int, optional (default=10000)
            Number of samples to take to construct null distribution
        percentile : float, optional (default=95.)
            Percentile to threshold "recruited" cells
        **lda_kwargs 
            Keyword arguments to pass to sklearn.decomposition.LatentDirichletAllocation
        
        See also
        -------
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
        """
        self.K = K
        self.method = method
        if not (self.method == 'bootstrap' or self.method == 'lda'):
            raise NotImplementedError("Only the `bootstrap` and `lda` inference methods are currently implemented")
        self.percentile = percentile
        self.n_bootstraps = n_bootstraps
        
        super().__init__(n_components=K, **lda_kwargs)
    
    def fit(self, X, y):
        """
        Learn model for the data X with variational Bayes method.

        When learning_method is ‘online’, use mini-batch update. Otherwise, use batch update
        
        Inputs
        -------
        X : array-like, shape=(n_timesteps, n_cells)
            Cell responsiveness array (e.g. dFOF), of shape n_timesteps x n_cells
        y : boolean array, shape=(n_timesteps,)
            Boolean array indicating where the events occur. 
            True = event, False = nonevent
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        #self.baseline = self._calculate_baseline(X, y)
        X_binarized = self._binarize(X, y)
        return super().fit(X_binarized[y])
    
    def transform(self, X, y):
        X_binarized = self._binarize(X, y)
        X_transformed = super().transform(X_binarized[y])
        
        output = np.zeros((X_binarized.shape[0], self.K))
        output[y] = X_transformed

        return output
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
            
    def _calculate_baseline(self, X, y):
        if self.method == 'bootstrap':
            nonevent = ~y
            baseline_X = X[nonevent]
            trigger_frames = np.random.choice(np.arange(len(baseline_X)), 
                                              size=self.n_bootstraps, 
                                              replace=True)
            bootstraps = baseline_X[trigger_frames]
            
            return np.nanpercentile(bootstraps, q=self.percentile, axis=0)
        elif self.method == 'lda':
            return np.zeros(X.shape[1])
    
    def _binarize(self, X, y=None):
        X = np.array(X)
        y = np.array(y)
        baseline = self._calculate_baseline(X, y)
        X_binarized = (X > baseline)
        
        X_binarized[~y] = None

        return X_binarized
    
    @property
    def ensembles_(self):
        return self.components_
    
    @property
    def true_activations_(self):
        return self._true_ensemble_activations    
    
    @property
    def true_ensembles_(self):
        return self._true_ensembles
    
    @property
    def true_recruitment_(self):
        return self._true_recruitment
        
    def sample(self, N, T, y=None, std=1., mu_active=3., alpha=None, beta=None, coactivation_gain=1.):
        """
        Sample from the model.

        Inputs
        -------
        N : int
            Number of cells
        T : int
            Number of timesteps
        y : (T,)-boolean array, or None
            Boolean array of events, or None. If None, assumes an event occurs
            at *every* sample.
            Note: we do not model the event process, though if you wish you 
            can sample y from a Poisson process.
        std : float, default=1.
            Standard deviation of responsiveness distribution
        mu_active : float, default=3.
            Mean responsiveness of an "activated" cell
        alpha : float, array-like, or None, optional
            Sparsity parameter of ??? Dirichlet. Default: Flat prior.
        beta : float, array-like, or None, optional
            Sparsity parameter of ensemble sampling Dirichlet. Default: Flat prior.
        coactivation_gain : float, optional. Default = 1.
            Modulates the recruitment threshold. This is not used in the paper
            but produces more biologically realistic samples by increasing the 
            likelihood of coactivation compared to straight marginalization. 
            
            A higher coactivation_gain reduces the sample complexity of 
            parameter inference.
            
        Returns
        -------
        X : (T,N) array of sample data
        y : (T,)-boolean array
        """
        
        if y is None:
            y = np.ones(T, dtype=bool)
        num_events = np.sum(y, dtype=int)
            
        if alpha is None:
            alpha = 1./self.K    
        alpha = np.ones(self.K)*alpha

        if beta is None:
            beta = 1./N
        beta = np.ones(N)*beta
        

        responsiveness = np.random.randn(num_events, N)*std
        
        # Sample the cell weights in each ensemble
        self._true_ensembles = np.random.dirichlet(alpha=beta, size=self.K)
        
        # Sample the ensemble weights in each event
        self._true_ensemble_activations = np.random.dirichlet(alpha=alpha, size=num_events)
        
        cell_activation_probabilities = self._true_ensemble_activations @ self._true_ensembles
        recruitment = np.random.rand(num_events, N) < cell_activation_probabilities*coactivation_gain
        self._true_recruitment = np.zeros((T,N), dtype=bool)
        self._true_recruitment[y] = recruitment
        
        for i in range(num_events):
            responsiveness[i, self._true_recruitment[i]] = np.random.randn(np.sum(self._true_recruitment[i]))*std + mu_active
        
        X = np.random.randn(T, N)*std
        X[y] = responsiveness
        
        return X, y