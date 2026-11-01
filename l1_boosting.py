"""Small, deterministic squared-error boosting trees with an L1 leaf penalty.

Each tree minimizes a second-order squared-loss approximation plus
alpha * sum(abs(leaf_score)) + 0.5 * reg_lambda * sum(leaf_score**2).
The structure is grown greedily using regularized split gains over quantile
histograms. This implements the regularized tree objective used in XGBoost;
it is an independent teaching implementation, not the XGBoost package.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class L1HistogramBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, reg_alpha=0.0, n_estimators=200, learning_rate=0.05,
                 max_depth=3, max_bins=32, min_samples_leaf=10, reg_lambda=0.0,
                 min_split_gain=1e-12):
        self.reg_alpha=reg_alpha
        self.n_estimators=n_estimators
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.max_bins=max_bins
        self.min_samples_leaf=min_samples_leaf
        self.reg_lambda=reg_lambda
        self.min_split_gain=min_split_gain

    @staticmethod
    def _soft_threshold(value, alpha):
        return np.sign(value)*np.maximum(np.abs(value)-alpha,0.0)

    def _tree_scores(self, bins, tree):
        scores=np.empty(len(bins),dtype=float)
        def fill(node, indices):
            if 'feature' not in node:
                scores[indices]=node['value'];return
            left=bins[indices,node['feature']]<=node['bin']
            fill(node['left'],indices[left])
            fill(node['right'],indices[~left])
        fill(tree,np.arange(len(bins)))
        return scores

    def fit(self,X,y):
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=float)
        if X.ndim!=2 or y.shape!=(len(X),) or not len(X):
            raise ValueError('Expected nonempty 2D X and 1D y.')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Impute missing values before fitting the booster.')
        if self.reg_alpha<0 or self.reg_lambda<0 or self.n_estimators<1:
            raise ValueError('Penalties must be nonnegative and n_estimators positive.')
        if self.max_bins<2 or self.min_samples_leaf<1 or self.max_depth<0:
            raise ValueError('Invalid tree configuration.')
        self.n_features_in_=X.shape[1]
        self.bin_thresholds_=[]
        for j in range(X.shape[1]):
            unique=np.unique(X[:,j])
            if len(unique)<=self.max_bins:
                cuts=(unique[:-1]+unique[1:])/2
            else:
                cuts=np.unique(np.quantile(X[:,j],np.arange(1,self.max_bins)/self.max_bins))
                cuts=cuts[(cuts>unique[0]) & (cuts<unique[-1])]
            self.bin_thresholds_.append(cuts)
        bins=self._bin_data(X)
        self.base_score_=float(y.mean())
        prediction=np.full(len(y),self.base_score_)
        self.trees_=[];self.training_mse_=[]
        importance=np.zeros(self.n_features_in_)
        offsets=np.arange(self.n_features_in_,dtype=np.int64)*self.max_bins
        alpha=float(self.reg_alpha);lam=float(self.reg_lambda)

        for _ in range(self.n_estimators):
            residual=y-prediction
            def grow(indices,depth):
                n=len(indices);G=float(residual[indices].sum())
                node={'value':float(self._soft_threshold(G,alpha)/(n+lam)), 'samples':n}
                if depth>=self.max_depth or n<2*self.min_samples_leaf:
                    return node
                encoded=(bins[indices].astype(np.int64)+offsets).ravel()
                size=self.n_features_in_*self.max_bins
                counts=np.bincount(encoded,minlength=size).reshape(self.n_features_in_,self.max_bins)
                sums=np.bincount(encoded,weights=np.repeat(residual[indices],self.n_features_in_),
                                 minlength=size).reshape(self.n_features_in_,self.max_bins)
                left_n=counts.cumsum(axis=1)[:,:-1]
                left_G=sums.cumsum(axis=1)[:,:-1]
                right_n=n-left_n;right_G=G-left_G
                valid=(left_n>=self.min_samples_leaf) & (right_n>=self.min_samples_leaf)
                # Denominators for invalid empty leaves are protected only to
                # avoid division warnings; those candidates are excluded below.
                gain=.5*(self._soft_threshold(left_G,alpha)**2/np.maximum(left_n+lam,1e-15)
                    +self._soft_threshold(right_G,alpha)**2/np.maximum(right_n+lam,1e-15)
                    -self._soft_threshold(G,alpha)**2/(n+lam))
                gain[~valid]=-np.inf
                j,k=np.unravel_index(np.argmax(gain),gain.shape)
                best_gain=float(gain[j,k])
                if best_gain<=self.min_split_gain or not np.isfinite(best_gain):
                    return node
                left=bins[indices,j]<=k
                assert left.any() and (~left).any()
                importance[j]+=best_gain
                node.update(feature=int(j),bin=int(k),gain=best_gain,
                            threshold=float(self.bin_thresholds_[j][k]),
                            left=grow(indices[left],depth+1),
                            right=grow(indices[~left],depth+1))
                return node
            tree=grow(np.arange(len(y)),0)
            self.trees_.append(tree)
            prediction+=self.learning_rate*self._tree_scores(bins,tree)
            self.training_mse_.append(float(np.mean((y-prediction)**2)))
        self.feature_importances_=importance/importance.sum() if importance.sum() else importance
        return self

    def _bin_data(self,X):
        return np.column_stack([np.searchsorted(cuts,X[:,j],side='left')
                                for j,cuts in enumerate(self.bin_thresholds_)]).astype(np.int16)

    def predict(self,X):
        check_is_fitted(self,'trees_')
        X=np.asarray(X,dtype=float)
        if X.ndim!=2 or X.shape[1]!=self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError('X has invalid shape or nonfinite values.')
        bins=self._bin_data(X)
        prediction=np.full(len(X),self.base_score_)
        for tree in self.trees_:
            prediction+=self.learning_rate*self._tree_scores(bins,tree)
        return prediction

    def save_model(self,path):
        check_is_fitted(self,'trees_')
        Path(path).write_text(json.dumps({'implementation':'L1HistogramBoostingRegressor',
            'parameters':self.get_params(),'base_score':self.base_score_,
            'thresholds':[x.tolist() for x in self.bin_thresholds_],'trees':self.trees_},indent=1))
