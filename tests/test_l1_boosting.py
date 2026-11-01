"""Checks for the independent regularized boosting implementation.
Run from the project root: python -m unittest discover -s tests
"""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from l1_boosting import L1HistogramBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

class BoostingChecks(unittest.TestCase):
    def test_leaf_score_minimizes_penalized_quadratic(self):
        X=np.arange(10,dtype=float).reshape(-1,1);y=np.arange(10,dtype=float)
        model=L1HistogramBoostingRegressor(reg_alpha=2,n_estimators=1,
            learning_rate=1,max_depth=1,max_bins=10,min_samples_leaf=2).fit(X,y)
        tree=model.trees_[0]
        self.assertIn('feature',tree)
        for side in ['left','right']:
            leaf=tree[side]
            mask=(X[:,0]<=tree['threshold']) if side=='left' else (X[:,0]>tree['threshold'])
            residual=(y-y.mean())[mask];w=leaf['value']
            def objective(value):return .5*np.sum((residual-value)**2)+2*abs(value)
            for offset in [-.5,-.01,.01,.5]:
                self.assertLessEqual(objective(w),objective(w+offset)+1e-12)

    def test_strong_penalty_means_no_tree_updates(self):
        X=np.arange(10).reshape(-1,1);y=np.arange(10,dtype=float)
        model=L1HistogramBoostingRegressor(reg_alpha=1e9,n_estimators=3).fit(X,y)
        np.testing.assert_allclose(model.predict(X),y.mean())
        self.assertTrue(all('feature' not in tree for tree in model.trees_))

    def test_alpha_zero_matches_exact_unpenalized_reference(self):
        rng=np.random.default_rng(9);X=rng.normal(size=(90,4))
        y=X[:,0]+np.sin(X[:,1])+rng.normal(size=90)*.1
        actual=L1HistogramBoostingRegressor(reg_alpha=0,n_estimators=20,learning_rate=.1,
            max_depth=2,max_bins=128,min_samples_leaf=3).fit(X,y)
        reference=GradientBoostingRegressor(n_estimators=20,learning_rate=.1,
            max_depth=2,min_samples_leaf=3,random_state=552).fit(X,y)
        np.testing.assert_allclose(actual.predict(X),reference.predict(X),atol=1e-10)

    def test_sklearn_pipeline_cv_interface(self):
        rng=np.random.default_rng(552);X=rng.normal(size=(100,6))
        y=2*X[:,0]+rng.normal(size=100)*.1
        pipeline=Pipeline([('impute',SimpleImputer()),
            ('model',L1HistogramBoostingRegressor(n_estimators=10,max_depth=2))])
        scores=cross_val_score(pipeline,X,y,cv=3,scoring='neg_mean_squared_error',error_score='raise')
        self.assertTrue(np.isfinite(scores).all())

if __name__=='__main__':unittest.main()
