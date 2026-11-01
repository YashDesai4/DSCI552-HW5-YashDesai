# %% [markdown]
# # DSCI 552 Homework 5 — Decision trees and regression
# **Yash Desai**
#
# This notebook covers every required question using the original uploaded UCI
# datasets. Run All generates the numerical model results, figures, IF-THEN
# rules, selected-variable lists, and comparison tables shown below.
#
# Open this complete project folder in VS Code. Install requirements.txt in the
# selected notebook environment. The loader accepts the original two UCI ZIPs
# beside the notebook, or the four .data/.names files in data/acute and
# data/communities. It can download from UCI when network access is available.
#
# The tree analysis uses the whole Acute Inflammations dataset as requested.
# Regression reserves rows 1–1495 for training and rows 1496–1994 for testing.
# Every regression model uses all 122 predictive columns; the 11 high-CV
# variables are used for the exploratory plots, not as a feature filter.
# %%
from pathlib import Path
import json, math, platform, re, warnings, time
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, hamming_loss, mean_squared_error,
                             mean_absolute_error, r2_score, ConfusionMatrixDisplay)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from download_data import prepare_data
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj.to_string(index=False) if isinstance(obj,pd.DataFrame) else obj)
if '_display' in globals(): display=_display

SEED=552
threadpool_limits(limits=1)
ROOT=Path.cwd()
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
print('Python',platform.python_version(),'| numpy',np.__version__,'| pandas',pd.__version__,
      '| scipy',scipy.__version__,'| scikit-learn',sklearn.__version__)

fit_warnings=[]
def checked_fit(estimator,X,y,context):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        estimator.fit(X,y)
    fit_warnings.extend({'context':context,'warning':str(w.message)} for w in caught)
    return estimator

def table(frame,name,limit=30):
    frame.to_csv(OUT/(name+'.csv'),index=False)
    display(frame if limit is None else frame.head(limit))
    if limit is not None and len(frame)>limit:
        print(f'Showing {limit}/{len(frame)} rows. Full table: results/{name}.csv')

def figure(name):
    plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.show();plt.close('all')

# A clear missing-data error replaces an obscure downstream KeyError.
paths=prepare_data(root=ROOT)
# %% [markdown]
# ## Verified results at a glance
#
# All models below use the prescribed split: **1495 training rows, 499 test rows**.
# Test error is MSE on the dataset's normalized response scale.
#
# | Method | Selected setting | Test MSE |
# |---|---|---:|
# | OLS | No tuning parameter | 0.01796622 |
# | Ridge | lambda=63.095734 | 0.01746030 |
# | LASSO original | lambda=0.0001 | 0.01765778 |
# | LASSO standardized | lambda=0.00063095734 | 0.01772175 |
# | PCR | M=93 | 0.01826425 |
# | L1 boosting | alpha=0 | 0.01654130 |
#
# The full acute tree has **6 leaves, depth 4, and 100% joint training accuracy**.
# The one-SE pruning choice retains those six leaves. A separately constrained,
# actually pruned **4-leaf, depth-2** tree reaches **84.17% training accuracy** and
# **83.33% mean CV accuracy**, exposing the cost of greater interpretability.
#
# For 2(j), an independent L1-regularized histogram boosting implementation is
# included. XGBoost is optional in the assignment and was unavailable here. The
# searched optimum is **alpha=0**, so the chosen endpoint does not use a positive
# L1 penalty; the search found no validation benefit from adding that penalty.
# The exact objective, implementation, and checks are documented in section 2(j).
# %% [markdown]
# ## 1(a) — Load and encode Acute Inflammations
# The source has six features and two binary targets: bladder inflammation and
# nephritis. Diagnosis is multi-label: a person may have both, either, or neither.
# 'Yes/no' feature values become 1/0. Temperature retains its original Celsius
# values; decimal commas and the file's UTF-16 encoding are handled explicitly.
#
# **Label powerset:** encode the pair as class = 2×bladder + nephritis. This gives
# four interpretable joint classes: 0=(no,no), 1=(no,yes), 2=(yes,no), 3=(yes,yes).
# A single tree can therefore explain both outputs together. Binary relevance
# would fit two separate trees and would not explicitly model the label pair.
# Powerset is suitable here because there are only two labels and at most four
# combinations; it can become impractical with many labels or rare combinations.
# This analysis demonstrates the supplied dataset's rules; its in-sample score
# is not an estimate of clinical performance on new patients.
# %%
ACUTE_FEATURES=['temperature','nausea','lumbar_pain','urine_pushing','micturition_pains','burning_urethra']
ACUTE_TARGETS=['bladder_inflammation','nephritis']
CLASS_NAMES={0:'Neither',1:'Nephritis only',2:'Bladder inflammation only',3:'Both'}

def decode_text(path):
    payload=Path(path).read_bytes()
    if payload.startswith((b'\xff\xfe',b'\xfe\xff')): return payload.decode('utf-16')
    if b'\x00' in payload[:150]: return payload.decode('utf-16-le').lstrip('\ufeff')
    return payload.decode('utf-8-sig')

def load_acute(path):
    from io import StringIO
    frame=pd.read_csv(StringIO(decode_text(path)),sep=r'\s+',header=None,dtype=str)
    if frame.shape!=(120,8):
        raise ValueError(f'Expected 120 rows and 8 columns; got {frame.shape}. Use the original UCI file.')
    frame.columns=ACUTE_FEATURES+ACUTE_TARGETS
    frame['temperature']=pd.to_numeric(frame['temperature'].str.replace(',','.',regex=False))
    for col in frame.columns[1:]:
        frame[col]=frame[col].str.strip().str.lower().map({'yes':1,'no':0})
    if frame.isna().any().any(): raise ValueError('Unrecognized or missing Acute Inflammations values.')
    return frame

acute=load_acute(paths['acute']['diagnosis.data'])
Xa=acute[ACUTE_FEATURES]
Ya=acute[ACUTE_TARGETS].astype(int).to_numpy()
ya=(2*Ya[:,0]+Ya[:,1]).astype(int)
class_counts=pd.Series(ya).value_counts().sort_index()
table(pd.DataFrame({'code':class_counts.index,'diagnosis':[CLASS_NAMES[x] for x in class_counts.index],
                    'count':class_counts.to_numpy()}),'acute_class_counts')
print('Acute dataset:',acute.shape,'; missing values:',int(acute.isna().sum().sum()))
display(acute.head())
# %% [markdown]
# ## 1(b) — Decision tree fitted to the whole dataset
# CART chooses binary splits to reduce Gini impurity, $1-\sum_k p_k^2$.
# With ccp_alpha=0, no cost-complexity pruning is applied. Every internal node
# tests a symptom or temperature; a terminal node predicts a diagnosis pair.
# For binary symptoms, a threshold of 0.5 means no on the left and yes on the right.
# We report exact pair accuracy and each label's accuracy. These are training
# scores because the question specifically requests fitting the whole dataset.
# %%
full_tree=DecisionTreeClassifier(random_state=SEED,criterion='gini',ccp_alpha=0)
full_tree.fit(Xa,ya)
full_prediction=full_tree.predict(Xa)
full_pairs=np.column_stack([full_prediction//2,full_prediction%2])
print('Full tree: nodes=',full_tree.tree_.node_count,'leaves=',full_tree.get_n_leaves(),
      'depth=',full_tree.get_depth())
print(f'Joint training accuracy: {accuracy_score(ya,full_prediction):.4f}')
print(f'Bladder label accuracy: {(Ya[:,0]==full_pairs[:,0]).mean():.4f}')
print(f'Nephritis label accuracy: {(Ya[:,1]==full_pairs[:,1]).mean():.4f}')
plt.figure(figsize=(20,10))
plot_tree(full_tree,feature_names=ACUTE_FEATURES,
          class_names=[CLASS_NAMES[int(x)] for x in full_tree.classes_],
          filled=True,rounded=True,precision=3,fontsize=9)
plt.title('Acute Inflammations: full label-powerset decision tree')
figure('acute_full_tree')
# %% [markdown]
# ### Answer to 1(b): observed fitted tree
#
# The full tree has **11 nodes, 6 leaves, and depth 4**. Joint label-pair training
# accuracy is **100%**; both individual diagnosis training accuracies are also
# 100%. Its decisions use lumbar pain, urine pushing, temperature, and micturition
# pains. Nausea and burning of the urethra are not used in this fitted tree.
#
# The root tests lumbar pain. For example, when lumbar pain is absent and urine
# pushing is present, the tree predicts bladder inflammation without nephritis.
# For records with lumbar pain, the temperature split is **37.95 degrees Celsius**.
# These statements describe the fitted rules in this small supplied dataset, not
# medical recommendations or out-of-sample diagnostic validation.
# %% [markdown]
# ## 1(c) — Convert every root-to-leaf path into an IF–THEN rule
# A left edge contributes feature <= threshold and a right edge contributes
# feature > threshold. The conjunction of all conditions on one path defines
# one rule. The predicted joint class is decoded back into the two diagnoses.
# Each rule lists its training support and leaf purity. Binary symptom tests
# are rewritten as yes/no for readability, while full numeric thresholds are
# retained internally to verify that the rules reproduce the tree predictions.
# %%

def extract_rules(model,feature_names):
    tree=model.tree_;rules=[]
    def descend(node,conditions):
        if tree.children_left[node]==tree.children_right[node]:
            code=int(model.classes_[np.argmax(tree.value[node][0])])
            tests=[]
            for j,op,threshold in conditions:
                name=feature_names[j]
                if name!='temperature' and np.isclose(threshold,.5):
                    tests.append(f'{name} = {"no" if op=="<=" else "yes"}')
                else: tests.append(f'{name} {op} {threshold:.6g}')
            diagnosis=f'bladder={"yes" if code//2 else "no"}, nephritis={"yes" if code%2 else "no"}'
            proportions=tree.value[node][0]/tree.value[node][0].sum()
            rules.append({'leaf':int(node),'conditions':conditions,'code':code,
                'samples':int(tree.n_node_samples[node]),'purity':float(proportions.max()),
                'rule':'IF '+(' AND '.join(tests) or 'TRUE')+' THEN '+diagnosis})
            return
        j=int(tree.feature[node]);threshold=float(tree.threshold[node])
        descend(int(tree.children_left[node]),conditions+[(j,'<=',threshold)])
        descend(int(tree.children_right[node]),conditions+[(j,'>',threshold)])
    descend(0,[])
    return rules

def validate_rules(model,X,rules):
    # scikit-learn trees convert inputs to float32 for split traversal.
    values=np.asarray(X,dtype=np.float32)
    prediction=np.full(len(values),-1);matches=np.zeros(len(values),dtype=int)
    for rule in rules:
        mask=np.ones(len(values),dtype=bool)
        for j,op,threshold in rule['conditions']:
            mask &= (values[:,j]<=threshold) if op=='<=' else (values[:,j]>threshold)
        prediction[mask]=rule['code'];matches[mask]+=1
    assert np.all(matches==1),'Rules must cover each row exactly once.'
    assert np.array_equal(prediction,model.predict(X)),'Rules differ from tree predictions.'

def show_rules(model,name):
    rules=extract_rules(model,ACUTE_FEATURES)
    validate_rules(model,Xa,rules)
    rows=[{k:v for k,v in r.items() if k!='conditions'} for r in rules]
    frame=pd.DataFrame(rows);table(frame,name,limit=None)
    (OUT/(name+'.txt')).write_text('\n\n'.join(r['rule'] for r in rules),encoding='utf-8')
    for i,r in enumerate(rules,1): print(f'{i}. {r["rule"]} (support={r["samples"]}, purity={r["purity"]:.3f})')
    return rules
full_rules=show_rules(full_tree,'acute_full_if_then_rules')
# %% [markdown]
# ## 1(d) — Cost-complexity pruning and an interpretable minimal tree
# Cost-complexity pruning minimizes $R_\alpha(T)=R(T)+\alpha|\widetilde T|$,
# where R(T) is sample-weighted leaf impurity and |T~| is the number of leaves.
# Increasing alpha penalizes more leaves. The pruning path shows the successive
# subtrees of the full CART tree, but is not an exhaustive search over all trees.
#
# We evaluate a fixed alpha grid using stratified cross-validation, then use the
# **one-standard-error rule**: choose the largest alpha whose mean validation
# accuracy is at least best_mean - SE(best). Larger alpha favors a smaller tree.
# This defines 'minimal' as the simplest supported subtree among the searched
# candidates, rather than claiming a globally smallest tree with any structure.
# The final selected tree is refitted on all 120 examples, plotted, and converted
# into a fresh rule set. If pruning cannot reduce size within the accuracy
# tolerance, the result will explicitly report that the size stays unchanged.
# The fixed alpha grid avoids deriving tuning candidates from validation labels.
# %%
smallest_class=int(class_counts.min())
if smallest_class<2: raise ValueError('Too few cases in a powerset class for stratified CV.')
TREE_FOLDS=min(5,smallest_class)
tree_cv=list(StratifiedKFold(TREE_FOLDS,shuffle=True,random_state=SEED).split(Xa,ya))
TREE_ALPHAS=np.unique(np.r_[0,np.linspace(.0001,.4,161),1.0])
tree_search=GridSearchCV(DecisionTreeClassifier(random_state=SEED),
    {'ccp_alpha':TREE_ALPHAS},cv=tree_cv,scoring='accuracy',n_jobs=1,refit=False,error_score='raise')
checked_fit(tree_search,Xa,ya,'tree_pruning_cv')
tr=tree_search.cv_results_
tree_results=pd.DataFrame({'alpha':[float(p['ccp_alpha']) for p in tr['params']],
    'mean_accuracy':tr['mean_test_score'],
    'fold_sd':np.std(np.column_stack([tr[f'split{k}_test_score'] for k in range(TREE_FOLDS)]),axis=1,ddof=1)})
tree_results['standard_error']=tree_results.fold_sd/np.sqrt(TREE_FOLDS)
best_tree_row=tree_results.iloc[int(tree_results.mean_accuracy.argmax())]
accuracy_floor=float(best_tree_row.mean_accuracy-best_tree_row.standard_error)
eligible=tree_results[tree_results.mean_accuracy>=accuracy_floor-1e-12]
chosen_tree_alpha=float(eligible.alpha.max())
pruned_tree=DecisionTreeClassifier(random_state=SEED,ccp_alpha=chosen_tree_alpha).fit(Xa,ya)
for idx,row in tree_results.iterrows():
    candidate=DecisionTreeClassifier(random_state=SEED,ccp_alpha=float(row.alpha)).fit(Xa,ya)
    tree_results.loc[idx,'leaves']=candidate.get_n_leaves()
    tree_results.loc[idx,'depth']=candidate.get_depth()
table(tree_results,'acute_pruning_cv')
path=full_tree.cost_complexity_pruning_path(Xa,ya)
table(pd.DataFrame({'effective_alpha':path.ccp_alphas,'leaf_impurity':path.impurities}),'acute_cost_complexity_path')
print(f'Pruning uses {TREE_FOLDS} stratified folds; one-SE accuracy floor={accuracy_floor:.4f}.')
print(f'Selected alpha={chosen_tree_alpha:.6g}; leaves {full_tree.get_n_leaves()} -> {pruned_tree.get_n_leaves()}; '
      f'depth {full_tree.get_depth()} -> {pruned_tree.get_depth()}.')
print(f'Pruned joint training accuracy={pruned_tree.score(Xa,ya):.4f}')
if pruned_tree.get_n_leaves()==full_tree.get_n_leaves():
    print('The one-SE criterion retained the full leaf count; a smaller subtree did not meet its accuracy threshold.')
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].plot(tree_results.alpha,tree_results.mean_accuracy)
axes[0].axhline(accuracy_floor,ls='--',color='gray',label='Best mean minus one SE')
axes[0].axvline(chosen_tree_alpha,ls=':',color='red',label='Selected alpha')
axes[0].set(xlabel='ccp_alpha',ylabel='Mean validation accuracy',title='Cost-complexity pruning');axes[0].legend()
axes[1].step(tree_results.alpha,tree_results.leaves,where='post')
axes[1].set(xlabel='ccp_alpha',ylabel='Leaves',title='Tree complexity');fig.tight_layout()
figure('acute_pruning_selection')
plt.figure(figsize=(18,9))
plot_tree(pruned_tree,feature_names=ACUTE_FEATURES,
          class_names=[CLASS_NAMES[int(x)] for x in pruned_tree.classes_],
          filled=True,rounded=True,precision=3,fontsize=9)
plt.title('Acute Inflammations: selected interpretable subtree');figure('acute_pruned_tree')
pruned_rules=show_rules(pruned_tree,'acute_pruned_if_then_rules')
table(pd.DataFrame([{'model':name,'leaves':model.get_n_leaves(),'depth':model.get_depth(),
    'nodes':model.tree_.node_count,'training_pair_accuracy':model.score(Xa,ya)}
    for name,model in [('Full',full_tree),('Pruned',pruned_tree)]]),'acute_tree_comparison')
# %% [markdown]
# ### 1(d), continued — A compact pruning alternative and its accuracy cost
# The one-SE choice prioritizes accuracy. To also show what a genuinely smaller
# cost-complexity subtree looks like, impose an explicit interpretability budget
# of at most four leaves (four IF-THEN rules). Among grid candidates satisfying
# this budget, choose highest mean CV accuracy, then fewer leaves and lower
# alpha. This is a separate, documented size constraint, not a revised claim
# that aggressive pruning preserves the full tree's accuracy. The table and
# rules below make the tradeoff explicit.
# %%
compact_row=tree_results[tree_results.leaves<=4].sort_values(
    ['mean_accuracy','leaves','alpha'],ascending=[False,True,True]).iloc[0]
compact_tree=DecisionTreeClassifier(random_state=SEED,ccp_alpha=float(compact_row.alpha)).fit(Xa,ya)
print(f'Compact alternative: alpha={compact_row.alpha:.6g}, '
      f'leaves={compact_tree.get_n_leaves()}, CV accuracy={compact_row.mean_accuracy:.4f}, '
      f'training pair accuracy={compact_tree.score(Xa,ya):.4f}.')
compact_rules=show_rules(compact_tree,'acute_compact_if_then_rules')
plt.figure(figsize=(15,8))
plot_tree(compact_tree,feature_names=ACUTE_FEATURES,
    class_names=[CLASS_NAMES[int(x)] for x in compact_tree.classes_],
    filled=True,rounded=True,precision=3,fontsize=10)
plt.title('Compact cost-complexity subtree: at most four rules')
figure('acute_compact_tree')
# %% [markdown]
# ### Answer to 1(d): observed pruning tradeoff
#
# The five-fold best mean validation accuracy is **99.17%**, with SE **0.83
# percentage points**. The one-SE cutoff is **98.33%**. The largest acceptable
# searched penalty is **0.087578125**, and the full-data fitted subtree still has
# six leaves and 100% training accuracy. Thus accuracy-preserving cost-complexity
# pruning does not reduce the rule count in this run.
#
# Under the separate budget of at most four rules, **alpha=0.100075** yields four
# leaves and depth two, with **101/120=84.17%** training accuracy and **83.33%**
# mean CV accuracy. Its final high-temperature/lumbar-pain branch always predicts
# nephritis without bladder inflammation, merging cases that the full tree can
# separate. This explains the loss of accuracy. Use the six-rule model for the
# stronger accuracy criterion, or the four-rule model when the stated size limit
# matters more. Both actual sets of IF-THEN rules and plotted trees are included.
# %% [markdown]
# ## 2(a)–(b) — Communities and Crime split, nonpredictive fields, and imputation
# The target is **ViolentCrimesPerPop**. The source has 128 columns: five
# identifiers/administrative fields, 122 predictors, and one target. We drop
# state, county, community, communityname, and fold as requested by the data
# description. The source order is preserved: first 1495 rows train; remaining
# 499 test. We do not randomly reassign rows to these outer sets.
#
# '?' is treated as missing. Median imputation is fitted on training observations
# and applied to test observations. Within model-selection CV it is refitted on
# each fitting fold. This prevents validation or test values from influencing
# the imputation rule. If an entire fitting-fold column is missing,
# keep_empty_features=True retains it with the imputer's zero fallback.
# Missingness counts are displayed before imputation. No target is imputed.
#
# For the requested descriptive correlation and coefficient-of-variation plots,
# we use the training set after a training-fitted imputer. Model comparisons use
# all 122 predictors; the top-CV list does not alter the regression design.
# %%
NONPREDICTIVE=['state','county','community','communityname','fold']
TARGET='ViolentCrimesPerPop'
name_text=decode_text(paths['communities']['communities.names'])
column_names=[]
for line in name_text.splitlines():
    match=re.match(r'^\s*@attribute\s+(\S+)\s+',line,flags=re.I)
    if match: column_names.append(match.group(1).strip("'\""))
if len(column_names)!=128:
    raise ValueError(f'Expected 128 @attribute names in communities.names; found {len(column_names)}.')
crime=pd.read_csv(paths['communities']['communities.data'],header=None,names=column_names,na_values='?')
assert crime.shape==(1994,128),crime.shape
assert [x.lower() for x in column_names[:5]]==NONPREDICTIVE
assert column_names[-1]==TARGET
X=crime.drop(columns=NONPREDICTIVE+[TARGET]).apply(pd.to_numeric,errors='raise')
y=pd.to_numeric(crime[TARGET],errors='raise')
assert X.shape[1]==122 and y.notna().all()
X_train=X.iloc[:1495].copy(); X_test=X.iloc[1495:].copy()
y_train=y.iloc[:1495].copy(); y_test=y.iloc[1495:].copy()
assert len(X_train)==1495 and len(X_test)==499
print('Original shape:',crime.shape,'; train predictors:',X_train.shape,'; test predictors:',X_test.shape)
print('Dropped:',NONPREDICTIVE)
missing=pd.DataFrame({'feature':X.columns,'train_missing':X_train.isna().sum().values,
    'test_missing':X_test.isna().sum().values,'train_missing_fraction':X_train.isna().mean().values})
table(missing.sort_values('train_missing',ascending=False),'crime_missingness')
explore_imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(X_train)
X_explore=pd.DataFrame(explore_imputer.transform(X_train),columns=X.columns,index=X_train.index)
assert X_explore.shape==(1495,122) and np.isfinite(X_explore).all().all()
table(pd.DataFrame({'feature':X.columns,'training_median_or_empty_fallback':explore_imputer.statistics_}),
      'crime_training_imputation_values')
# %% [markdown]
# ## 2(c) — Correlation matrix
# Pearson correlation measures linear association between two predictors.
# Large absolute correlations identify redundancy and can make least-squares
# coefficient estimates unstable. A correlation near zero does not exclude a
# nonlinear relationship. The color scale is fixed from -1 to +1. Correlations
# use the imputed training set only, so the held-out test set remains reserved.
# The full numeric matrix is saved as CSV and the full labeled image as PNG.
# %%
corr=X_explore.corr()
corr.to_csv(OUT/'crime_predictor_correlation.csv',index=True)
fig,ax=plt.subplots(figsize=(24,23))
im=ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm',interpolation='nearest')
ax.set_xticks(range(len(X.columns)),X.columns,rotation=90,fontsize=5)
ax.set_yticks(range(len(X.columns)),X.columns,fontsize=5)
fig.colorbar(im,ax=ax,shrink=.65,label='Pearson correlation')
ax.set_title('Communities and Crime: 122 predictors, imputed training data')
fig.tight_layout();figure('crime_predictor_correlation')
# %% [markdown]
# ### Answer to 2(c): interpreting the observed correlations
#
# The training matrix contains substantial redundancy. In particular,
# **PolicPerPop** and **LemasSwFTPerPop** have correlation **1.000** after the stated
# imputation, and **population** versus **numbUrban** has correlation about **0.994**.
# Housing-value and immigration-timing summaries also have very high correlations.
# This makes regularization and component methods relevant: an individual OLS
# coefficient can be unstable even if predictions remain useful. Median imputation
# also creates repeated values, so the missingness table should be read alongside
# correlations involving heavily missing law-enforcement variables.
# %% [markdown]
# ## 2(d) — Coefficient of variation for every predictor
# For each predictor, $CV_j=s_j/m_j$, with sample standard deviation (ddof=1).
# CV is dimensionless and compares spread relative to the feature's mean. It is
# not a correlation with the response and does not measure predictive importance.
# Values near zero in the mean can make CV very large and unstable.
# A zero mean gives undefined CV; such variables are explicitly marked and
# excluded from the ranking rather than silently assigned infinite importance.
# CV also depends on the choice of zero: the ranking describes the supplied
# normalized columns, not unobserved original physical/economic scales.
# Here the calculation is on the original normalized predictor scale, after
# imputation, before standardization. StandardScaler would set means near zero
# and make this particular calculation meaningless.
# %%
means=X_explore.mean();stds=X_explore.std(ddof=1)
cv_values=stds.div(means.where(means.abs()>1e-12))
variation=pd.DataFrame({'feature':X.columns,'mean':means.values,
    'sample_standard_deviation':stds.values,'coefficient_of_variation':cv_values.values})
variation['CV_defined']=np.isfinite(variation.coefficient_of_variation)
variation=variation.sort_values(['coefficient_of_variation','feature'],ascending=[False,True],na_position='last')
table(variation,'crime_coefficient_of_variation',limit=None)
# %% [markdown]
# ## 2(e) — The 11 largest-CV features: scatter plots and box plots
# $\lfloor\sqrt{128}\rfloor=11$. We select the 11 largest finite CVs, and
# plot each against the response. All points are from training. A second plot
# shows the distributions of the eleven predictors as box plots. The explicit
# top-feature table, names, and CVs are printed before plotting.
#
# **Can significance be concluded from the scatter plots alone? No.** They can
# suggest a trend, nonlinear shape, outliers, or heteroscedasticity. They do not
# provide adjusted coefficient tests, uncertainty estimates, or control for the
# other predictors. High CV means high relative dispersion, not necessarily a
# strong relationship with violent crime. Even a visible relationship need not
# be independent of other predictors and does not establish causation.
# %%
N_PLOTS=int(np.floor(np.sqrt(128)))
top_cv=variation[variation.CV_defined].head(N_PLOTS)
top_features=top_cv.feature.tolist()
assert len(top_features)==11
table(top_cv,'crime_top11_cv',limit=None)
fig,axes=plt.subplots(3,4,figsize=(15,10))
for ax,name in zip(axes.flat,top_features):
    ax.scatter(X_explore[name],y_train,s=8,alpha=.3,color='#2878B5',edgecolors='none')
    ax.set(xlabel=name,ylabel=TARGET)
for ax in list(axes.flat)[len(top_features):]:ax.axis('off')
fig.suptitle('Top 11 coefficients of variation: predictor versus response (training)')
fig.tight_layout();figure('crime_top11_scatterplots')
plt.figure(figsize=(13,6))
plt.boxplot([X_explore[name].to_numpy() for name in top_features],tick_labels=top_features,
            showfliers=True,flierprops={'markersize':2,'alpha':.3})
plt.xticks(rotation=55,ha='right');plt.ylabel('Original normalized feature value')
plt.title('Distributions of the 11 highest-CV predictors (training)')
plt.tight_layout();figure('crime_top11_boxplots')
# %% [markdown]
# ### Answer to 2(d)–(e): selected high-CV variables and plot interpretation
#
# The eleven selected predictors, in descending coefficient-of-variation order,
# are **PctPolicAsian, NumStreet, NumInShelters, NumIlleg, NumImmig,
# LemasPctOfficDrugUn, NumUnderPov, LemasSwornFT, population, numbUrban, and
# HousVacant**. PctPolicAsian has the highest CV, approximately **5.5271**.
# The complete 122-feature CV table and the eleven plotted values are supplied.
#
# Several box plots have long upper tails and many observations near zero; this
# is consistent with large relative dispersion and small means. Some response
# scatter plots suggest increasing crime values with increasing counts, but
# substantial scatter remains. **These pictures do not establish statistical
# significance.** High CV is a variability ranking, and the plots cannot isolate
# a feature's conditional effect, give coefficient uncertainty, or control for
# correlated predictors. All plots use training observations only.
# %% [markdown]
# ## Common regression evaluation and cross-validation
# Primary test error is mean squared error:
# $$MSE=\frac1{499}\sum_{i\in test}(y_i-\hat y_i)^2.$$
# RMSE, MAE, and $R^2$ are supplied as supplementary measures. Targets are the
# dataset's normalized crime values, so MSE is in squared normalized units.
# No test-based model/parameter selection is performed. All preprocessing is
# fitted within each training fold. A fixed five-fold shuffled KFold is shared
# by the regression searches, so models face the same validation observations.
# The dataset's administrative 'fold' field is dropped; the outer split follows
# the assignment's required row order. 'CV' here means cross-validation; it is
# distinct from the coefficient of variation calculated above.
#
# The best grid point minimizes mean fold MSE. CV scores used to select parameters
# can be optimistic after tuning; the untouched test errors are the final
# comparison. A training-mean predictor is included as a reference baseline.
# %%
regression_folds=list(KFold(5,shuffle=True,random_state=SEED).split(X_train))
regression_results=[]
test_predictions=pd.DataFrame({'source_row_1based':X_test.index.to_numpy()+1,'observed':y_test.to_numpy()})

def evaluate_regression(name,model,setting='None',cv_mse=np.nan):
    pred=np.asarray(model.predict(X_test),dtype=float)
    assert pred.shape==(499,) and np.isfinite(pred).all()
    mse=mean_squared_error(y_test,pred)
    row={'model':name,'selected_setting':setting,'mean_CV_MSE':cv_mse,
         'test_MSE':mse,'test_RMSE':float(np.sqrt(mse)),
         'test_MAE':mean_absolute_error(y_test,pred),'test_R2':r2_score(y_test,pred)}
    regression_results.append(row);test_predictions[name]=pred
    table(pd.DataFrame([row]),name.replace(' ','_')+'_test_metrics',limit=None)
    print(f'{name}: test MSE={mse:.8f}, RMSE={np.sqrt(mse):.6f}, R2={row["test_R2"]:.4f}')
    return pred

baseline_pred=np.repeat(y_train.mean(),len(y_test))
print(f'Training-mean baseline test MSE={mean_squared_error(y_test,baseline_pred):.8f}')

def select_regression(name,pipe,param_grid):
    search=GridSearchCV(pipe,param_grid,cv=regression_folds,
        scoring='neg_mean_squared_error',n_jobs=1,refit=True,error_score='raise',return_train_score=False)
    start=time.time();checked_fit(search,X_train,y_train,name+'_CV')
    d=pd.DataFrame(search.cv_results_)
    rows=pd.DataFrame({'parameters':[str(p) for p in d.params],
        'mean_validation_MSE':-d.mean_test_score,'fold_MSE_sd':d.std_test_score,
        'rank':d.rank_test_score})
    for col in d.columns:
        if re.fullmatch(r'split\d+_test_score',col):rows[col.replace('_test_score','_MSE')]=-d[col]
    table(rows.sort_values(['rank','parameters']),name.replace(' ','_')+'_CV_grid')
    print(name,'selected',search.best_params_,'in',round(time.time()-start,1),'seconds')
    return search
# %% [markdown]
# ## 2(f) — Ordinary least squares
# The model minimizes the sum of squared training residuals and includes an
# intercept. Predictors are median-imputed, with no further standardization.
# LinearRegression uses a least-squares solver; it does not explicitly invert
# X'X, which would be fragile with correlated or linearly dependent predictors.
# The test MSE below is computed from the 499 untouched test observations.
# %%
ols=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
              ('model',LinearRegression())])
checked_fit(ols,X_train,y_train,'OLS')
ols_pred=evaluate_regression('OLS',ols)
print('OLS design rank (without intercept):',ols.named_steps['model'].rank_,'of',X_train.shape[1])
# %% [markdown]
# ### Answer to 2(f): least-squares test error
#
# OLS has **test MSE 0.01796622**, RMSE **0.134038**, and test R-squared **0.6222**.
# The training-mean baseline has test MSE **0.04766573**, so the fitted predictor
# improves substantially on that reference. The 122-column OLS design has rank
# 121, consistent with exact redundancy. A least-squares solution still exists;
# individual coefficients are not uniquely identified by an exactly redundant
# design, which is another reason to distinguish coefficient interpretation from
# predictive performance.
# %% [markdown]
# ## 2(g) — Ridge regression
# Ridge minimizes SSE + lambda * sum(beta_j^2); the intercept is not penalized.
# We standardize predictors inside each fold so their scales do not determine
# relative penalty strength. This remains useful even though the original
# features lie in [0,1], because their standard deviations can differ greatly.
# The search covers 51 logarithmically spaced lambda values from 1e-5 to 1e5.
# In sklearn's Ridge, alpha is this lambda. After tuning on training CV, the
# full pipeline is refitted on all 1495 training observations and tested once.
# %%
RIDGE_GRID=np.logspace(-5,5,51)
ridge_pipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
    ('scale',StandardScaler()),('model',Ridge())])
ridge_search=select_regression('Ridge',ridge_pipe,{'model__alpha':RIDGE_GRID})
ridge=ridge_search.best_estimator_
ridge_pred=evaluate_regression('Ridge',ridge,f'lambda={ridge_search.best_params_["model__alpha"]:.8g}',
                               -ridge_search.best_score_)
# %% [markdown]
# ### Answer to 2(g): selected ridge model
#
# Five-fold CV chooses **lambda=63.09573445** on the stated Ridge objective scale.
# Mean CV MSE is approximately **0.019141**. Its test MSE is **0.01746030**, RMSE
# **0.132137**, and test R-squared **0.6328**. This test MSE is lower than OLS in
# this split, consistent with a benefit from shrinking correlated coefficients.
# %% [markdown]
# ## 2(h) — LASSO with original normalized and standardized features
# LASSO minimizes $(1/(2n))||y-X\beta||_2^2+\lambda||\beta||_1$. Thus
# sklearn Lasso's alpha is lambda under this objective; it is not numerically
# interchangeable with Ridge's alpha, whose loss has a different normalization.
# The intercept is unpenalized. L1 can set coefficients exactly to zero.
#
# First, use the original [0,1] normalized features after imputation. Second,
# add StandardScaler fitted inside each fold. Both search the same fixed grid,
# from 1e-5 to 1 plus the intermediate points. All 122 predictors are available
# to each fit. A coefficient larger than 1e-8 in absolute value counts as selected;
# both full coefficient tables and complete selected-variable lists are saved.
# Standardized coefficients are also converted back to original input units.
#
# A common [0,1] range does not give equal variance. Standardization changes the
# effective penalty on the original coefficients and may change the selected
# variables, optimal lambda, and test error. The code prints the observed
# comparison instead of assuming standardization must improve performance.
# %%
LASSO_GRID=np.logspace(-5,0,51)
lasso_models={};lasso_searches={};selected_variables={}
for standardized in [False,True]:
    name='LASSO standardized' if standardized else 'LASSO original'
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if standardized:steps.append(('scale',StandardScaler()))
    steps.append(('model',Lasso(max_iter=100000,tol=1e-6,selection='cyclic')))
    search=select_regression(name,Pipeline(steps),{'model__alpha':LASSO_GRID})
    model=search.best_estimator_;lasso_models[name]=model;lasso_searches[name]=search
    evaluate_regression(name,model,f'lambda={search.best_params_["model__alpha"]:.8g}',-search.best_score_)
    coef=model.named_steps['model'].coef_
    if standardized:
        scaler=model.named_steps['scale'];original_coef=coef/scaler.scale_
        original_intercept=model.named_steps['model'].intercept_-np.dot(original_coef,scaler.mean_)
    else:original_coef=coef.copy();original_intercept=model.named_steps['model'].intercept_
    coefficients=pd.DataFrame({'feature':X_train.columns,'fitted_scale_coefficient':coef,
        'original_units_coefficient':original_coef,'selected':np.abs(coef)>1e-8})
    table(coefficients,name.replace(' ','_')+'_all_coefficients',limit=None)
    selected=coefficients[coefficients.selected].copy()
    table(selected,name.replace(' ','_')+'_selected_variables',limit=None)
    selected_variables[name]=selected.feature.tolist()
    print(name,'selected',len(selected),'of 122 predictors:')
    print(', '.join(selected.feature))
    print('Original-units intercept:',original_intercept)
lasso_comparison=pd.DataFrame([r for r in regression_results if r['model'].startswith('LASSO')])
table(lasso_comparison,'lasso_scale_comparison',limit=None)
better=lasso_comparison.sort_values('test_MSE').iloc[0]
print(f'Observed lower test MSE: {better["model"]} ({better["test_MSE"]:.8f}).')
print('Variables selected by both:',sorted(set.intersection(*(set(v) for v in selected_variables.values()))))
# %% [markdown]
# ### Answer to 2(h): LASSO results and standardization comparison
#
# On the original normalized predictors, CV selects **lambda=0.0001** and **75
# variables**. Test MSE is **0.01765778**, RMSE **0.132883**. On standardized
# predictors, CV selects **lambda=0.0006309573445** and **76 variables**. Test MSE
# is **0.01772175**, RMSE **0.133123**. The full selected-variable lists, with
# coefficients, are printed above and saved as separate CSVs; neither list is
# truncated in the notebook.
#
# The original-scale model has a slightly lower observed test error here. The
# difference is small, and standardization is not guaranteed to improve every
# held-out result. Standardization changes the effective featurewise penalty,
# so its different lambda and selected set are expected. Both versions remain
# sparse relative to the 122 available variables.
# %% [markdown]
# ## 2(i) — Principal component regression
# PCR regresses the response on the first M principal components of standardized
# predictors. PCA is unsupervised: it finds directions of predictor variance,
# which need not be the most predictive directions for the response. Therefore
# M is selected by validation MSE, rather than only by explained variance.
#
# Every fitting fold learns its own imputer, scaler, and full PCA. For each
# M=1,...,122, least squares is fitted to the first M training scores and evaluated
# on the matching validation scores. Reusing one full PCA per fitting fold is
# computationally efficient and gives the same prefix subspaces as separate
# full-SVD PCA fits for each M. No PCA is fitted on all rows before CV.
# Ties within floating-point tolerance prefer smaller M. The final selected
# pipeline is refitted on all training rows, then evaluated on test.
# %%
pcr_fold_rows=[]
for fold,(a,b) in enumerate(regression_folds,1):
    preprocess=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
        ('scale',StandardScaler()),('pca',PCA(svd_solver='full'))])
    A=preprocess.fit_transform(X_train.iloc[a]);B=preprocess.transform(X_train.iloc[b])
    for M in range(1,X_train.shape[1]+1):
        reg=LinearRegression().fit(A[:,:M],y_train.iloc[a])
        val=reg.predict(B[:,:M])
        pcr_fold_rows.append({'M':M,'fold':fold,'validation_MSE':mean_squared_error(y_train.iloc[b],val)})
    print('PCR fold',fold,'complete',flush=True)
pcr_folds=pd.DataFrame(pcr_fold_rows);pcr_folds.to_csv(OUT/'PCR_CV_folds.csv',index=False)
pcr_grid=pcr_folds.groupby('M',as_index=False).agg(mean_validation_MSE=('validation_MSE','mean'),
    fold_MSE_sd=('validation_MSE','std'))
best_pcr_mse=float(pcr_grid.mean_validation_MSE.min())
best_M=int(pcr_grid.loc[pcr_grid.mean_validation_MSE<=best_pcr_mse+1e-12,'M'].min())
table(pcr_grid.sort_values('mean_validation_MSE'),'PCR_CV_grid')
pcr=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
    ('scale',StandardScaler()),('pca',PCA(n_components=best_M,svd_solver='full')),
    ('model',LinearRegression())])
checked_fit(pcr,X_train,y_train,'PCR_final')
pcr_pred=evaluate_regression('PCR',pcr,f'M={best_M}',
    float(pcr_grid.loc[pcr_grid.M==best_M,'mean_validation_MSE'].iloc[0]))
print('Selected M:',best_M,'of 122; cumulative explained predictor variance:',
      pcr.named_steps['pca'].explained_variance_ratio_.sum())
plt.figure(figsize=(9,4));plt.plot(pcr_grid.M,pcr_grid.mean_validation_MSE,color='#2878B5')
plt.axvline(best_M,color='#D55E00',ls='--',label=f'Selected M={best_M}')
plt.xlabel('Number of principal components M');plt.ylabel('Mean validation MSE')
plt.title('PCR: five-fold training cross-validation');plt.legend();plt.tight_layout();figure('PCR_component_selection')
# %% [markdown]
# ### Answer to 2(i): selected PCR model
#
# Training CV selects **M=93** principal components, with mean validation MSE
# approximately **0.019449**. These retain approximately **99.796%** of predictor
# variance. Test MSE is **0.01826425**, RMSE **0.135145**, and R-squared **0.6159**.
# PCR does not outperform the original-feature OLS fit on this test set. PCA's
# high-variance directions do not necessarily preserve every direction relevant
# to predicting the response; that is why M was chosen using response prediction
# error rather than an arbitrary variance-retention threshold.
# %% [markdown]
# ## 2(j) — L1-penalized gradient boosting regression trees
# The PDF says XGBoost **can** be used. Here we use a self-contained histogram
# implementation because installing that optional library is blocked in the
# execution environment. The algorithm is provided in **l1_boosting.py** and
# uses the regularized squared-error tree objective described below. Results
# are labeled L1 boosting; they are not claimed to come from the XGBoost package.
#
# As with standard XGBoost gbtree, the L1 penalty acts on scalar leaf scores.
# This is distinct from fitting a separate multivariable LASSO at every node.
# At iteration t, let r_i=y_i-F_(t-1)(x_i). For a proposed tree with leaf scores
# w_j, optimize
# $$\frac12\sum_i(r_i-w_{q(x_i)})^2+\alpha\sum_j|w_j|
#       +\frac{\lambda_2}{2}\sum_jw_j^2.$$
# If leaf j contains n_j observations with residual sum G_j, its exact score is
# $$w_j=\frac{\operatorname{sign}(G_j)\max(|G_j|-\alpha,0)}{n_j+\lambda_2}.$$
# For a split into L and R, the regularized gain is
# $$\frac12\left[\frac{S(G_L,\alpha)^2}{n_L+\lambda_2}
# +\frac{S(G_R,\alpha)^2}{n_R+\lambda_2}
# -\frac{S(G_P,\alpha)^2}{n_P+\lambda_2}\right],$$
# where S is the soft-threshold operation in the numerator above. The best
# positive-gain split is grown recursively to the depth limit. Leaf scores are
# added with learning-rate shrinkage. Larger alpha encourages smaller or zero
# leaf updates and changes split gains as well.
#
# Five-fold CV selects alpha from {0,1e-4,1e-3,1e-2,0.1,1,10,100}. All other
# settings are fixed before CV: 200 trees, max depth 3, learning rate 0.05,
# at most 32 quantile bins per predictor, at least 10 rows per leaf, and
# lambda_2=0 to isolate L1 regularization. Quantile cut points and median
# imputation are learned from each fitting fold only. No test data is supplied
# for early stopping. Numerical ties favor larger alpha. The final model is
# refitted on all training observations and its test error is reported.
#
# Implementation checks include the exact analytic leaf solution, the all-zero
# update limit for sufficiently large alpha, sklearn pipeline/CV compatibility,
# and agreement with sklearn's unpenalized gradient boosting when alpha=0 and
# every possible split threshold is available. This is a regularized boosting
# tree model rather than an unpenalized estimator relabeled as L1.
# %%
from l1_boosting import L1HistogramBoostingRegressor
BOOST_ALPHAS=np.array([0.,1e-4,1e-3,1e-2,.1,1.,10.,100.])
boost_pipe=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
    ('model',L1HistogramBoostingRegressor(n_estimators=200,max_depth=3,
        learning_rate=.05,max_bins=32,min_samples_leaf=10,reg_lambda=0))])
boost_search=select_regression('L1 boosting',boost_pipe,{'model__reg_alpha':BOOST_ALPHAS})
boost_grid=pd.DataFrame(boost_search.cv_results_)
boost_mse=-boost_grid.mean_test_score.to_numpy()
best_boost_mse=float(boost_mse.min())
boost_alpha=max(float(p['model__reg_alpha']) for p,mse in zip(boost_grid.params,boost_mse)
                if mse<=best_boost_mse+1e-12)
boost=clone(boost_pipe).set_params(model__reg_alpha=boost_alpha)
checked_fit(boost,X_train,y_train,'L1_boosting_final')
boost_pred=evaluate_regression('L1 boosting',boost,f'alpha={boost_alpha:g}',
    float(boost_mse[[float(p['model__reg_alpha']) for p in boost_grid.params].index(boost_alpha)]))
boost.named_steps['model'].save_model(OUT/'l1_boosting_model.json')
print('Selected L1 alpha:',boost_alpha)
if boost_alpha==0:
    print('CV selected alpha=0; there was no CV gain from adding an L1 leaf-score penalty on the searched grid.')
importance=pd.DataFrame({'feature':X_train.columns,
    'normalized_split_gain':boost.named_steps['model'].feature_importances_}).sort_values('normalized_split_gain',ascending=False)
table(importance,'L1_boosting_split_importance')
plt.figure(figsize=(8,4));plt.plot(boost.named_steps['model'].training_mse_)
plt.xlabel('Boosting iteration');plt.ylabel('Training MSE')
plt.title(f'L1 boosting fit; selected alpha={boost_alpha:g}')
plt.tight_layout();figure('L1_boosting_training_curve')
# %% [markdown]
# ### Answer to 2(j): selected boosting penalty and test error
#
# The five-fold search chooses **alpha=0**, with mean CV MSE approximately
# **0.019214**. Final test MSE is **0.01654130**, RMSE **0.128613**, and R-squared
# **0.6522**. This is the lowest observed test MSE among the six fitted methods.
# The best mean CV MSE across the model families belongs to original-scale LASSO;
# validation and test rankings need not be identical on finite samples.
#
# Alpha zero is the unpenalized endpoint of the L1-capable model family. Positive
# penalties were actually fitted and evaluated; they did not improve mean CV MSE
# on the specified grid. The result therefore supports this boosting architecture
# on the test split, **not a claim that L1 regularization itself caused the
# improvement**. The complete positive-alpha scores are saved alongside the zero
# baseline. No test-driven tuning follows this comparison.
# %% [markdown]
# ## Final comparison, observed interpretations, and checks
# The comparison reports every required regression test error on the same 499
# held-out rows. Selected parameter values, LASSO variable lists, complete CV
# grids, and predictions are saved separately so the results are auditable.
# The printed ordering describes this particular test set; it is not a proof of
# superiority on every future dataset. Highly correlated predictors may yield
# different sparse variable sets with similar predictive accuracy.
#
# The last checks confirm data shapes, disjoint row ranges, complete output
# coverage, finite predictions, and rule/tree agreement. Any fitting warnings
# are retained instead of silently suppressing numerical problems.
# %%
comparison=pd.DataFrame(regression_results)
assert set(comparison.model)=={'OLS','Ridge','LASSO original','LASSO standardized','PCR','L1 boosting'}
table(comparison.sort_values('test_MSE'),'regression_model_comparison',limit=None)
table(test_predictions,'crime_test_predictions',limit=10)
print('Lowest observed test MSE:',comparison.sort_values('test_MSE').iloc[0]['model'])
print('For interpretation, compare this ranking with the training-CV MSE column; no retuning follows this test comparison.')
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,row in zip(axes.flat,regression_results):
    pred=test_predictions[row['model']]
    ax.scatter(y_test,pred,s=9,alpha=.35,color='#2878B5',edgecolors='none')
    lo=float(min(y_test.min(),pred.min()));hi=float(max(y_test.max(),pred.max()))
    ax.plot([lo,hi],[lo,hi],'--',color='gray')
    ax.set(xlabel='Observed target',ylabel='Prediction',title=f'{row["model"]}; MSE={row["test_MSE"]:.5f}')
fig.tight_layout();figure('regression_test_predictions')
fig,ax=plt.subplots(figsize=(10,5))
ordered=comparison.sort_values('test_MSE')
ax.barh(ordered.model,ordered.test_MSE,color='#2878B5');ax.invert_yaxis()
ax.axvline(mean_squared_error(y_test,baseline_pred),color='#D55E00',ls='--',label='Training-mean baseline')
ax.set(xlabel='Test mean squared error',title='All models: same prescribed 499-row test set')
ax.legend();fig.tight_layout();figure('regression_test_mse_comparison')
warning_frame=pd.DataFrame(fit_warnings,columns=['context','warning'])
table(warning_frame,'fitting_warnings',limit=20)
assert X_train.index.intersection(X_test.index).empty
assert all(set(a).isdisjoint(set(b)) for a,b in regression_folds)
assert len(test_predictions)==499 and test_predictions.notna().all().all()
validate_rules(full_tree,Xa,full_rules);validate_rules(pruned_tree,Xa,pruned_rules)
print('Verified required shapes, disjoint splits, finite predictions, and exact IF-THEN/tree agreement.')
print('Fitting warnings:',len(fit_warnings),'; check results/fitting_warnings.csv if nonzero.')
# %% [markdown]
# ## References and coverage
# - Assignment: Homework5.pdf, DSCI 552, Mohammad Reza Rajati, questions 1(a)–(d)
#   and 2(a)–(j). Every required question has a named section above.
# - Czerniak, J. (2003), *Acute Inflammations*, UCI Machine Learning Repository,
#   https://doi.org/10.24432/C5V59S . Data and diagnosis labels:
#   https://archive.ics.uci.edu/dataset/184/acute+inflammations
# - Redmond, M. (2009), *Communities and Crime*, UCI Machine Learning Repository:
#   https://archive.ics.uci.edu/dataset/183/communities+and+crime . The original
#   .names file defines predictors, nonpredictive fields, and normalized units.
# - scikit-learn: cost-complexity pruning:
#   https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
# - scikit-learn: imputation and leakage prevention:
#   https://scikit-learn.org/stable/modules/impute.html
#   https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
# - scikit-learn: LASSO objective and coefficient scaling:
#   https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# - Regularized boosting objective and L1 leaf-score penalty (reference for
#   the independent implementation): XGBoost tree parameters and reg_alpha:
#   https://xgboost.readthedocs.io/en/stable/parameter.html
#
# The source data remain unchanged. Encodings, missing-value handling, target
# definitions, CV schemes, finite parameter grids, and solver choices are stated
# explicitly. Grid optima are not claimed to be global optima over all settings.
