# DSCI 552 Homework 5 — Decision trees and regression

**Yash Desai**

The main notebook answers every required question in Homework5.pdf, with code,
explanations, saved model outputs, and plots. Both original UCI datasets are
included. Read `VERIFICATION.txt` for the completed execution checks.

## Run in VS Code

Use Python 3.11–3.13. Open this complete folder with **File → Open Folder**.
In the VS Code terminal, install dependencies into the selected Python environment:

```bash
python -m pip install -r requirements.txt
```

Open `Yash_Desai_DSCI552_HW5_Explained.ipynb`, select that environment, and choose
**Restart → Run All**. Save the notebook after running it. The full searches may
take several minutes, depending on the computer. The equivalent source runs with:

```bash
python run_hw5.py
```

Run from this project root, where `data/`, `download_data.py`, and
`l1_boosting.py` are located. No ZIP extraction or external download is needed
when using the complete package. Do not open a separate notebook copy from a
different folder: its working directory must contain this project.

## Files

- `Yash_Desai_DSCI552_HW5_Explained.ipynb`: primary submission with explanations and saved outputs.
- `run_hw5.py`: equivalent analysis source with notebook-style section markers.
- `l1_boosting.py`: documented, sklearn-compatible implementation for 2(j).
- `download_data.py`: optional official-data download or local-ZIP helper.
- `data/acute/`: original `diagnosis.data` and `diagnosis.names`.
- `data/communities/`: original `communities.data` and `communities.names`.
- `results/`: complete numeric tables, CV scores, selected variables, IF-THEN rules,
  model parameters, test predictions, and figure PNGs.
- `requirements.txt`: pinned numerical packages plus notebook tools.
- `VERIFICATION.txt`: checks performed on the completed project.

## Required question coverage

| Question | Included analysis and output |
|---|---|
| 1(a) | Six features and two diagnosis labels; UTF-16 and decimal-comma parsing |
| 1(b) | Label-powerset tree on all 120 rows, plotted; training accuracy |
| 1(c) | Every full-tree root-to-leaf IF-THEN rule, decoded labels, support, purity |
| 1(d) | Cost-complexity path, stratified CV, one-SE choice, selected tree/rules, and a genuinely smaller four-rule alternative showing the accuracy tradeoff |
| 2(a) | First 1495 rows train; remaining 499 test, preserving source order |
| 2(b) | Five nonpredictive fields removed; fold-fitted median imputation |
| 2(c) | Full 122-predictor correlation matrix and numeric CSV |
| 2(d) | Sample standard deviation divided by mean for every predictor |
| 2(e) | Top 11 CV predictors, scatter and box plots; statistical-significance explanation |
| 2(f) | OLS fit and test MSE, RMSE, MAE, R-squared |
| 2(g) | Ridge lambda chosen by five-fold CV and test metrics |
| 2(h) | Original and standardized LASSO; CV lambda, full selected-variable lists, test comparison |
| 2(i) | PCR with M=1,...,122 selected within five-fold CV; final test metrics |
| 2(j) | L1-regularized histogram boosting; alpha selected by five-fold CV; final test metrics |

## Important implementation choices

The acute task has two labels. Label powerset represents their joint pair as
one of four diagnosis combinations, allowing one interpretable tree to explain
both labels. Whole-dataset scores are training scores, not clinical validation.

Pruning chooses the largest alpha within one standard error of best CV accuracy.
The same full leaf count can remain optimal under that criterion. A separate
four-leaf cost-complexity subtree is also shown so the interpretability/accuracy
tradeoff is explicit. Neither is claimed to be globally optimal among every
possible decision tree structure.

All regression fits have all 122 predictors available. The eleven highest-CV
features are used for the requested exploratory plots only. Every regression
search uses the same five training folds; imputation, scaling, PCA, and boosting
histogram thresholds are fitted within the appropriate fold. The 499 test rows
are never used to select model parameters or for early stopping.

For **2(j)** the PDF permits XGBoost rather than requiring that package. Installing
it is blocked in the execution environment, so this solution includes an independent
implementation of L1-regularized histogram gradient boosting in `l1_boosting.py`.
It uses regularized split gains and exact soft-thresholded leaf scores. The code,
objective, grids, and fixed settings are explained in the notebook. Results are
labeled **L1 boosting**, not XGBoost library results. This is the same type of
L1 leaf-score regularization used by standard XGBoost gbtree; neither fits a
separate multivariable LASSO regression at every internal node.

## Original data and attribution

The raw dataset files are preserved unchanged. They came from the user-provided
original UCI ZIPs. To restore them if removed, run `python download_data.py` on a
computer with Internet access, or use `--archives` with both original ZIP paths.

- Czerniak, J. (2003). *Acute Inflammations*. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5V59S
- Redmond, M. (2009). *Communities and Crime*. UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/183/communities+and+crime

Further references are in the notebook. The `.names` files describe the fields,
normalization, missingness, and intended target definitions.

## GitHub submission

Create a new public repository for HW5 and push the extracted project contents,
including the notebook, both code modules, data, results, and requirements.
Check that GitHub displays the notebook outputs and submit the repository URL
through the course's designated submission channel.
