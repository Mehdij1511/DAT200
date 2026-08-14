# DAT200 — Applied Machine Learning

This repository holds my DAT200 coursework (CA1–CA5). Each CA is a self-contained assignment demonstrating stages of an end-to-end applied machine learning workflow: exploratory data analysis, preprocessing, feature engineering, classical and advanced modeling, tuning, and a final applied case study.

## Table of contents

- Repository contents
- Per-assignment summaries (technical + code highlights)
- Tools & technologies
- How to run / reproduce (next steps)
- About me / abilities demonstrated
- Next steps I can take for you
- Repository rename instructions

## Repository contents

- CA1.pdf — Coursework Assignment 1 (EDA & preprocessing)
- CA2.pdf — Coursework Assignment 2 (feature engineering & baseline models)
- CA3.pdf — Coursework Assignment 3 (model tuning & evaluation)
- CA4.pdf — Coursework Assignment 4 (pipelines & ensemble models)
- CA5.pdf — Coursework Assignment 5 (end-to-end applied case study)

> Note: currently the repo contains the assignment reports as PDFs. If you want runnable code, I can extract and add notebooks or scripts.

---

## Per-assignment summaries (technical + code highlights)

Each section below summarizes the technical goals of the assignment, the key steps you can find in the report, the skills demonstrated, and a short, ready-to-run code snippet that matches the approaches used in the assignment. Use these snippets as starting points for notebooks or reproducible scripts.

### CA1 — Data exploration & preprocessing

Summary
- Performed exploratory data analysis (univariate and multivariate), missing-value analysis and simple imputation, date parsing and type fixes, outlier detection, and initial visualizations used to inform feature engineering.

Skills demonstrated
- pandas data cleaning, missing-data strategies, visualization with matplotlib/seaborn, basic domain-driven transformations.

Code highlights
- Load dataset, inspect types and distributions, impute missing values, and plot a target distribution.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/dataset.csv")
print(df.info())
print(df.describe())

# missing values
print(df.isna().sum().sort_values(ascending=False).head(10))

# median imputation example
if "age" in df.columns:
    df["age"] = df["age"].fillna(df["age"].median())

sns.histplot(df["target"], kde=True)
plt.title("Target distribution")
plt.show()
```

Artifacts to look for in the PDF
- Distribution plots, tables of missing values, decisions for data cleaning, and a short section describing how EDA influenced feature choices.

---

### CA2 — Feature engineering & classical models

Summary
- Engineered features (interaction terms, aggregations, encodings), evaluated correlation and predictive power, trained baseline models (linear / logistic regression, decision trees), and used cross-validation for baseline performance.

Skills demonstrated
- Feature engineering, categorical encoding, scikit-learn pipelines, cross-validation, baseline model interpretability.

Code highlights
- ColumnTransformer to preprocess numeric and categorical features, then a logistic regression baseline evaluated with cross-validation.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

num_cols = ["age", "income", "transactions_count"]
cat_cols = ["region", "category"]

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

preproc = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline([
    ("preproc", preproc),
    ("model", LogisticRegression(max_iter=1000))
])

scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print("CV AUC:", scores.mean(), "+/-", scores.std())
```

Artifacts to look for in the PDF
- Feature importance tables, engineered-variable descriptions, baseline model coefficients, and CV performance summaries.

---

### CA3 — Model tuning & evaluation

Summary
- Carried out hyperparameter tuning (GridSearchCV/RandomizedSearchCV), diagnostic plots (learning curves, validation curves), and robust evaluation on hold-out data using metrics like ROC AUC, precision/recall, and confusion matrices.

Skills demonstrated
- Hyperparameter search, nested validation patterns, detailed model diagnostics and metric-driven decisions about bias/variance.

Code highlights
- Randomized search over Gradient Boosting hyperparameters, and test-set evaluation.

```python
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7]
}

base = GradientBoostingClassifier(random_state=42)
rs = RandomizedSearchCV(base, param_dist, n_iter=12, cv=5, scoring="roc_auc", n_jobs=-1)
rs.fit(X_train, y_train)

print("Best params:", rs.best_params_)
print(classification_report(y_test, rs.predict(X_test)))
print("Test AUC:", roc_auc_score(y_test, rs.predict_proba(X_test)[:, 1]))
```

Artifacts to look for in the PDF
- Hyperparameter tables, learning curves, final model selection rationale, and test-set performance with confidence intervals where applicable.

---

### CA4 — Advanced models & pipelines

Summary
- Implemented robust scikit-learn Pipelines combining preprocessing and advanced estimators (Random Forest, XGBoost/LightGBM), investigated feature importances, and saved reproducible artifacts (trained model files).

Skills demonstrated
- Pipelines, ensemble learning, model persistence (joblib), and interpretability (feature importance, SHAP if used).

Code highlights
- Fit a pipeline with a RandomForest and save it for deployment or later evaluation.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

pipeline = Pipeline([
    ("preproc", preproc),  # reuse preproc from CA2
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)
dump(pipeline, "models/ca4_pipeline.joblib")

# feature importances
importances = pipeline.named_steps["clf"].feature_importances_
print("Top features indices:", importances.argsort()[-10:][::-1])
```

Artifacts to look for in the PDF
- Pipeline diagrams or descriptions, ensemble performance versus baselines, saved-model hashes or file names, and discussions of interpretability.

---

### CA5 — Project / applied case study (end-to-end)

Summary
- Integrated everything into a final applied project: problem definition, EDA, preprocessing, feature engineering, robust model selection, final evaluation and business/policy recommendations. Emphasis on reproducibility and communication of results.

Skills demonstrated
- End-to-end ML workflow, reproducible experiments, model interpretation, visualization, and communicating results to stakeholders.

Notebook outline (skeleton)

```python
# 1. Load & quick EDA
# 2. Preprocessing (impute, encode, scale)
# 3. Feature engineering (domain features)
# 4. Model selection (CV + hyperparam tuning)
# 5. Final fit on full train, evaluate on holdout
# 6. Save model and create viz artifacts (ROC, confusion matrix)
# 7. Conclusions & suggested next steps
```

Artifacts to look for in the PDF
- Final recommendations, clear visual storytelling (figures with captions), and an appendix listing reproducibility details (seed, packages, data sources).

---

## Tools & technologies

- Languages: Python (analysis & modeling), Markdown
- Key libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib (optionally xgboost, lightgbm, shap)
- Development: Git, GitHub, Jupyter Notebooks

## How to run / reproduce (what I can add)

Right now the repo contains reports only. If you want runnable artifacts, I suggest the following additions I can create:

- notebooks/CA1.ipynb ... notebooks/CA5.ipynb — extracted & cleaned code from the PDFs ready to run
- requirements.txt — pin the main Python dependencies
- scripts/run_all.sh or Makefile — build virtualenv, install deps, run notebooks to HTML
- models/ — directory to store trained pipelines and sample serialized models

Example quick-start (after I add notebooks and requirements):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/CA1.ipynb
```

I can also add a small GitHub Actions workflow that runs nbconvert on each notebook to ensure they execute without errors on push.

## About me / Abilities demonstrated

- Strong data cleaning and preprocessing skills; able to prepare messy data for modeling.
- Practical experience with classical and ensemble ML models and with model-selection best practices.
- Competent at reproducible workflows (pipelines, saved models) and communicating results through plots and written summaries.

## Next steps I can take for you (choose one)

1. Extract runnable Python code from each PDF into Jupyter notebooks (fast). I will:
   - Parse the PDFs for code blocks, clean them, place them in notebooks/CAx.ipynb, and add a requirements.txt.
2. Attempt to run and fix the extracted code (requires access to the original data files or simulated data). I will:
   - Run notebooks in CI, fix obvious import/path issues, and create small sample datasets if needed.
3. Add CI to run the notebooks and produce HTML/PDF artifacts on each push.
4. Create a script or workflow to rename the repository using the GitHub API (requires a token you run or provide).

Tell me which of the above you want me to do and I will implement it.

---

## Repository rename instructions

I updated the README to use the name "DAT200 — Applied Machine Learning". To rename the GitHub repository itself to `DAT200 Applied Machine Learning`:

1. Go to the repository on github.com → Settings → General → Repository name and change it.

OR

2. Use the REST API (requires a token with repo scope):

```bash
curl -X PATCH -H "Authorization: token YOUR_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/Mehdij1511/DAT200 \
  -d '{"name":"DAT200 Applied Machine Learning"}'
```

If you want, I can add a small script or GitHub Action that attempts the rename when provided with a token, but note that actions that change repository settings require elevated permissions and must be run by you or an account with admin access.
