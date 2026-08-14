# DAT200 — Applied Machine Learning

This repository collects my coursework for DAT200 (Applied Machine Learning). It contains the course assignments (CA1–CA5) submitted as PDFs and demonstrates the data science and machine learning skills I used to complete them.

## Repository contents

- CA1.pdf — Coursework Assignment 1
- CA2.pdf — Coursework Assignment 2
- CA3.pdf — Coursework Assignment 3
- CA4.pdf — Coursework Assignment 4
- CA5.pdf — Coursework Assignment 5

(Each PDF is the submitted assignment report. Open them to see full details, figures, and code references.)

## Overview of assignments

Below is a short, professional summary you can use on your CV or portfolio. If you want more specific wording for any assignment, tell me which CA (1–5) and I can extract highlights from the PDF and expand the summary.

- CA1 — Data exploration & preprocessing
  - Summary: Conducted exploratory data analysis (EDA), cleaned and preprocessed raw data, handled missing values and outliers, and created visualizations to summarize distributions and relationships.
  - Skills showcased: data cleaning (pandas), EDA, data visualization (matplotlib / seaborn), descriptive statistics.

- CA2 — Feature engineering & classical models
  - Summary: Engineered features to improve predictive power, selected features using correlation and basic selection techniques, and trained classical models (linear regression, logistic regression, decision trees) with evaluation using cross-validation.
  - Skills showcased: feature engineering, scikit-learn modeling, cross-validation, model selection, evaluation metrics.

- CA3 — Model tuning & evaluation
  - Summary: Performed hyperparameter tuning (grid/random search), used validation curves and learning curves to diagnose overfitting/underfitting, and compared model performance on hold-out data.
  - Skills showcased: hyperparameter search (GridSearchCV / RandomizedSearchCV), model diagnostics, performance metrics (ROC, AUC, precision/recall), reproducible experiments.

- CA4 — Advanced models & pipelines
  - Summary: Implemented more advanced models (ensemble methods such as Random Forests or Gradient Boosting), built end-to-end pipelines for preprocessing and modeling, and analyzed feature importances.
  - Skills showcased: ensemble methods, scikit-learn Pipelines, feature importance interpretation, scalable workflows.

- CA5 — Project / applied case study
  - Summary: Completed an applied case study that integrates the previous steps into a cohesive solution: problem definition, data analysis, modeling, and reporting results with conclusions and suggestions for future work.
  - Skills showcased: end-to-end ML workflow, critical evaluation, clear reporting and visualization, domain-aware recommendations.

## Tools & technologies

- Languages: Python (primary), Markdown
- Key libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, (optionally: xgboost, lightgbm)
- Development: Git, GitHub

## How to use this repo

- View each CA PDF to see the full report, code references, figures, and results.
- If you want runnable notebooks or scripts extracted from the reports, I can add a `notebooks/` or `src/` folder with cleaned code and data-processing pipelines.

## About me / Abilities demonstrated

- Strong data cleaning and preprocessing skills; able to transform messy data into analysis-ready form.
- Practical experience training and evaluating classical and ensemble machine learning models.
- Familiarity with model selection, hyperparameter tuning, and diagnosing model behavior.
- Able to communicate results clearly with visualizations and written summaries.

## Next steps I can do for you

- Extract runnable code from the PDFs into Jupyter notebooks and add example datasets.
- Add scripts and CI to reproduce experiments.
- Expand the README with assignment-specific highlights extracted from each PDF.

---

Note: you asked to rename the repository to `DAT200 Applied Machine Learning`. I updated this README to use that name, but I cannot rename the repository directly from this interface. To rename the repository on GitHub:

1. In your repository on github.com, go to Settings → General → Repository name and change it to `DAT200 Applied Machine Learning`.
2. Or use the GitHub REST API (requires a token) with a PATCH request:

   curl -X PATCH -H "Authorization: token YOUR_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/Mehdij1511/DAT200 \
     -d '{"name":"DAT200 Applied Machine Learning"}'

If you want, I can create a small script or a GitHub Actions workflow to rename the repo using a token you provide, or I can open an issue with the rename request. Tell me which you'd prefer.
