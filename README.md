# Iris Flower Classification — Predictive Modeling

A machine learning project built as part of my data science internship. 
The goal was to classify iris flower species using petal and sepal 
measurements, and compare multiple classification algorithms to find 
the best performer.

## What This Project Does

I took the classic Iris dataset and built a full ML pipeline — from raw 
data exploration to training three different models and evaluating which 
one works best. The final output includes confusion matrices, ROC curves, 
and feature importance charts.

## Dataset

- **Source:** sklearn.datasets.load_iris (built-in)
- **Rows:** 150 samples
- **Features:** sepal length, sepal width, petal length, petal width (all in cm)
- **Target:** 3 flower species — Setosa, Versicolor, Virginica
- **Missing values:** None

## Project Structure

- `Iris_Flower_Classification.ipynb` — Main notebook, full pipeline
- `README.md` — Project documentation
- `visuals/` — All charts generated inside the notebook

## Steps I Followed

### 1. Data Exploration
Loaded the dataset and checked its shape, class balance, and statistics. 
All 150 samples were evenly split across 3 species (50 each), so no class 
imbalance to deal with.

### 2. Visualizations
- Pairplot — to see how features separate across species
- Heatmap — to check correlations between features
- Boxplots — to compare feature spread per species

I noticed that petal length and petal width are the most separating 
features, while sepal width overlaps quite a bit between classes.

### 3. Preprocessing
- Split data: 75% train, 25% test (stratified)
- Applied StandardScaler so all features are on the same scale — 
  important for Logistic Regression

### 4. Models Trained
- **Logistic Regression** — Good linear baseline
- **Decision Tree** — Interpretable, prone to overfit
- **Random Forest** — Ensemble method, usually more robust

Used 5-fold cross-validation on training data before testing on the 
held-out set.

### 5. Evaluation
- Confusion matrices for each model
- ROC curves using One-vs-Rest strategy
- Classification report with precision, recall, F1

## Results

| Model | CV Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | 96.4% | 92.1% |
| Decision Tree | 92.9% | 92.1% |
| Random Forest | 93.8% | 92.1% |

Logistic Regression had the best cross-validation score. All three 
models reached the same test accuracy, which shows the dataset is 
cleanly separable with any decent algorithm.

## Key Learnings

- Feature scaling matters for distance/gradient-based models like 
  Logistic Regression
- Petal measurements are far more predictive than sepal measurements 
  for this dataset
- Cross-validation gives a more reliable accuracy estimate than a 
  single train-test split
- ROC curves for multi-class problems require the One-vs-Rest approach

## How to Run

1. Open `Iris_Flower_Classification.ipynb` in Google Colab
2. Click Runtime → Run all
3. All charts will display inline — no installs needed

## Tech Stack

- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
