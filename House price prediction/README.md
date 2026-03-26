# 🏠 Housing Price Prediction

A end-to-end machine learning project that predicts residential property prices based on structural and locational features. Multiple regression algorithms are benchmarked, with **XGBoost Regressor** identified as the best-performing model.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project builds a supervised regression pipeline to estimate housing prices from a structured tabular dataset. It covers the full ML workflow — data loading, encoding, exploratory analysis, feature scaling, model training, and performance evaluation — using Python and scikit-learn.

---

## Dataset

**File:** `Housing.csv`  
**Rows:** 545 &nbsp;|&nbsp; **Columns:** 13

| Feature | Type | Description |
|---|---|---|
| `price` | int | Target variable — sale price of the property |
| `area` | int | Total area in square feet |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `stories` | int | Number of floors |
| `mainroad` | str (binary) | Whether the property is on a main road (`yes`/`no`) |
| `guestroom` | str (binary) | Availability of a guest room (`yes`/`no`) |
| `basement` | str (binary) | Availability of a basement (`yes`/`no`) |
| `hotwaterheating` | str (binary) | Hot water heating system present (`yes`/`no`) |
| `airconditioning` | str (binary) | Air conditioning present (`yes`/`no`) |
| `parking` | int | Number of parking spaces |
| `prefarea` | str (binary) | Located in a preferred area (`yes`/`no`) |
| `furnishingstatus` | str (categorical) | Furnishing level (`furnished` / `semi-furnished` / `unfurnished`) |

---

## Project Structure

```
housing-price-prediction/
├── Housing.csv          # Raw dataset
├── code.ipynb           # Main Jupyter notebook (full pipeline)
└── README.md            # Project documentation
```

---

## Pipeline

The notebook follows a structured ML workflow:

```
1. Data Loading
       │
       ▼
2. Encoding
   ├── Label Encoding     →  Binary categorical columns (yes/no)
   └── One-Hot Encoding   →  Multi-class column (furnishingstatus)
       │
       ▼
3. Exploratory Data Analysis (EDA)
   ├── Scatter plot  — Price vs. Area
   └── Correlation heatmap
       │
       ▼
4. Train / Test Split   (80% train | 20% test | random_state=42)
       │
       ▼
5. Feature Scaling      (StandardScaler)
       │
       ▼
6. Model Training & Evaluation
       │
       ▼
7. Best Model Selection  →  XGBoost Regressor
```

---

## Models Evaluated

| # | Model | Notes |
|---|---|---|
| 1 | **Linear Regression** | Baseline linear model |
| 2 | **RidgeCV** | L2 regularisation, cross-validated alpha selection |
| 3 | **LassoCV** | L1 regularisation, cross-validated alpha selection |
| 4 | **ElasticNet** | Combined L1 + L2 regularisation |
| 5 | **Decision Tree Regressor** | `max_depth=8`, `min_samples_leaf=2` |
| 6 | **Random Forest Regressor** | 100 estimators, `max_depth=8` |
| 7 | **Gradient Boosting Regressor** | 200 estimators, `learning_rate=0.1` |
| 8 | **XGBoost Regressor** ⭐ | 200 estimators, `learning_rate=0.1`, `max_depth=5` |
| 9 | **SVR (RBF kernel)** | Tuned via GridSearchCV |
| 10 | **K-Nearest Neighbors Regressor** | `n_neighbors=5` |

---

## Results

Models are compared on three standard regression metrics:

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | Mean Absolute Error | Average magnitude of prediction errors |
| **MSE** | Mean Squared Error | Penalises large errors more heavily |
| **R² Score** | Coefficient of Determination | Proportion of variance explained (higher = better) |

> ✅ **Best Model: XGBoost Regressor** — achieved the highest R² score among all evaluated models.

---

## Installation

```bash
# 1. Clone the repository
git https://github.com/saifullah857/50-Machine-Learning-Projects.git
cd housing-price-prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook code.ipynb
```

---

## Usage

1. Place `Housing.csv` in the project root directory.
2. Open `code.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells sequentially (`Kernel → Restart & Run All`).
4. Review per-model evaluation metrics printed at the end of each model section.

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Preprocessing, model training, evaluation |
| `xgboost` | XGBoost Regressor |
| `matplotlib` | Visualisations |
| `seaborn` | Statistical plots (scatter, heatmap) |
| `jupyter` | Interactive notebook environment |

Install all at once:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

---

## License

This project is released under the [MIT License](LICENSE).

---

> *Built with Python 3 · scikit-learn · XGBoost*