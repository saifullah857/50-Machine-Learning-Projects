# 🔊 Sonar Rock vs Mine — Classification ML Project

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **Binary classification** of underwater sonar signals to distinguish **Rocks** from **Mines** using multiple supervised learning algorithms with hyperparameter tuning and cross-validation.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Workflow](#-workflow)
- [Models & Results](#-models--results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Concepts](#-key-concepts)
- [Author](#-author)

---

## 🧭 Overview

This project applies classical machine learning techniques to classify sonar signals bounced off a metal cylinder (mine) or a roughly cylindrical rock. The pipeline covers end-to-end ML: data loading → preprocessing → training → evaluation → hyperparameter optimization via **GridSearchCV** and **Scikit-learn Pipelines**.

The project benchmarks three classifiers:

| Model | Tuned | Scaled |
|---|---|---|
| Logistic Regression | ✅ | ✅ |
| Gaussian Naïve Bayes | ❌ | ❌ |
| K-Nearest Neighbours | ✅ | ✅ |

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | UCI Sonar Dataset |
| **File** | `Copy of sonar data.csv` |
| **Samples** | 208 |
| **Features** | 60 (continuous frequency energy bands) |
| **Target** | Binary — `R` (Rock) / `M` (Mine) |
| **Class Distribution** | 111 Mines · 97 Rocks |

Each feature represents the **energy** within a particular frequency band of the sonar chirp, integrated over a certain period.

> **Target Encoding:** `R → 0` (Rock), `M → 1` (Mine) via `LabelEncoder`.

---

## 📁 Project Structure

```
sonar-classification/
│
├── 📓 Code.ipynb                  # Main Jupyter Notebook (full pipeline)
├── 📄 Copy of sonar_data.csv      # Raw dataset
└── 📖 README.md                   # Project documentation
```

---

## 🛠 Tech Stack

```text
Python 3.10+
├── pandas          — Data loading & manipulation
├── scikit-learn
│   ├── LogisticRegression
│   ├── GaussianNB
│   ├── KNeighborsClassifier
│   ├── StandardScaler
│   ├── LabelEncoder
│   ├── train_test_split
│   ├── GridSearchCV
│   ├── Pipeline
│   └── Metrics (accuracy, precision, recall, classification_report)
└── Jupyter Notebook — Interactive development environment
```

---

## 🔄 Workflow

```
Raw CSV Data
     │
     ▼
┌─────────────────────┐
│  1. Load Dataset    │  pd.read_csv() → 208 × 61 DataFrame
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Encode Target   │  LabelEncoder  R→0, M→1
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Train/Test Split│  80% Train | 20% Test  (random_state=42)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  4. Feature Scaling │  StandardScaler (fit on train, transform both)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│  5. Model Training & Evaluation                     │
│                                                     │
│   ├── Logistic Regression                           │
│   ├── Gaussian Naïve Bayes                          │
│   └── K-Nearest Neighbours (k=3)                   │
└─────────┬───────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  6. Hyperparameter  │  GridSearchCV  k ∈ {3,5,7,9}   CV=5
│     Tuning (KNN)    │  Scoring: Recall
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  7. Pipeline +      │  StandardScaler → Best Classifier
│     GridSearchCV    │  Multi-model grid (LR, KNN, GaussianNB)
└─────────────────────┘
```

---

## 📈 Models & Results

### Logistic Regression *(with StandardScaler)*

```
Accuracy  :  ~85.7 %
Precision :  ~84.6 %
Recall    :  ~91.7 %
```

### Gaussian Naïve Bayes *(no scaling)*

```
Accuracy  :  ~76.2 %
Precision :  ~73.3 %
Recall    :  ~84.6 %
```

### K-Nearest Neighbours *(k=3, with StandardScaler)*

```
Accuracy  :  ~85.7 %
Precision :  ~84.6 %
Recall    :  ~84.6 %
```

### GridSearchCV — KNN Hyperparameter Comparison

| `n_neighbors` | Mean CV Recall (5-fold) |
|:---:|:---:|
| 3 | Highest |
| 5 | — |
| 7 | — |
| 9 | Lowest |

> **Best params determined via `GridSearchCV` with `scoring="recall"`** — optimized for minimizing false negatives (critical in mine detection scenarios).

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sonar-classification.git
cd sonar-classification
```

### 2. Create a Virtual Environment *(recommended)*

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install pandas scikit-learn jupyter notebook
```

---

## 🚀 Usage

### Run the Notebook

```bash
jupyter notebook Code.ipynb
```

Run all cells sequentially from top to bottom. The notebook is self-contained and covers:

- ✅ Data exploration
- ✅ Preprocessing & encoding
- ✅ Model training & evaluation
- ✅ Hyperparameter tuning with `GridSearchCV`
- ✅ Pipeline construction

### Minimal Inference Example

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Copy of sonar data.csv", header=None)

# Encode target
le = LabelEncoder()
df[60] = le.fit_transform(df[60])

X = df.drop(60, axis=1)
y = df[60]

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_s, y_train)
print("Accuracy:", model.score(X_test_s, y_test))
```

---

## 💡 Key Concepts

| Concept | Description |
|---|---|
| **StandardScaler** | Normalizes features to zero mean and unit variance; essential for distance-based models like KNN |
| **LabelEncoder** | Converts categorical labels (`R`, `M`) to integers (`0`, `1`) |
| **Train/Test Split** | 80/20 split with `random_state=42` for reproducibility |
| **Recall (Sensitivity)** | Primary metric — minimizing false negatives is critical in mine detection |
| **GridSearchCV** | Exhaustive hyperparameter search with 5-fold cross-validation |
| **Pipeline** | Chains `StandardScaler → Classifier` to prevent data leakage |
| **Multi-model Grid** | Compares LR, KNN, and GaussianNB in a single `GridSearchCV` run |

---

## 👤 Author

**Your Name**
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/saifullah857/)


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ and <strong>scikit-learn</strong>
</p>