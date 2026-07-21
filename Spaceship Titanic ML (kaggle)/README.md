<div align="center">

# 🚀 Spaceship Titanic — Passenger Transport Prediction

**Binary classification project built from scratch to learn end-to-end ML: cleaning → feature engineering → model comparison → hyperparameter tuning → submission**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Final%20Model-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20Series-20BEFF?logo=kaggle&logoColor=white)
![Status](https://img.shields.io/badge/Validation%20Accuracy-78.9%25-brightgreen)

[Problem](#-the-problem) • [Approach](#-approach) • [Results](#-results) • [How to Run](#-how-to-run) • [Learnings](#-key-learnings)

</div>

---

## 📌 The Problem

In the year 2912, the *Spaceship Titanic* collided with a spacetime anomaly. Roughly half its
passengers were transported to an alternate dimension. Given each passenger's info — home planet,
cabin, spending habits, cryosleep status — **predict whether they were transported**.

This is a **binary classification** problem, evaluated on accuracy, built as a hands-on learning
project rather than a copy-pasted tutorial — every cleaning decision and modeling choice below has a
reasoned "why" behind it.

## 🗂 Dataset

| | |
|---|---|
| Source | [Kaggle — Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) |
| Rows | 8,693 (train) / 4,277 (test) |
| Target | `Transported` (True / False) — ~50.4% / 49.6% split (balanced) |
| Missing data | ~2% per feature, handled with domain logic where possible |

## 🛠 Approach

<details>
<summary><b>1. Data Cleaning — click to expand</b></summary>

<br>

Rather than blindly filling gaps with mean/mode everywhere, missing values were recovered using
**domain logic**:

- **CryoSleep** — sleeping passengers can't spend money, so zero total spending strongly implies
  `CryoSleep = True`. Validated against the actual missing-data subset before applying.
- **Cabin** — recovered from cabin-mates in the same travel group (extracted from `PassengerId`)
  where possible; mode fallback (`Deck='F'`, `Side='S'`) for solo travelers with no group-mate.
- **Age** — filled with **median**, not mean, due to mild right-skew in the distribution.
- **Spending columns** — CryoSleep passengers filled with 0 first, remaining gaps filled with median.
- **HomePlanet / Destination / VIP** — mode fallback, no reliable logical rule available.

</details>

<details>
<summary><b>2. Feature Engineering — click to expand</b></summary>

<br>

- Split `Cabin` (`deck/num/side`) into three usable columns
- Extracted travel `Group` from `PassengerId` to recover missing cabin data
- One-hot encoded multi-category columns (`HomePlanet`, `Destination`, `Deck`, `Side`) — avoids
  falsely implying a numeric order between categories
- Converted boolean columns (`CryoSleep`, `VIP`, `Transported`) to 0/1

</details>

<details>
<summary><b>3. Modeling — click to expand</b></summary>

<br>

Three model families compared, each with a documented reason for trying it:

| Model | Purpose | Val. Accuracy |
|---|---|---|
| Logistic Regression | Baseline — needed feature scaling to fix a convergence issue | 0.7798 |
| Random Forest (default) | Expected to beat baseline; barely did → signal of overfitting | 0.7826 |
| Random Forest (tuned) | Fixed via `max_depth=10`, `min_samples_leaf=5` | 0.7872 |
| **XGBoost (RandomizedSearchCV, 3-fold CV)** | **Systematic tuning, sequential error-correction** | **0.7890 ✅** |

</details>

<details>
<summary><b>4. Validation Strategy — click to expand</b></summary>

<br>

- 80/20 train-validation split (`random_state=42` for reproducibility)
- 3-fold cross-validation during hyperparameter search
- Every fill-statistic (median/mode) and the feature scaler fit **only on training data**, then
  applied unchanged to validation/test — no data leakage

</details>

## 📊 Results

```
Final Model:        XGBoost Classifier
Best Parameters:    n_estimators=400, max_depth=6, learning_rate=0.01,
                     subsample=0.8, colsample_bytree=1.0
Cross-val Accuracy: 80.77%
Held-out Val Acc:   78.90%
```

## 💻 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/<Sahil-SCOE>/ML Project Portfolio/spaceship-titanic-ml.git
cd spaceship-titanic-ml

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost

# 3. Download competition data from Kaggle and place train.csv / test.csv in /data

# 4. Run the notebook
jupyter notebook spaceship_titanic.ipynb
```

## 📁 Project Structure

```
spaceship-titanic-ml/
├── spaceship_titanic.ipynb        # Full notebook — cleaning, EDA-lite, modeling, submission
├── submission.csv                 # Final Kaggle submission
├── docs/
│   ├── Spaceship_Titanic_Learning_Report.pdf   # Full step-by-step learning writeup
│   └── Spaceship_Titanic_Project_Report.pdf    # Concise model-selection report
└── README.md
```

## 🎓 Key Learnings

- **A baseline model isn't optional** — without Logistic Regression's 0.7798 to compare against, a
  first Random Forest scoring 0.7826 would have looked fine instead of exposing overfitting.
- **Feature scaling matters for linear models, not tree-based ones** — diagnosed a real
  `ConvergenceWarning` and fixed the actual cause (unscaled features) instead of just raising
  `max_iter`.
- **Domain logic beats blind imputation** — inferring `CryoSleep` from spending behavior recovered
  real signal that a plain mode-fill would have thrown away.
- **Systematic tuning (RandomizedSearchCV + cross-validation) beats manual guessing** — found a
  meaningfully better parameter combination than hand-tuning would likely have found.

## 📄 Full Reports

For the complete step-by-step reasoning behind every cleaning and modeling decision (with diagrams),
see [`docs/Spaceship_Titanic_Learning_Report.pdf`](docs/Spaceship_Titanic_Learning_Report.pdf).

For a concise technical summary (input/output spec, model comparison, evaluation methodology), see
[`docs/Spaceship_Titanic_Project_Report.pdf`](docs/Spaceship_Titanic_Project_Report.pdf).

---

<div align="center">
Built as a hands-on learning project — first end-to-end ML competition project.
</div>
