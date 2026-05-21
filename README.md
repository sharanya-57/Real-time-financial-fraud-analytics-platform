# 🛡️ Real-Time Financial Fraud Analytics Platform

> An end-to-end AI-powered web application for detecting fraudulent credit card transactions using Machine Learning, with user authentication, CSV dataset upload, and an interactive analytics dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/sklearn-RandomForest-orange)](https://scikit-learn.org)
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue)](https://sqlite.org)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔐 **Auth System** | Register & login with hashed passwords (Werkzeug bcrypt) |
| 📁 **CSV Upload** | Upload any Kaggle-format transaction CSV (up to 150 MB) |
| 🤖 **ML Prediction** | Random Forest scores every transaction row for fraud probability |
| 📊 **Analytics Dashboard** | KPIs, bar chart, doughnut chart, model performance metrics |
| 🚨 **Alert Panel** | Top high-risk flagged transactions highlighted |
| 🗃️ **Analysis History** | Every user has a private history of their uploaded datasets |
| 👤 **Profile Page** | Account overview with cumulative stats |
| 🧠 **SMOTE Balancing** | Training handles extreme class imbalance (0.17% fraud) |
| 💾 **Persistent Storage** | SQLite database per-user analysis records |

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3 (custom design system), Vanilla JS |
| **Charts** | Chart.js 4.x |
| **Backend** | Python 3.10+, Flask 3.x, Flask-SQLAlchemy |
| **Database** | SQLite via SQLAlchemy ORM |
| **Auth** | Werkzeug password hashing, Flask sessions |
| **ML Model** | Scikit-learn — Random Forest Classifier |
| **Data** | Pandas, NumPy |
| **Class Balance** | imbalanced-learn (SMOTE) |
| **Persistence** | Joblib (model serialization) |
| **Plots** | Matplotlib, Seaborn (confusion matrix, feature importance) |

---

## 📁 Project Structure

```
Real-time-financial-fraud-analytics-platform/
│
├── data/
│   └── creditcard.csv          # Kaggle dataset (place here manually)
│
├── models/
│   ├── fraud_model.pkl         # Trained Random Forest
│   ├── scaler.pkl              # StandardScaler for Time & Amount
│   └── metrics.json            # Saved evaluation metrics
│
├── templates/
│   ├── base.html               # Shared base layout
│   ├── landing.html            # Public landing / hero page
│   ├── auth.html               # Login + Register (shared)
│   ├── home.html               # User dashboard after login
│   ├── upload.html             # CSV dataset upload
│   ├── result.html             # Full analytics results dashboard
│   └── profile.html            # User profile page
│
├── static/
│   ├── css/style.css           # Global design system
│   ├── js/main.js              # Shared JS utilities
│   └── images/                 # Generated ML plots
│
├── uploads/                    # Uploaded CSV files (gitignored)
├── screenshots/                # App screenshots
├── main.py                     # Flask application
├── train_model.py              # ML training pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/sharanya-57/Real-time-financial-fraud-analytics-platform.git
cd Real-time-financial-fraud-analytics-platform
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset *(optional but recommended)*

Download [creditcard.csv from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

> **No dataset?** The training script auto-generates realistic synthetic data so the app works out of the box.

### 5. Train the ML model

```bash
python train_model.py
```

Outputs:
- `models/fraud_model.pkl` — trained Random Forest
- `models/scaler.pkl` — fitted StandardScaler
- `models/metrics.json` — accuracy, precision, recall, F1, ROC-AUC
- `static/images/confusion_matrix.png`
- `static/images/feature_importance.png`

### 6. Run the application

```bash
python main.py
```

Open **http://localhost:5000** in your browser.

**Demo login:** username `demo` · password `demo123`

---

## 🔗 Routes

| Route | Method | Auth | Description |
|---|---|---|---|
| `/` | GET | — | Landing page |
| `/register` | GET, POST | — | Create account |
| `/login` | GET, POST | — | Sign in |
| `/logout` | GET | ✓ | Sign out |
| `/home` | GET | ✓ | User dashboard |
| `/upload` | GET, POST | ✓ | Upload CSV & run analysis |
| `/result/<id>` | GET | ✓ | View analysis results |
| `/result/<id>/delete` | POST | ✓ | Delete an analysis |
| `/profile` | GET | ✓ | User profile |

---

## 📊 Dataset

**Kaggle Credit Card Fraud Detection Dataset**
- 284,807 transactions, 2 days
- 492 fraudulent (0.172%)
- 28 PCA-anonymized features (V1–V28) + Time + Amount
- `Class`: 0 = Normal, 1 = Fraud

---

## 🤖 ML Model Details

**Algorithm:** Random Forest Classifier  
**Why Random Forest?**
- Ensemble of 100 decision trees — reduces variance and overfitting
- Native `class_weight='balanced'` handles imbalanced fraud datasets
- Provides interpretable feature importances
- Fast inference (<5 ms per batch)
- No feature scaling required (but applied to Time/Amount anyway for consistency)

**Imbalance handling:** SMOTE oversamples the minority (fraud) class before training

**Typical metrics (Kaggle dataset):**

| Metric | Score |
|---|---|
| Accuracy | ~99.8% |
| Precision | ~94–96% |
| Recall | ~92–95% |
| F1-Score | ~93–95% |
| ROC-AUC | ~97–99% |

---

## 🎤 Interview Q&A

**Q: Why Random Forest?**
> Ensemble averaging reduces variance. `class_weight='balanced'` adjusts loss without needing manual resampling. Feature importance natively interpretable. Sub-millisecond inference per row in production.

**Q: How is class imbalance handled?**
> Two-pronged: (1) SMOTE synthesizes fraud samples in feature space during training, (2) `class_weight='balanced'` up-weights fraud samples in the loss function.

**Q: False positives vs false negatives in fraud?**
> FP = legitimate transaction flagged (customer friction, disputes). FN = fraud missed (direct financial loss). Fraud detection prioritizes **Recall** (minimize FN) while keeping Precision acceptable.

**Q: How does the CSV upload pipeline work?**
> CSV → Pandas DataFrame → feature alignment to V1–V28+Time+Amount → StandardScaler on Time/Amount → `model.predict_proba()` → threshold 0.5 → fraud flag + risk score per row → stored in SQLite → rendered in dashboard.

---

## 🔮 Future Improvements

- [ ] Email alerts on high-risk detections (SendGrid)
- [ ] SHAP Explainable AI — per-transaction feature contribution
- [ ] XGBoost / LightGBM comparison panel
- [ ] Real-time WebSocket transaction stream
- [ ] Docker containerization
- [ ] Cloud deployment (Railway / Render / AWS)
- [ ] Password reset flow

---

## 📄 License

MIT © 2025 Sharanya — [GitHub](https://github.com/sharanya-57)
