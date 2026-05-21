"""
train_model.py — CCFD
Trains Random Forest on Kaggle creditcard.csv (or synthetic data).
Run: python train_model.py
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data", "creditcard.csv")
MODELS = os.path.join(BASE, "models")
IMGS   = os.path.join(BASE, "static", "images")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(IMGS,   exist_ok=True)

FEATURE_COLS = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]

def synthetic_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * 0.02)
    n_norm  = n - n_fraud
    def rows(count, fraud):
        d = {'Time': rng.uniform(0, 172800, count),
             'Amount': rng.exponential(300 if fraud else 120, count)}
        for i in range(1, 29):
            d[f'V{i}'] = rng.normal(
                rng.uniform(-3,3) if fraud else rng.uniform(-1,1),
                rng.uniform(1,4)  if fraud else rng.uniform(.5,2), count)
        d['Class'] = int(fraud)
        return pd.DataFrame(d)
    df = pd.concat([rows(n_norm, False), rows(n_fraud, True)],
                   ignore_index=True).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True)

def main():
    print("="*55)
    print("  CCFD — Random Forest Training Pipeline")
    print("="*55)

    if os.path.exists(DATA):
        print(f"\n[1/6] Loading {DATA}")
        df = pd.read_csv(DATA)
    else:
        print("\n[1/6] creditcard.csv not found → generating synthetic data")
        df = synthetic_data()
        os.makedirs(os.path.join(BASE,"data"), exist_ok=True)
        df.to_csv(DATA, index=False)
    print(f"      {len(df):,} rows | fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")

    print("\n[2/6] Preprocessing …")
    X = df[FEATURE_COLS].copy()
    y = df['Class'].values
    scaler = StandardScaler()
    X[['Time','Amount']] = scaler.fit_transform(X[['Time','Amount']])

    print("\n[3/6] Train/test split 80/20 …")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2,
                                           random_state=42, stratify=y)

    print("\n[4/6] SMOTE balancing …")
    if HAS_SMOTE:
        try:
            Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
            print(f"      After SMOTE — Normal: {(ytr==0).sum():,}  Fraud: {(ytr==1).sum():,}")
        except Exception as e:
            print(f"      SMOTE skipped ({e})")
    else:
        print("      imbalanced-learn not available, skipping SMOTE")

    print("\n[5/6] Training Random Forest (100 trees) …")
    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                 class_weight='balanced', random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)

    print("\n[6/6] Evaluating …")
    yp   = clf.predict(Xte)
    yprb = clf.predict_proba(Xte)[:,1]
    acc  = accuracy_score(yte, yp)
    prec = precision_score(yte, yp, zero_division=0)
    rec  = recall_score(yte, yp, zero_division=0)
    f1   = f1_score(yte, yp, zero_division=0)
    try:   auc = roc_auc_score(yte, yprb)
    except: auc = 0.0

    print(f"\n  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1        : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc*100:.2f}%")
    print("\n" + classification_report(yte, yp, target_names=['Normal','Fraud']))

    joblib.dump(clf,    os.path.join(MODELS,"fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS,"scaler.pkl"))
    metrics = dict(accuracy=round(acc*100,2), precision=round(prec*100,2),
                   recall=round(rec*100,2), f1_score=round(f1*100,2),
                   roc_auc=round(auc*100,2),
                   total_samples=int(len(df)), fraud_samples=int(df['Class'].sum()),
                   normal_samples=int((df['Class']==0).sum()))
    with open(os.path.join(MODELS,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved model, scaler, metrics → models/")

    # Confusion matrix plot
    cm = confusion_matrix(yte, yp)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'], ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig(os.path.join(IMGS,"confusion_matrix.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Feature importance plot
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh([FEATURE_COLS[i] for i in reversed(idx)],
            [imp[i] for i in reversed(idx)], color='steelblue')
    ax.set_xlabel('Importance'); ax.set_title('Top 15 Feature Importances')
    plt.tight_layout()
    fig.savefig(os.path.join(IMGS,"feature_importance.png"), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved plots → static/images/")
    print("\n✓ Training complete!\n")

if __name__ == "__main__":
    main()
