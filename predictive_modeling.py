"""
Predictive Modeling Using Machine Learning - starter pipeline
Usage: python predictive_modeling.py your_data.csv target_column
Outputs: charts saved in ./model_charts/ and a printed model comparison.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)

sns.set_theme(style="whitegrid")
os.makedirs("model_charts", exist_ok=True)

# 1. LOAD DATA ----------------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "your_data.csv"
target = sys.argv[2] if len(sys.argv) > 2 else "target_column"

df = pd.read_csv(path)
df = df.drop_duplicates().dropna()
print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. PREPARE FEATURES ---------------------------------------------------------
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y = df[target]
if y.dtype == "object":
    y = y.astype("category").cat.codes  # encode labels as 0/1/2...

is_binary = len(np.unique(y)) == 2

# 3. TRAIN / TEST SPLIT -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. MODELS -------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
roc_fig, roc_ax = plt.subplots(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    results[name] = acc

    print(f"\n=== {name} ===")
    print(f"Test accuracy: {acc:.3f} | 5-fold CV accuracy: {cv:.3f}")
    print(classification_report(y_test, pred))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(f"model_charts/confusion_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close(fig)

    # ROC curve (binary targets only)
    if is_binary:
        prob = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")

if is_binary:
    roc_ax.plot([0, 1], [0, 1], "k--")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("ROC Curves")
    roc_ax.legend()
    roc_fig.tight_layout()
    roc_fig.savefig("model_charts/roc_curves.png", dpi=150)
plt.close(roc_fig)

# 5. COMPARE MODELS -----------------------------------------------------------
plt.figure(figsize=(7, 4))
pd.Series(results).sort_values().plot(kind="barh", title="Model accuracy comparison")
plt.tight_layout()
plt.savefig("model_charts/model_comparison.png", dpi=150)
plt.close()

best = max(results, key=results.get)
print(f"\nBest model by test accuracy: {best} ({results[best]:.3f})")

# 6. FEATURE IMPORTANCE (Random Forest) ---------------------------------------
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
plt.figure(figsize=(7, 5))
imp.sort_values().plot(kind="barh", title="Top 10 important features")
plt.tight_layout()
plt.savefig("model_charts/feature_importance.png", dpi=150)
plt.close()

print("Done. Charts are in the model_charts/ folder.")
