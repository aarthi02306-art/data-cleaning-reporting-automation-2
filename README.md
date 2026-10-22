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

# 1. Load data (use your cleaned dataset)
df = pd.read_csv("your_data.csv")      # <- change this
target = "target_column"               # <- the column you want to predict

# 2. Prepare features
df = df.drop_duplicates().dropna()
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
y = df[target]
if y.dtype == "object":
    y = y.astype("category").cat.codes   # encode labels as 0/1

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4. Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
plt.figure(figsize=(7, 6))
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
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # ROC curve (binary targets only)
    if len(np.unique(y)) == 2:
        prob = model.predict_proba(X_test_s)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")

if len(np.unique(y)) == 2:
    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.show()

# 5. Compare models
pd.Series(results).sort_values().plot(kind="barh", title="Model accuracy comparison")
plt.show()

# 6. Feature importance (Random Forest)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(10)
imp.sort_values().plot(kind="barh", title="Top 10 important features")
plt.show()
