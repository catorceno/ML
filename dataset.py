import time, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report)

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print('Dimensiones:', X.shape)
print('Clases:', list(data.target_names))        # ['malignant', 'benign']
print(y.value_counts(normalize=True).round(3))    # ¿está balanceado?
print(X.describe().T.round(2))

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
print(X_tr.shape, X_te.shape)
