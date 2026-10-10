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

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# Modelo Referencia
arbol = DecisionTreeClassifier(random_state=42)
arbol.fit(X_tr, y_tr)

# Modelo 1: Bootstrap Aggregating
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),  # sklearn<1.2: base_estimator=
    n_estimators=100,     # número de modelos del ensemble
    max_samples=1.0,      # tamaño de cada muestra bootstrap (fracción de N)
    bootstrap=True,       # True = bagging (con reemplazo); False = pasting
    oob_score=True,       # evaluación out-of-bag
    n_jobs=-1, random_state=42)
bagging.fit(X_tr, y_tr)
print('OOB score (Bagging):', round(bagging.oob_score_, 4))

# Modelo 2: Random Forest
bosque = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',  # subconjunto aleatorio de variables por división
    oob_score=True,
    n_jobs=-1, random_state=42)
bosque.fit(X_tr, y_tr)
print('OOB score (Random Forest):', round(bosque.oob_score_, 4))

# Evaluación de Rendimiento
def evaluar(nombre, modelo):
    yp = modelo.predict(X_te)
    proba = modelo.predict_proba(X_te)[:, 1]
    print(f'\n== {nombre} ==')
    print('Accuracy :', round(accuracy_score(y_te, yp), 4))
    print('Precision:', round(precision_score(y_te, yp), 4))
    print('Recall   :', round(recall_score(y_te, yp), 4))
    print('F1       :', round(f1_score(y_te, yp), 4))
    print('ROC-AUC  :', round(roc_auc_score(y_te, proba), 4))
    print('Matriz de confusion:\n', confusion_matrix(y_te, yp))
 
for nombre, m in [('Arbol', arbol), ('Bagging', bagging), ('Random Forest', bosque)]:
    evaluar(nombre, m)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for nombre, m in [('Bagging', bagging), ('Random Forest', bosque)]:
    f1 = cross_val_score(m, X, y, cv=cv, scoring='f1')
    print(nombre, 'F1 CV:', f1.round(3), 'media =', round(f1.mean(), 4))

imp = pd.Series(bosque.feature_importances_, index=X.columns)
imp = imp.sort_values(ascending=False)
print(imp.head(10).round(4))
imp.head(10).plot(kind='barh'); plt.gca().invert_yaxis()
plt.title('Top-10 importancia de variables - Random Forest')
plt.tight_layout(); plt.savefig('importancia_rf.png', dpi=150)

