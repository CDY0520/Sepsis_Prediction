import random
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
from optuna.samplers import TPESampler

random.seed(42)
np.random.seed(42)


X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").squeeze()

# Optuna 목적 함수 정의
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'class_weight': 'balanced_subsample',
        'n_jobs': -1,
        'random_state': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_train_cv, y_train_cv)

        y_pred_prob = model.predict_proba(X_val_cv)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val_cv, y_pred_prob))
        f1s.append(f1_score(y_val_cv, y_pred))

    mean_auc = np.mean(aucs)
    mean_f1 = np.mean(f1s)

    # AUC와 F1 Score 가중 평균
    return 0.5 * mean_auc + 0.5 * mean_f1

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42)
)
study.optimize(objective, n_trials=50)

# 결과 출력
print("🔍 Best Parameters:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")
print(f"\n🎯 Best Score (AUC+F1/2): {study.best_value:.4f}")