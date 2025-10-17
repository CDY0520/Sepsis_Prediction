#### 1. 라이브러리
import optuna
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

#### 2. 데이터 로딩 및 전처리
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv").squeeze()

# 범주형 인코딩
categorical_cols = [col for col in [
    'liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal',
    'Immunocompromised', 'arf', 'Sex', 'mv', 'vaso', 'AF_hos'
] if col in X.columns]

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_np = X.values
y_np = y.values

#### 3. Optuna 목적 함수 정의 (AUC + F1 + threshold 튜닝)
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 7.0),
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss"
    }



    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []

    for train_idx, val_idx in skf.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_proba))
        f1s.append(f1_score(y_val, y_pred))

    mean_auc = np.mean(aucs)
    mean_f1 = np.mean(f1s)

    return 0.5 * mean_auc + 0.5 * mean_f1

#### 4. Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 결과 출력
print("Best parameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
print(f"Best score: {study.best_value:.4f}")
