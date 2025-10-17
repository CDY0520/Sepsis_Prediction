#### 라이브러리
import pandas as pd
import numpy as np
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder



#### 데이터 불러오기
X = pd.read_csv("X_train.csv").astype(np.float32)
y = pd.read_csv("y_train.csv").squeeze().astype(np.int64)

# 범주형 인코딩
categorical_columns = [col for col in [
    'liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal',
    'Immunocompromised', 'arf', 'Sex', 'mv', 'vaso', 'AF_hos'
] if col in X.columns]

# 라벨 인코딩 (범주형 -> 숫자형)
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

cat_idxs = [i for i, col in enumerate(X.columns) if col in categorical_columns]
cat_dims = [X[col].nunique() for col in categorical_columns]

X_np = X.values
y_np = y.values




#### Optuna 목적 함수 정의
def objective(trial):
    params = {
        "n_d": trial.suggest_int("n_d", 16, 64, step=16),
        "n_a": trial.suggest_int("n_a", 16, 64, step=16),
        "n_steps": trial.suggest_int("n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
        "optimizer_params": dict(lr=trial.suggest_float("lr", 1e-3, 2e-2, log=True)),
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_emb_dim": 10,
        "mask_type": "entmax",
        "verbose": 0
    }


    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, f1s = [], []

    for train_idx, val_idx in skf.split(X_np, y_np):
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]

        model = TabNetClassifier(**params)
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False
        )

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_proba))
        f1s.append(f1_score(y_val, y_pred))

    mean_auc = np.mean(aucs)
    mean_f1 = np.mean(f1s)
    weighted_score = 0.5 * mean_auc + 0.5 * mean_f1

    return weighted_score




#### Optuna 튜닝 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)



#### 최적 결과 출력
print("Best params:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Best score: {study.best_value:.4f}")
