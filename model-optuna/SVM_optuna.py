# 기본 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.samplers import TPESampler

# 텐서플로우 관련 임포트
from tensorflow.keras.optimizers import Adam

# 전처리 및 머신러닝 관련 임포트
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)
# 모델 관련 임포트
from sklearn.svm import SVC

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)


X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


# 2. Optuna 목적 함수 정의 (threshold 제거)
def objective(trial):
    C = trial.suggest_float("C", 0.1, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])

    aucs, f1s = [], []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        model.fit(X_tr, y_tr)

        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_val_prob))
        f1s.append(f1_score(y_val, y_val_pred))

    return 0.5 * np.mean(aucs) + 0.5 * np.mean(f1s)

# 3. Optuna 수행
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42)
)
study.optimize(objective, n_trials=30)


# 4. 최적 파라미터 추출
best_params = study.best_params
best_C = best_params["C"]
best_gamma = best_params["gamma"]
best_kernel = best_params["kernel"]

print("Best parameters:", best_params)

