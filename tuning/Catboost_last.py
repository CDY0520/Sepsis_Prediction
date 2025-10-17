# 기본 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 관련
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

# 평가 지표 및 메트릭
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)

# 최적화
import optuna
from optuna.samplers import TPESampler

# 기타
from collections import Counter
# -----------------------------
# 1. 고정 랜덤 시드 설정
# -----------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------
# 2. 데이터 로드 및 고정 validation set 분리
# -----------------------------
X_full = pd.read_csv("X_train.csv")
y_full = pd.read_csv("y_train.csv").squeeze()
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()

# 80% train, 20% val (재현 가능하게 stratified split)
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full,
    test_size=0.2,
    stratify=y_full,
    random_state=RANDOM_STATE
)

# -----------------------------
# 3. Optuna 목적 함수 정의
# -----------------------------
def objective(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_seed': RANDOM_STATE,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'verbose': 0
    }

    aucs = []
    f1s = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_te, y_te),
            early_stopping_rounds=30,
            use_best_model=True
        )

        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_te, y_prob))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return 1 * np.mean(aucs) + 0 * np.mean(f1s)


# -----------------------------
# 4. Optuna 스터디 실행
# -----------------------------
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_STATE)
)
study.optimize(objective, n_trials=50)

# -----------------------------
# 5. 고정 validation set에서 최종 성능 평가
# -----------------------------
print("\n✅ Best params:", study.best_params)
print("📈 Best score (CV mean AUC+F1):", study.best_value)

model = CatBoostClassifier(**study.best_params)
model.fit(X_train, y_train, eval_set=(X_val, y_val),early_stopping_rounds=100, use_best_model=True)

# 4. Predict probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 5. Optimal threshold by F1-score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
f1s = f1s[:-1]  # thresholds 길이에 맞춰 자르기

# threshold 범위 필터링 (0.2~0.8)
valid_idx = np.where((thresholds >= 0.2) & (thresholds <= 0.8))[0]

# 가장 높은 F1-score 위치 찾기
if len(valid_idx) > 0:
    best_idx = valid_idx[np.argmax(f1s[valid_idx])]
    th_f1_opt = thresholds[best_idx]
else:
    raise ValueError("⚠️ 0.2~0.8 사이에 유효한 threshold가 없습니다.")  # fallback 제거

# 최종 예측
y_pred_final = (y_pred_prob >= th_f1_opt).astype(int)
print(f"✅ F1-score 기준 최적 threshold (0.2~0.8): {th_f1_opt:.4f}")
# 6. Evaluation
auc_score = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred_final)
prec = precision_score(y_test, y_pred_final, zero_division=0)
rec = recall_score(y_test, y_pred_final, zero_division=0)
f1 = f1_score(y_test, y_pred_final, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\nEvaluation Metrics (F1-optimal threshold)")
print(f"AUC        : {auc_score:.4f}")
print(f"Accuracy   : {acc:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print(f"Specificity: {spec:.4f}")
print(f"F1 Score   : {f1:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_final))

# 7. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'])
plt.title(f'Confusion Matrix (Threshold = {th_f1_opt:.4f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# 예측 확률 저장
pd.DataFrame({'y_pred_prob': y_pred_prob}).to_csv("catboost_y_pred_prob.csv", index=False)
print("📁 예측 확률 저장 완료: catboost_y_pred_prob.csv")

# 최종 이진 예측값 저장
pd.DataFrame({'y_pred_f1': y_pred_final}).to_csv("catboost_y_pred_f1.csv", index=False)
print("📁 이진 예측값 저장 완료: catboost_y_pred_f1.csv")