#### 1. 라이브러리 불러오기
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, precision_recall_curve
)

# Seed 고정 함수
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

#### 2. 데이터 불러오기
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)



#### 3. optuna 딕셔너리
best_params = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.0317270587384051,
    "subsample": 0.7745028729399002,
    "colsample_bytree": 0.9984296690008223,
    "scale_pos_weight": 6.8657996734034485,
    "use_label_encoder": False,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss"
}


#### 4. 모델 학습
xgb_model = XGBClassifier(**best_params)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)



#### 5. 예측 및 평가 지표 출력
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
spec = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)




#### 6. Threshold 최적화 - Youden's J
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
youden_j = tpr - fpr
best_idx_youden = np.argmax(youden_j)
best_th_youden = thresholds_roc[best_idx_youden]
y_pred_youden = (y_pred_prob >= best_th_youden).astype(int)
acc_y = accuracy_score(y_test, y_pred_youden)
prec_y = precision_score(y_test, y_pred_youden)
rec_y = recall_score(y_test, y_pred_youden)
f1_y = f1_score(y_test, y_pred_youden)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_youden).ravel()
spec_y = tn / (tn + fp)

# F1 Score 기준
precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
idx_f1 = np.argmax(f1_scores)
th_f1 = thresholds_pr[idx_f1] if idx_f1 < len(thresholds_pr) else 0.5
y_pred_f1 = (y_pred_prob >= th_f1).astype(int)
acc_f = accuracy_score(y_test, y_pred_f1)
prec_f = precision_score(y_test, y_pred_f1)
rec_f = recall_score(y_test, y_pred_f1)
f1_f = f1_score(y_test, y_pred_f1)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_f1).ravel()
spec_f = tn / (tn + fp)




#### 7. 시각화
thresholds_plot = np.linspace(0, 1, 200)
rec_list, spec_list, f1_list = [], [], []
for t in thresholds_plot:
    y_pred_t = (y_pred_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    rec_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    f1_list.append(f1_score(y_test, y_pred_t, zero_division=0))

plt.figure(figsize=(12, 6))
plt.plot(thresholds_plot, rec_list, label='Recall', color='blue')
plt.plot(thresholds_plot, spec_list, label='Specificity', color='orange')
plt.plot(thresholds_plot, f1_list, label='F1 Score', color='green', linestyle='--')
plt.axvline(best_th_youden, color='red', linestyle=':', label=f"Youden's J = {best_th_youden:.2f}")
plt.axvline(th_f1, color='green', linestyle=':', label=f"Best F1 = {th_f1:.2f}")
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold vs Recall / Specificity / F1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#### 8. 성능 요약 출력
print("\n 결과 요약 (Threshold 기준별 성능)")

print(f"[기본 Threshold 0.5]")
print(f"AUC : {auc:.4f} | Accuracy : {acc:.4f} | Precision : {prec:.4f} | Recall : {rec:.4f} | Specificity : {spec:.4f} | F1 Score : {f1:.4f}")

print(f"\n[Youden's J 기준 Threshold: {best_th_youden:.4f}]")
print(f"AUC : {roc_auc_score(y_test, y_pred_youden):.4f} | Accuracy : {acc_y:.4f} | Precision : {prec_y:.4f} | Recall : {rec_y:.4f} | Specificity : {spec_y:.4f} | F1 Score : {f1_y:.4f}")

print(f"\n[F1 Score 기준 Threshold: {th_f1:.4f}]")
print(f"AUC : {roc_auc_score(y_test, y_pred_f1):.4f} | Accuracy : {acc_f:.4f} | Precision : {prec_f:.4f} | Recall : {rec_f:.4f} | Specificity : {spec_f:.4f} | F1 Score : {f1_f:.4f}")