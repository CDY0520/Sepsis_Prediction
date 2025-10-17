import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)

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
from sklearn.ensemble import RandomForestClassifier
import random

random.seed(42)
np.random.seed(42)

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# 모델 정의
model = RandomForestClassifier(
    n_estimators=101,
    max_depth=10,
    min_samples_split=16,
    min_samples_leaf=9,
    max_features='log2',
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------
# 1. Threshold = 0.5 (기본)
# -------------------------------
threshold_default = 0.5
y_pred_default = (y_pred_prob >= threshold_default).astype(int)

acc_0 = accuracy_score(y_test, y_pred_default)
prec_0 = precision_score(y_test, y_pred_default)
rec_0 = recall_score(y_test, y_pred_default)
f1_0 = f1_score(y_test, y_pred_default)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_default).ravel()
spec_0 = tn / (tn + fp)

# -------------------------------
# 2. Youden's J 기준
# -------------------------------
fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_prob)
youden_j_arr = tpr - fpr
idx_youden = np.argmax(youden_j_arr)
th_youden = thresholds_roc[idx_youden]
y_pred_youden = (y_pred_prob >= th_youden).astype(int)

acc_y = accuracy_score(y_test, y_pred_youden)
prec_y = precision_score(y_test, y_pred_youden)
rec_y = recall_score(y_test, y_pred_youden)
f1_y = f1_score(y_test, y_pred_youden)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_youden).ravel()
spec_y = tn / (tn + fp)

# -------------------------------
# 3. F1 Score 기준
# -------------------------------
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

# -------------------------------
# 결과 출력
# -------------------------------
auc_score = roc_auc_score(y_test, y_pred_prob)

print("\n📌 결과 요약 (Threshold 기준별 성능)")
print(f"[기본 Threshold 0.5]")
print(f"AUC : {auc_score:.4f} | Accuracy : {acc_0:.4f} | Precision : {prec_0:.4f} | Recall : {rec_0:.4f} | Specificity : {spec_0:.4f} | F1 Score : {f1_0:.4f}")

print(f"\n[Youden's J 기준 Threshold: {th_youden:.4f}]")
print(f"AUC : {auc_score:.4f} | Accuracy : {acc_y:.4f} | Precision : {prec_y:.4f} | Recall : {rec_y:.4f} | Specificity : {spec_y:.4f} | F1 Score : {f1_y:.4f}")

print(f"\n[F1 Score 기준 Threshold: {th_f1:.4f}]")
print(f"AUC : {auc_score:.4f} | Accuracy : {acc_f:.4f} | Precision : {prec_f:.4f} | Recall : {rec_f:.4f} | Specificity : {spec_f:.4f} | F1 Score : {f1_f:.4f}")

# -------------------------------
# Threshold vs Metric Plot
# -------------------------------
thresholds_to_test = np.linspace(0.0, 1.0, 200)
recalls_plot, specificities_plot, f1s_plot = [], [], []

for t in thresholds_to_test:
    y_pred = (y_pred_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)
    recalls_plot.append(recall)
    specificities_plot.append(specificity)
    f1s_plot.append(f1)

plt.figure(figsize=(12, 7))
plt.plot(thresholds_to_test, recalls_plot, label="Recall (Sensitivity)", color='blue')
plt.plot(thresholds_to_test, specificities_plot, label="Specificity", color='orange')
plt.plot(thresholds_to_test, f1s_plot, label="F1 Score", color='green', linestyle='--')

plt.axvline(th_youden, color='red', linestyle=':', linewidth=2, label=f"Best Youden's J: {th_youden:.2f}")
plt.axvline(th_f1, color='green', linestyle=':', linewidth=2, label=f"Best F1: {th_f1:.2f}")

plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Threshold vs Recall / Specificity / F1", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()