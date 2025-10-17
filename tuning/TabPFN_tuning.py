
# 필요한 라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score,
    precision_recall_curve)
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import LabelEncoder



# 1. 데이터 불러오기
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze().astype(np.int64)
y_test = pd.read_csv("y_test.csv").squeeze().astype(np.int64)

# 2. 범주형/수치형 변수 정의
categorical_vars = [col for col in ['liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal', 'Immunocompromised',
                                    'arf', 'Sex', 'mv', 'vaso', 'AF_hos'] if col in X_train.columns]

continuous_vars = [col for col in ['Age', 'Cr', 'Na', 'k', 'hct', 'wbc', 'pH', 'Temp', 'hr', 'rr',
                                   'mbp', 'PaO2/FiO2', 'gcs', 'Urine output', 'Plt', 'Bil', 'lactate',
                                   'APACHE II score', 'sofa', 'pco2', 'A-a gradient', 'FiO2', 'bmi'] if col in X_train.columns]

# 3. 범주형 변수 Label Encoding
encoded_train = X_train.copy()
encoded_test = X_test.copy()

for col in categorical_vars:
    le = LabelEncoder()
    encoded_train[col] = le.fit_transform(X_train[col].astype(str))
    encoded_test[col] = le.transform(X_test[col].astype(str))

# 4. 모델 정의
MODEL_PATH = "/tabpfn_models/tabpfn-v2-classifier.ckpt"
clf = TabPFNClassifier(
    device='cpu',
    model_path=MODEL_PATH,
    ignore_pretraining_limits=True,
    n_estimators=64
)

# 5. 모델 학습
clf.fit(encoded_train.values, y_train.values)

# 6. 예측 및 평가
y_pred_prob = clf.predict_proba(encoded_test.values)[:, 1]
predictions_labels = clf.predict(encoded_test.values)


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

from sklearn.metrics import roc_auc_score

auc_0 = roc_auc_score(y_test, y_pred_prob)  # AUC는 확률 기반이므로 동일하게 사용
auc_y = roc_auc_score(y_test, y_pred_prob)
auc_f = roc_auc_score(y_test, y_pred_prob)


print("\n📌 결과 요약 (Threshold 기준별 성능)")
print(f"[기본 Threshold 0.5]")
print(f"AUC : {auc_0:.4f} | Accuracy : {acc_0:.4f} | Precision : {prec_0:.4f} | Recall : {rec_0:.4f} | Specificity : {spec_0:.4f} | F1 Score : {f1_0:.4f}")

print(f"\n[Youden's J 기준 Threshold: {th_youden:.4f}]")
print(f"AUC : {auc_y:.4f} | Accuracy : {acc_y:.4f} | Precision : {prec_y:.4f} | Recall : {rec_y:.4f} | Specificity : {spec_y:.4f} | F1 Score : {f1_y:.4f}")

print(f"\n[F1 Score 기준 Threshold: {th_f1:.4f}]")
print(f"AUC : {auc_f:.4f} | Accuracy : {acc_f:.4f} | Precision : {prec_f:.4f} | Recall : {rec_f:.4f} | Specificity : {spec_f:.4f} | F1 Score : {f1_f:.4f}")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# threshold 구간 생성
thresholds_to_test = np.linspace(0.0, 1.0, 200)

recalls = []
specificities = []
f1_scores = []

for t in thresholds_to_test:
    y_pred = (y_pred_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_test, y_pred, zero_division=0)

    recalls.append(recall)
    specificities.append(specificity)
    f1_scores.append(f1)

# Plot
plt.figure(figsize=(12, 7))
plt.plot(thresholds_to_test, recalls, label="Recall (Sensitivity)", color='blue')
plt.plot(thresholds_to_test, specificities, label="Specificity", color='orange')
plt.plot(thresholds_to_test, f1_scores, label="F1 Score", color='green', linestyle='--')

# 기준 threshold 세로선
plt.axvline(th_youden, color='red', linestyle=':', linewidth=2,
            label=f"Best Youden's J: {th_youden:.2f}")
plt.axvline(th_f1, color='green', linestyle=':', linewidth=2,
            label=f"Best F1: {th_f1:.2f}")

# Label 및 출력 포맷
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Threshold vs Recall / Specificity / F1", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()





