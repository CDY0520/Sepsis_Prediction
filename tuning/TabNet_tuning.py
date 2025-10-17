#### 1. 라이브러리
import os
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # 완전 재현성 확보 (느릴 수 있음)
    torch.backends.cudnn.benchmark = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import random
import torch
import tensorflow as tf
from torch.nn.functional import threshold
from pytorch_tabnet.tab_model import TabNetClassifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.python.keras import Sequential

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score, precision_recall_curve
)

# from scipy.stats import chi2_contingency, ttest_ind

set_seed(42)

#### 2. 데이터 로드
X_train = pd.read_csv("X_train.csv").astype(np.float32)
X_test = pd.read_csv("X_test.csv").astype(np.float32)
y_train = pd.read_csv("y_train.csv").squeeze().astype(np.int64)
y_test = pd.read_csv("y_test.csv").squeeze().astype(np.int64)

# 범주형 변수 정의
categorical_columns = [col for col in [
    'liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal',
    'Immunocompromised', 'arf', 'Sex', 'mv', 'vaso','AF_hos'
] if col in X_train.columns]

# Label Encoding
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# TabNet용 인덱스 정보
cat_idxs = [i for i, col in enumerate(X_train.columns) if col in categorical_columns]
cat_dims = [X_train[col].nunique() for col in categorical_columns]



#### 3. TabNet 모델 정의 (최적 파라미터 적용)
tabnet_params = {
    'n_d': 64,
    'n_a': 32,
    'n_steps': 10,
    'gamma': 1.3015577906379272,
    'lambda_sparse': 0.002043089014816355,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': {'lr': 0.01350807659307432},
    'cat_idxs': cat_idxs,
    'cat_dims': cat_dims,
    'cat_emb_dim': 10,
    'mask_type': 'sparsemax'
}

clf = TabNetClassifier(**tabnet_params)




#### 4. 모델 학습
clf.fit(
    X_train=X_train.values,
    y_train=y_train.values,
    eval_set=[(X_train.values, y_train.values), (X_test.values, y_test.values)],
    eval_name=['train', 'valid'],
    eval_metric=['auc', 'accuracy'],
    max_epochs=100,
    patience=30,
    batch_size=1024,
    virtual_batch_size=256,
    num_workers=0,
    weights=1,
    drop_last=False
)




#### 5. 예측 및 평가 지표 출력
y_pred_prob = clf.predict_proba(X_test.values)[:, 1]
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
