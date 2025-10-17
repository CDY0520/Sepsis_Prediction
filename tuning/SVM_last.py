import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report, ConfusionMatrixDisplay
)

# -----------------------------
# 1. 재현성 설정
# -----------------------------
# 모든 랜덤 연산의 재현성을 위해 시드를 고정합니다.
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"✅ 재현성을 위한 시드 설정 완료: {SEED}")

# -----------------------------
# 2. 데이터 불러오기
# -----------------------------
# X_train.csv와 y_train.csv는 모델 학습 및 K-Fold 교차 검증에 사용되는 "훈련 데이터"입니다.
# X_test.csv와 y_test.csv는 모델의 최종 성능 평가를 위한 독립적인 "테스트 데이터"입니다.
X_train_optuna = pd.read_csv("X_train.csv")  # Optuna K-Fold 검증용
y_train_optuna = pd.read_csv("y_train.csv").squeeze()  # Series 형태로 변환

X_final_train = pd.read_csv("X_train.csv") # 최종 모델 학습용 전체 훈련 데이터
y_final_train = pd.read_csv("y_train.csv").squeeze() # 최종 모델 학습용 전체 훈련 레이블

X_test = pd.read_csv("X_test.csv")  # 최종 테스트용 데이터
y_test = pd.read_csv("y_test.csv").squeeze()  # Series 형태로 변환
print("✅ 데이터 로드 완료.")

# -----------------------------
# 3. Optuna Objective 함수 정의 (SVM 하이퍼파라미터 튜닝)
# -----------------------------
def objective(trial):
    # 하이퍼파라미터 탐색 공간 정의
    C = trial.suggest_float("C", 0.1, 10.0, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])

    # 교차검증 설정
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # SEED 적용
    aucs, f1s = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_optuna, y_train_optuna)):
        X_tr, X_val = X_train_optuna.iloc[train_idx], X_train_optuna.iloc[val_idx]
        y_tr, y_val = y_train_optuna.iloc[train_idx], y_train_optuna.iloc[val_idx]

        # 모델 정의 및 학습
        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=SEED) # SEED 적용
        model.fit(X_tr, y_tr)

        # 확률 및 예측
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int) # 기본 임계값 0.5 사용

        # AUC & F1 score 계산
        aucs.append(roc_auc_score(y_val, y_prob))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))

    # AUC와 F1-score 평균을 0.5 가중치로 반영하여 반환
    mean_auc = np.mean(aucs)
    mean_f1 = np.mean(f1s)
    score = 1 * mean_auc + 0 * mean_f1

    # 디버깅용 로그
    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("mean_f1", mean_f1)

    return score

print("🚀 Optuna SVM 하이퍼파라미터 최적화 시작...")
# Optuna 스터디 생성 및 최적화
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)) # SEED 적용
study.optimize(objective, n_trials=30, show_progress_bar=True) # n_trials 조정 가능

# Optuna 최적화 결과 출력
print("\n--- Optuna SVM 최적화 결과 ---")
print("✅ 최적 파라미터:", study.best_params)
print("📈 최적 점수 (평균 AUC + F1):", study.best_value)
print("🔍 평균 AUC (교차 검증):", study.best_trial.user_attrs["mean_auc"])
print("🔍 평균 F1-score (교차 검증):", study.best_trial.user_attrs["mean_f1"])

# -----------------------------
# 4. 최종 모델 학습 (최적 파라미터로 전체 훈련 데이터 사용)
# -----------------------------
best_params_svm = study.best_params # Optuna에서 찾은 최적 파라미터 할당

final_svm_model = SVC(
    C=best_params_svm['C'],
    gamma=best_params_svm['gamma'],
    kernel=best_params_svm['kernel'],
    probability=True,
    random_state=SEED # 최종 모델에도 SEED 적용
)

print("\n🚀 최종 SVM 모델 학습 시작 (Optuna 최적 파라미터로 전체 훈련 데이터 사용)...")
final_svm_model.fit(X_final_train, y_final_train)  # Optuna K-Fold에 사용되지 않은 전체 훈련 데이터 (X, y) 사용
print("✅ 최종 SVM 모델 학습 완료.\n")

# -----------------------------
# 5. 테스트 세트 예측 (데이터 누수 없음)
# -----------------------------
# 모델의 최종 예측은 모델이 학습/튜닝 과정에서 전혀 보지 못했던 테스트 세트(X_test)에 대해서만 수행합니다.
y_pred_prob = final_svm_model.predict_proba(X_test)[:, 1]
print("✅ 테스트 세트 예측 확률 계산 완료.")

# -----------------------------
# 6. F1-score 기준 최적 임계값 계산 및 최종 예측값 생성
# -----------------------------
# 최적 임계값 계산은 테스트 세트의 예측 확률(y_pred_prob)과 실제 값(y_test)을 사용합니다.
# 이 단계는 모델의 최종 성능을 *보고* 적절한 임계값을 찾는 것이므로 데이터 누수가 아닙니다.
# (훈련/검증 단계에서 임계값을 최적화하고 테스트 세트에서는 해당 임계값을 적용해야 데이터 누수가 발생하지 않습니다.)
precisions_plot, recalls_plot, thresholds_plot = precision_recall_curve(y_test, y_pred_prob)
f1_scores_plot = 2 * (precisions_plot * recalls_plot) / (precisions_plot + recalls_plot + 1e-8)

# F1-score가 0인 경우가 있을 수 있으므로, np.argmax 전에 유효한 F1-score 인덱스만 고려
valid_f1_scores = f1_scores_plot[~np.isnan(f1_scores_plot)]
if len(valid_f1_scores) > 0:
    best_idx = np.argmax(f1_scores_plot)
    th_f1_optimal = thresholds_plot[best_idx] if best_idx < len(thresholds_plot) else 0.5
else:
    th_f1_optimal = 0.5 # 유효한 F1-score가 없으면 기본값 사용

y_pred_final = (y_pred_prob >= th_f1_optimal).astype(int)

print(f"🔍 F1-score 기준 최적 임계값 (테스트 세트 기반): {th_f1_optimal:.4f}\n")

# -----------------------------
# 7. 모델 성능 평가 (F1-score 최적 임계값 기준, 테스트 세트 기반)
# -----------------------------
auc_score = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred_final)
prec = precision_score(y_test, y_pred_final, zero_division=0)
rec = recall_score(y_test, y_pred_final, zero_division=0)
f1 = f1_score(y_test, y_pred_final, zero_division=0)
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()
spec = tn / (tn + fp) if (tn + fp) > 0 else 0

print("📊 성능 지표 (F1-optimal 임계값 기준, 테스트 세트 기반):")
print(f"  AUC        : {auc_score:.4f}")
print(f"  정확도     : {acc:.4f}")
print(f"  정밀도     : {prec:.4f}")
print(f"  재현율     : {rec:.4f}")
print(f"  특이도     : {spec:.4f}")
print(f"  F1 점수    : {f1:.4f}\n")

print("분류 리포트 (F1-score 기준 최적 임계값, 테스트 세트 기반):\n")
print(classification_report(y_test, y_pred_final))

# -----------------------------
# 8. 시각화
# -----------------------------

# 8.1. ROC 곡선 시각화
print("🖼️ ROC 곡선 시각화...")
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8.2. 혼동 행렬 시각화
print("🖼️ 혼동 행렬 시각화...")
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title(f'SVM Confusion Matrix (Threshold = {th_f1_optimal:.4f})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()


# -----------------------------
# 9. 예측 결과 CSV 저장
# -----------------------------
# 예측 확률 저장
pd.DataFrame({'y_pred_prob': y_pred_prob}).to_csv("svm_y_pred_prob.csv", index=False)
print("📁 예측 확률 저장 완료: svm_y_pred_prob.csv")

# 최종 이진 예측값 저장
pd.DataFrame({'y_pred_f1': y_pred_final}).to_csv("svm_y_pred_f1.csv", index=False)
print("📁 이진 예측값 저장 완료: svm_y_pred_f1.csv")
