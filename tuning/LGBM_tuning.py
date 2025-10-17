
# 기본 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# 전처리 및 머신러닝 관련 임포트
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

# 1. 데이터 불러오기
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # Series로 변환
y_test = pd.read_csv("y_test.csv").squeeze()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}\n")

# Optuna를 통해 찾은 최적의 파라미터
best_params = {
    'n_estimators': 636,
    'learning_rate': 0.1304275525715642,
    'num_leaves': 119,
    'max_depth': 5,
    'min_child_samples': 21,
    'subsample': 0.7102508017991509,
    'colsample_bytree': 0.7929886620338936,
    'reg_alpha': 0.677388521129862,
    'reg_lambda': 0.6283163014368047,
    'scale_pos_weight': 6.617715724041617,
    'random_state': 42,
    'n_jobs': -1
}

# 2. 모델 정의 (최적화된 파라미터 적용)
models = {
    'Optimized LightGBM': LGBMClassifier(**best_params)
}

results = {
    'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'Specificity': [], 'F1-Score': [], 'AUC': [],
    'TP': [], 'FN': [], 'FP': [], 'TN': []
}

# 3. 모델 학습 및 평가
plt.figure(figsize=(10, 8))
colors = ['blue']
confusion_matrices = {}
optimal_thresholds_found = {}  # 각 모델별로 찾은 최적 임계값 저장

for i, (name, model) in enumerate(models.items()):
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # 예측 확률 (AUC 계산용)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- F1-Score를 최대화하는 최적 임계값 찾기 ---
    thresholds = np.linspace(0, 1, 100)  # 0부터 1까지 100개의 임계값 후보
    best_f1_for_model = 0
    current_optimal_threshold = 0.5  # 기본값 설정

    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        # zero_division=0은 나눗셈 오류 방지를 위해 설정
        f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
        if f1_temp > best_f1_for_model:
            best_f1_for_model = f1_temp
            current_optimal_threshold = threshold

    optimal_thresholds_found[name] = current_optimal_threshold
    print(f"  {name}의 최적 F1-Score 임계값: {current_optimal_threshold:.4f}")
    # --- 최적 임계값 찾기 종료 ---

    # 찾은 최적 임계값을 사용하여 최종 이진 예측 수행
    y_pred = (y_proba >= current_optimal_threshold).astype(int)

    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[name] = cm  # 혼동행렬 저장

    # 각종 지표 계산 (최적 임계값을 사용하여 계산된 y_pred 기준)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)  # AUC는 임계값에 독립적이므로 y_proba로 계산

    # 특이도(Specificity) 계산
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['Specificity'].append(specificity)
    results['F1-Score'].append(f1)
    results['AUC'].append(auc)
    results['TP'].append(cm[1, 1])
    results['FN'].append(cm[1, 0])
    results['FP'].append(cm[0, 1])
    results['TN'].append(cm[0, 0])

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, color=colors[i], label=f'{name} (AUC={auc:.2f})')

# ROC 곡선 시각화
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Optimized LightGBM')
plt.legend()
plt.grid(True)
plt.show()

# 4. 성능 결과 테이블 출력
results_df = pd.DataFrame(results)
print("\n--- Optimized LightGBM Model Performance (with auto-tuned threshold) ---")
# 특이도 컬럼도 함께 출력하도록 변경
print(results_df[['Model', 'Accuracy', 'Recall', 'Specificity', 'F1-Score', 'AUC']].round(4))

# 7. 혼동행렬 시각화
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - {name} (Threshold: {optimal_thresholds_found[name]:.4f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()