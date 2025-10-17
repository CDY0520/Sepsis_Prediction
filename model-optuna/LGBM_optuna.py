
# 라이브러리 목록
import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
import lightgbm as lgb

# 데이터 로드
X = pd.read_csv("X_train.csv")
y = pd.read_csv("y_train.csv")
if 'target' in y.columns:
     y = y['target']

# target 클래스 확인
class_counts = y.value_counts()
print("클래스별 개수:")
print(class_counts)
print("\n")

class_proportions = y.value_counts(normalize=True)
print("클래스별 비율:")
print(class_proportions)
print("\n")

# Optuna objective 함수 정의
def objective(trial):
    param = {
        'objective': 'binary', # 이진 분류 문제 설정
        'metric': 'auc', # 모델 평가 지표로 AUC 사용
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500), # 학습할 트리의 개수
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2), # 각 트리의 기여도 (학습률)
        'num_leaves': trial.suggest_int('num_leaves', 20, 300), # 한 트리가 가질 수 있는 최대 잎 노드 개수
        'max_depth': trial.suggest_int('max_depth', 4, 12), # 트리의 최대 깊이
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100), # 리프 노드를 만드는 데 필요한 최소 샘플 수
        'subsample': trial.suggest_float('subsample', 0.6, 1.0), # 각 트리 학습에 사용할 데이터 샘플 비율
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # 각 트리 학습에 사용할 피처(열) 비율
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0), # L1 정규화 강도 (과적합 방지)
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0), # L2 정규화 강도 (과적합 방지)
        'random_state': 42, # 결과 재현성을 위한 난수 시드 고정
        'n_jobs': -1, # 모든 CPU 코어 사용하여 학습 속도 향상
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0) # 불균형 데이터셋에서 소수 클래스 가중치 부여
    }

    threshold = 0.5

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    aucs = []
    recalls = []
    precisions = []
    specificities = []
    accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        y_proba = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, y_proba)
        aucs.append(auc)

        y_pred = (y_proba >= threshold).astype(int)

        f1 = f1_score(y_val, y_pred, zero_division=0)
        f1_scores.append(f1)

        recall = recall_score(y_val, y_pred, zero_division=0)
        recalls.append(recall)

        precision = precision_score(y_val, y_pred, zero_division=0)
        precisions.append(precision)

        specificity = recall_score(1 - y_val, 1 - y_pred, zero_division=0)
        specificities.append(specificity)

        accuracy = accuracy_score(y_val, y_pred)
        accuracies.append(accuracy)


    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(aucs)
    mean_specificity = np.mean(specificities)
    mean_accuracy = np.mean(accuracies)


    trial.set_user_attr("mean_f1_score", mean_f1)
    trial.set_user_attr("mean_auc", mean_auc)
    trial.set_user_attr("mean_recall", np.mean(recalls))
    trial.set_user_attr("mean_precision", np.mean(precisions))
    trial.set_user_attr("mean_specificity", mean_specificity)
    trial.set_user_attr("mean_accuracy", mean_accuracy)


    return (0.5 * mean_f1 + 0.5 * mean_auc)

# Optuna 스터디 생성 및 최적화 시작
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 최적 하이퍼파라미터 및 결과 출력
print("---")
print("Best params:", study.best_params)
print("Best mean (0.5 * F1-Score + 0.5 * AUC):", study.best_value)
print("Mean F1-Score for the best trial:", study.best_trial.user_attrs["mean_f1_score"])
print("Mean AUC for the best trial:", study.best_trial.user_attrs["mean_auc"])
print("Mean Accuracy for the best trial:", study.best_trial.user_attrs["mean_accuracy"])
print("Mean Precision for the best trial:", study.best_trial.user_attrs["mean_precision"])
print("Mean Recall (Sensitivity) for the best trial:", study.best_trial.user_attrs["mean_recall"])
print("Mean Specificity for the best trial:", study.best_trial.user_attrs["mean_specificity"])