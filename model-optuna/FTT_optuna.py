
# 라이브러리 임포트 (기존과 동일)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Lambda

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW
from tabtransformertf.models.fttransformer import FTTransformer, FTTransformerEncoder
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna

# 데이터 로드 (기존과 동일)
X_train_df = pd.read_csv("X_train.csv")  # DataFrame 형태로 유지
X_test_df = pd.read_csv("X_test.csv")  # DataFrame 형태로 유지
y_train = pd.read_csv("y_train.csv").squeeze().astype(np.int64)
y_test = pd.read_csv("y_test.csv").squeeze().astype(np.int64)

# 범주형 변수 정의 기준 (예시: 고유값 20 미만)
categorical_vars = [col for col in ['liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal', 'Immunocompromised',
                                    'arf', 'Sex', 'mv', 'vaso', 'AF_hos'
                                    ] if col in X_train_df.columns]

continuous_vars = [col for col in ['Age', 'Cr', 'Na', 'k', 'hct', 'wbc', 'pH', 'Temp', 'hr', 'rr',
                                   'mbp', 'PaO2/FiO2', 'gcs', 'Urine output', 'Plt', 'Bil', 'lactate',
                                   'APACHE II score', 'sofa', 'pco2', 'A-a gradient', 'FiO2', 'bmi'
                                   ] if col in X_train_df.columns]


# 입력 변환 함수 (기존과 동일)
def reshape_inputs(data_df, cat_cols, num_cols):
    d = {}
    for c in cat_cols:
        d[c] = np.expand_dims(data_df[c].astype(str).values, -1)
    for c in num_cols:
        d[c] = np.expand_dims(data_df[c].astype(np.float32).values, -1)
    return d


# Optuna objective 함수 정의
def objective(trial):
    # 하이퍼파라미터 탐색 범위 설정 (임계값 제외)
    embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64, 128])
    depth = trial.suggest_int('depth', 2, 6)
    heads = trial.suggest_categorical('heads', [4, 6, 8])
    attn_dropout = trial.suggest_float('attn_dropout', 0.1, 0.5)
    ff_dropout = trial.suggest_float('ff_dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])

    # 임계값은 Optuna 탐색 대상에서 제거합니다.
    fixed_threshold = 0.5  # 여기서는 기본값 0.5를 임시로 사용

    # 1. 모델 초기화
    tf.keras.backend.clear_session()

    # FT-Transformer 인코더 정의
    ft_encoder = FTTransformerEncoder(
        numerical_features=continuous_vars,
        categorical_features=categorical_vars,
        numerical_data=X_train_df[continuous_vars].astype(np.float32).values,
        categorical_data=X_train_df[categorical_vars].astype(str).values,
        y=None,
        numerical_embedding_type='linear',
        embedding_dim=embedding_dim,
        depth=depth,
        heads=heads,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        explainable=False
    )

    # 모델 정의
    inputs = {}
    for col in categorical_vars:
        inputs[col] = Input(shape=(1,), dtype=tf.string, name=col)
    for col in continuous_vars:
        inputs[col] = Input(shape=(1,), dtype=tf.float32, name=col)

    transformer_output = ft_encoder(inputs)
    cls_token_output = Lambda(lambda x: x[:, 0])(transformer_output)

    # 최종 출력 레이어
    final_output = Dense(1, activation='sigmoid')(cls_token_output)
    ft_model = Model(inputs=inputs, outputs=final_output)

    optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    ft_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

    # 입력 데이터 변환 (기존과 동일)
    X_train_dict = reshape_inputs(X_train_df, categorical_vars, continuous_vars)
    X_val_dict = reshape_inputs(X_test_df, categorical_vars, continuous_vars)

    # 학습
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = ft_model.fit(
        X_train_dict, y_train,
        validation_data=(X_val_dict, y_test),
        epochs=100,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    # 평가 및 예측
    y_pred_proba = ft_model.predict(X_val_dict, verbose=0).flatten()

    # Optuna에서 임계값을 탐색하지 않으므로, 고정된 임계값 (fixed_threshold)을 사용합니다.
    y_pred = (y_pred_proba >= fixed_threshold).astype(int)

    # F1-Score 계산 (Optuna의 최적화 목표)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 다른 성능 지표들도 계산하여 user_attrs에 저장
    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0

    trial.set_user_attr("f1_score", f1)
    trial.set_user_attr("auc_score", auc_score)
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)
    trial.set_user_attr("specificity", specificity)
    trial.set_user_attr("applied_threshold", fixed_threshold)  # 사용된 임계값도 저장

    # Optuna의 최적화 목표를 F1-Score로 설정
    return f1


# Optuna Study 실행 및 결과 출력
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

print("FT-Transformer에 대한 Optuna 최적화를 시작합니다 (목표: F1-Score 최대화, 임계값 고정)...")
study.optimize(objective, n_trials=5, show_progress_bar=True)  # n_trials는 필요에 따라 조정

print("\n--- FT-Transformer Optuna 최적화 결과 ---")
print("최적 목표 값 (F1-Score):", study.best_value)
print("최적 파라미터:", study.best_params)

print("최적 trial의 F1-Score:", study.best_trial.user_attrs["f1_score"])
print("최적 trial의 AUC:", study.best_trial.user_attrs["auc_score"])
print("최적 trial의 Accuracy:", study.best_trial.user_attrs["accuracy"])
print("최적 trial의 Precision:", study.best_trial.user_attrs["precision"])
print("최적 trial의 Recall (Sensitivity):", study.best_trial.user_attrs["recall"])
print("최적 trial의 Specificity:", study.best_trial.user_attrs["specificity"])
print("최적 trial에 적용된 임계값:", study.best_trial.user_attrs["applied_threshold"])  # 적용된 임계값 출력