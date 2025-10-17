
# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import AdamW
from tabtransformertf.models.fttransformer import FTTransformer, FTTransformerEncoder
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




# 데이터 로드
X_train_df = pd.read_csv("X_train.csv").astype(np.float32)
X_test_df = pd.read_csv("X_test.csv").astype(np.float32)
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# 범주형 변수 정의 (X_train_df의 컬럼 존재 여부를 확인합니다)
categorical_vars = [col for col in [
    'liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal', 'Immunocompromised',
    'arf', 'Sex', 'mv', 'vaso', 'AF_hos'
] if col in X_train_df.columns]

# 연속형 변수 정의 (X_train_df의 컬럼 존재 여부를 확인합니다)
continuous_vars = [col for col in [
    'Age', 'Cr', 'Na', 'k', 'hct', 'wbc', 'pH', 'Temp', 'hr', 'rr',
    'mbp', 'PaO2/FiO2', 'gcs', 'Urine output', 'Plt', 'Bil', 'lactate',
    'APACHE II score', 'sofa', 'pco2', 'A-a gradient', 'FiO2', 'bmi'
] if col in X_train_df.columns]

# Optuna를 통해 찾은 최적 파라미터 (임계값 제외)
best_params = {
    'embedding_dim': 16,
    'depth': 2,
    'heads': 4,
    'attn_dropout': 0.4579309401710595,
    'ff_dropout': 0.3391599915244341,
    'learning_rate': 0.0069782812651260325,
    'weight_decay': 1.5030900645056814e-06,
    'batch_size': 2048
}

# 4. 입력 정의
inputs = {}
for col in categorical_vars:
    inputs[col] = Input(shape=(1,), dtype=tf.string, name=col)
for col in continuous_vars:
    inputs[col] = Input(shape=(1,), dtype=tf.float32, name=col)

# 5. FT-Transformer 인코더 정의 (최적 파라미터 적용)
ft_encoder = FTTransformerEncoder(
    numerical_features=continuous_vars,
    categorical_features=categorical_vars,
    numerical_data=X_train_df[continuous_vars].astype(np.float32).values,
    categorical_data=X_train_df[categorical_vars].astype(str).values,
    y=None,
    numerical_embedding_type='linear',
    embedding_dim=best_params['embedding_dim'],
    depth=best_params['depth'],
    heads=best_params['heads'],
    attn_dropout=best_params['attn_dropout'],
    ff_dropout=best_params['ff_dropout'],
    explainable=False
)

# 6. 모델 정의
transformer_output = ft_encoder(inputs)
cls_token_output = Lambda(lambda x: x[:, 0])(transformer_output)

final_output = Dense(1, activation='sigmoid')(cls_token_output)
ft_model = Model(inputs=inputs, outputs=final_output)

# 옵티마이저 정의 (최적 파라미터 적용)
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=best_params['learning_rate'],
    weight_decay=best_params['weight_decay']
)

ft_model.compile(optimizer=optimizer, loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.AUC(name='AUC')])

# 7. 입력 변환 함수
def reshape_inputs(data_df, cat_cols, num_cols):
    d = {}
    for c in cat_cols:
        d[c] = np.expand_dims(data_df[c].astype(str).values, -1)
    for c in num_cols:
        d[c] = np.expand_dims(data_df[c].astype(np.float32).values, -1)
    return d

# X_train_dict, X_val_dict 정의
X_train_dict = reshape_inputs(X_train_df, categorical_vars, continuous_vars)
X_val_dict = reshape_inputs(X_test_df, categorical_vars, continuous_vars)

# 8. 학습
print("FT-Transformer 모델 학습을 시작합니다 (Optuna 최적 파라미터 적용)...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = ft_model.fit(
    X_train_dict, y_train,
    validation_data=(X_val_dict, y_test),
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stopping],
    verbose=1
)
print("FT-Transformer 모델 학습 완료.")

# 9. 평가 및 예측
y_pred_proba = ft_model.predict(X_val_dict).flatten()

# --- F1-Score를 최대로 하는 임계값 동적 탐색 ---
thresholds = np.linspace(0, 1, 100)  # 0부터 1까지 100개의 임계값 후보 생성
best_f1 = 0
optimal_threshold = 0.5  # 초기 최적 임계값

for threshold in thresholds:
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    current_f1 = f1_score(y_test, y_pred_temp, zero_division=0)
    if current_f1 > best_f1:
        best_f1 = current_f1
        optimal_threshold = threshold

print(f"\n학습된 모델에서 F1-Score를 최대로 하는 임계값: {optimal_threshold:.4f} (해당 F1-Score: {best_f1:.4f})")
# --- 임계값 동적 탐색 종료 ---

# 찾은 최적 임계값을 사용하여 최종 이진 예측 수행
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# 지표 계산
loss, acc, auc_score_eval = ft_model.evaluate(X_val_dict, y_test, verbose=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0) # 이 F1-스코어는 위에서 찾은 best_f1과 동일해야 함
auc_score_calc = roc_auc_score(y_test, y_pred_proba)

# 특이도(Specificity) 계산
cm = confusion_matrix(y_test, y_pred)
TN = cm[0, 0] # True Negative
FP = cm[0, 1] # False Positive
specificity = TN / (TN + FP) if (TN + FP) != 0 else 0


print(f"\n--- FT-Transformer 모델 성능 (Optuna 최적 파라미터 및 동적 임계값 {optimal_threshold:.4f} 적용) ---")
print(f"Loss: {loss:.4f}, AUC (Keras Eval): {auc_score_eval:.4f}, Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}") # 특이도 출력 추가
print(f"F1 Score: {f1:.4f}")
print(f"AUC (Sklearn Calc): {auc_score_calc:.4f}")

# 10. 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix - FT-Transformer (Threshold: {optimal_threshold:.4f})")
plt.show()

# 11. ROC 곡선
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - FT-Transformer')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()