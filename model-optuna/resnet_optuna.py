import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Add
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf

# 데이터 로드
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()

# ResNet Block 정의
def residual_block(x, units, projection=False):
    shortcut = x
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    if projection or shortcut.shape[-1] != units:
        shortcut = Dense(units)(shortcut)
    x = Add()([x, shortcut])
    x = ReLU()(x)
    return x

# 모델 정의
def build_model(input_dim, units, learning_rate):
    inputs = Input(shape=(input_dim,))
    x = Dense(units)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, units, projection=True)
    x = residual_block(x, units)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['AUC'])
    return model

def objective(trial):
    # 시드 고정
    seed = 42
    import os, random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 파라미터 서치 공간
    units = trial.suggest_categorical('units', [64, 128, 256])
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 30, 100)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, f1s = [], []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = build_model(X_tr.shape[1], units, learning_rate)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )

        y_pred_prob = model.predict(X_val).flatten()

        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1 = np.max(f1_scores)
        auc = roc_auc_score(y_val, y_pred_prob)

        aucs.append(auc)
        f1s.append(best_f1)

    return 0.5 * np.mean(aucs) + 0.5 * np.mean(f1s)



# Optuna 튜닝 시작
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 결과 출력
print("Best trial:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
