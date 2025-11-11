# ICU Sepsis Mortality Early Prediction
중환자실(ICU) 패혈증 환자의 임상 정보로 사망 조기 예측 모델 개발

프로젝트 개요
목표: 입실 초기 임상 변수로 사망 위험을 조기 예측하여 임상의 의사결정을 지원
데이터: ICU 패혈증 환자 전자의무기록(EMR) 기반 표형(tabular) 변수. 학습/검증/시험 분할 완료 데이터 사용
접근: 표형데이터에 강한 모델군을 폭넓게 비교 → Optuna로 하이퍼파라미터 탐색 → 통계적 검정으로 우월성 확인

---

# 폴더 구조

```
ICU_Sepsis_EarlyPred/
├─ data/
│  └─ ICU_Clinical_split_data.zip         # 분할된 입력 데이터(학습/검증/시험)
├─ model-optuna/                           # 모델별 Optuna 탐색 스크립트
│  ├─ FTT_optuna.py                        # FT-Transformer
│  ├─ LGBM_optuna.py                       # LightGBM
│  ├─ RF_optuna.py                         # Random Forest
│  ├─ SVM_optuna.py                        # SVM
│  ├─ TabNet_optuna.py                     # TabNet
│  ├─ XGBoost_optuna.py                    # XGBoost
│  └─ resnet_optuna.py                     # (이미지형 미사용 시 보조 실험)
├─ tuning/                                 # 최종 튜닝 적용 스크립트(재현용)
│  ├─ Catboost_last.py
│  ├─ FTT_tuning.py
│  ├─ LGBM_tuning.py
│  ├─ RF_tuning.py
│  ├─ SVM_last.py
│  ├─ TabNet_tuning.py
│  ├─ TabPFN_tuning.py
│  └─ XGBoost_tuning.py
└─ README.md
```

---

# 관련 연구 정리 요약

표형/임상 데이터에서 XGBoost, LightGBM, CatBoost, RandomForest, 그리고 TabNet·FT-Transformer·TabPFN이 주로 강세
본 연구는 선행연구에서 성능이 우수했던 위 모델군을 동일 데이터 분할에서 공정 비교

---

# 전처리 파이프라인(요지)

결측/이상치 처리: 임상적 비현실 구간 제거, Null 대체 전략 적용
변수 엔지니어링: BMI, 소변량(uo), AF 과거력 등 파생·변환
중복/충돌 정리: 중복 표본/충돌 지표 제거, 수집 시점 정합성 확인
스케일링: 연속형은 표준화 또는 IQR기반 스케일링
레이블: 입실 초기 정보 기반의 사망(이진)

---

# 특징 선택(Feature Selection)

SHAP 중요도로 임상적 기여도 높은 변수 우선순위 파악
VIF로 다중공선성 점검 → 공선성 높거나 중복·파생 중복 변수 제거

---

# 모델 비교·평가 설계

지표: AUC(주), 민감도(Sens), 특이도(Spec), 정밀도(PPV), 음성예측도(NPV), 균형정확도

통계 검정
DeLong test: AUC 차이 유의성 검정
McNemar / Q-statistics: 모델 간 오분류 상관 및 상보성 검토
재현성: 고정 시드, 동일 분할(hold-out 또는 CV), 동일 전처리

---

# 주요 결과(핵심 포인트)

DeLong 기준 SVM vs TabPFN 유사 성능 구간 존재
Q-statistics·McNemar 결과, 모델 간 상보성은 제한적
CatBoost가 AUC, 민감도, 균형정확도 전반에서 최상 → 임상 적용 가능성 가장 높음

최종 리더보드: CatBoost ≥ (XGBoost ≈ LGBM) ≥ RF ≈ FT-Transformer ≥ TabNet ≈ SVM ≥ TabPFN
(데이터/스플릿에 따라 변동 여지. 상세 수치는 실험 결과 표에 기재 권장.)
