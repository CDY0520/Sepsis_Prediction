# 중환자실 임상 데이터를 활용한 근거기반 사망 위험 조기 예측 분석
기간: 2025. 06. 05 ~ 2025. 06. 26 (총 3주)
팀 구성: 5명
담당 역할(개인 기여 기준): 데이터 수집·전처리(50%), 임상 변수 조사(80%), 분류 모델 적용 및 변수 중요도 분석(80%)

---

# 주요목표
중환자실 임상 변수를 기반으로 사망 위험과 연관된 핵심 요인을 분석하여 의료 현장에서 설명 가능한 조기 사망 예측 근거 도출

---

# 핵심기여
선행 논문 및 임상 가이드라인 기반 사망 위험 관련 임상 변수 조사
분류 모델에 적용하여 변수 중요도 및 영향 방향 분석
SHAP 분석을 활용한 예측 결과 해석 및 임상적 설명 정리

---

# 배경 및 문제 정의
*AI 예측 성능보다, 예측에 사용되는 임상 데이터의 설계가 실제 활용성을 결정*

AI 예측 접근의 한계:
단편적 임상 수치 기반 예측은 해석·설명에 한계
변수 의미가 불명확하면 의사결정에 활용 어려움

현장 적용 문제:
조기 사망 예측은 수치 자체보다 맥락과 변화가 중요
근거 없는 변수 선택은 운영 단계 오류 위험 증가

프로젝트 접근 방향:
선행연구 기반 임상적 의미가 검증된 변수 우선 정의
성능 중심이 아닌 현장 설명 가능성 중심의 변수 분석

---

# 주요 성과

패혈증 조기 사망 예측에서 SVM AUC 0.75로 기본 운영 가능 수준 성능 확보 → 일반 운영 기준(AUC ≥ 0.7)에 근접한 수준으로 평가
선행연구·임상 가이드라인 기반으로 조기 사망 예측에 필요한 핵심 변수 체계 정리

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
