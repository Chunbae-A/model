import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

def train_stacking_models(X_train, y_reg_train, y_cls_train, random_state=42):
    """
    3대장(LGBM, XGB, RF)을 Base 모델로 사용하고,
    선형 모델(Ridge, LogisticRegression)을 최종 Meta 모델로 사용하는 스태킹 앙상블.
    """
    print("🛠️ [스태킹 앙상블] Base 모델 및 Meta 모델 조립 중...")

    # ==========================================
    # 1. 회귀(Regression) 스태킹 앙상블
    # ==========================================
    reg_base_models = [
        ('lgbm', LGBMRegressor(random_state=random_state)),
        ('xgb', XGBRegressor(random_state=random_state)),
        ('rf', RandomForestRegressor(random_state=random_state))
    ]
    
    # 회귀 최종 결정권자: Ridge (과적합을 방지하는 안정적인 선형 모델)
    reg_meta_model = Ridge()

    stacking_regressor = StackingRegressor(
        estimators=reg_base_models,
        final_estimator=reg_meta_model,
        cv=5, # 5-Fold로 교차 검증하며 메타 데이터 생성
        n_jobs=-1
    )

    # ==========================================
    # 2. 분류(Classification) 스태킹 앙상블
    # ==========================================
    # 분류는 조류 경보(Recall)를 놓치지 않도록 class_weight를 강하게 줍니다!
    scale_weight = 5.0 
    
    cls_base_models = [
        ('lgbm', LGBMClassifier(random_state=random_state, class_weight={0: 1, 1: scale_weight})),
        ('xgb', XGBClassifier(random_state=random_state, scale_pos_weight=scale_weight)),
        ('rf', RandomForestClassifier(random_state=random_state, class_weight={0: 1, 1: scale_weight}))
    ]
    
    # 분류 최종 결정권자: 로지스틱 회귀 (가장 깔끔하게 확률을 종합해 줌)
    cls_meta_model = LogisticRegression(class_weight='balanced', random_state=random_state)

    stacking_classifier = StackingClassifier(
        estimators=cls_base_models,
        final_estimator=cls_meta_model,
        cv=5,
        n_jobs=-1
    )

    # ==========================================
    # 3. 모델 학습 (Fit)
    # ==========================================
    print("🚀 [스태킹 회귀] 앙상블 모델 학습 시작 (시간이 조금 걸릴 수 있습니다)...")
    stacking_regressor.fit(X_train, y_reg_train)
    
    print("🚀 [스태킹 분류] 앙상블 모델 학습 시작...")
    stacking_classifier.fit(X_train, y_cls_train)
    
    print("✅ 스태킹 앙상블 모델 학습 완료!")
    
    return stacking_regressor, stacking_classifier