import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, recall_score, f1_score, fbeta_score
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# 하이퍼파라미터 튜닝 시 조용히 넘어가기 위해 로깅 비활성화
optuna.logging.set_verbosity(optuna.logging.WARNING)

def tune_hyperparameters(X, y_reg, y_cls, n_trials=30, random_state=42):
    """
    TimeSeriesSplit을 사용하여 회귀와 분류 모델의 최적 파라미터를 찾습니다.
    (현재는 가장 강력한 3대장인 LightGBM, XGBoost, RandomForest만 튜닝하도록 짰습니다)
    """
    # 5개의 Fold로 나누어 미래 데이터를 예측하게 만듦
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 1. 회귀 모델 (목표: RMSE 최소화)
    def reg_objective(trial):
        # 모델 선택 (LightGBM, XGBoost, RF 중 택 1)
        reg_model_type = trial.suggest_categorical('regressor', ['LightGBM', 'XGBoost', 'RandomForest'])
        
        if reg_model_type == 'LightGBM':
            params = {
                'objective': 'regression',
                'random_state': random_state,
                'n_estimators': trial.suggest_int('n_estimators', 50, 150), # 트리 개수 제한
                'max_depth': trial.suggest_int('max_depth', 3, 6),          # 얕은 트리 강제
                'num_leaves': trial.suggest_int('num_leaves', 7, 31),       # 가지치기 억제
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50), # 말단 노드 최소 데이터 수
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),    # 데이터 60~90%만 샘플링
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9), # 피처 60~90%만 샘플링
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),   # L1 페널티
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),  # L2 페널티
                'verbose': -1
            }
            model = LGBMRegressor(**params)
            
        elif reg_model_type == 'XGBoost':
            params = {
                'objective': 'reg:squarederror',
                'random_state': random_state,
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True)}
            model = XGBRegressor(**params)
            
        else: # RandomForest
            params = {
                'random_state': random_state,
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']) # 전체 피처 사용 금지
            }
            model = RandomForestRegressor(**params)
            
        # TimeSeriesSplit 교차 검증 수행
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_scores.append(rmse)
            
        return np.mean(cv_scores) # 평균 RMSE 반환
    
    # 2. 분류 모델 (목표: Recall 최대화)
    def cls_objective(trial):
        cls_model_type = trial.suggest_categorical('classifier', ['LightGBM', 'XGBoost', 'RandomForest'])
        scale_pos_weight = trial.suggest_float('scale_pos_weight', 3.0, 10.0) # 불균형 데이터 가중치 (고정)
        
        if cls_model_type == 'LightGBM':
            params = {
                'objective': 'binary',
                'random_state': random_state,
                'scale_pos_weight': scale_pos_weight,
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'num_leaves': trial.suggest_int('num_leaves', 7, 31),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'verbose': -1
            }
            model = LGBMClassifier(**params)
            
        elif cls_model_type == 'XGBoost':
            params = {
                'objective': 'binary:logistic',
                'random_state': random_state,
                'scale_pos_weight': scale_pos_weight,
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True)
            }
            model = XGBClassifier(**params)
            
        else: # RandomForest
            params = {
                'random_state': random_state,
                'class_weight': {0: 1, 1: scale_pos_weight},
                'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'min_samples_split': trial.suggest_int('min_samples_split', 10, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestClassifier(**params)
            
        # TimeSeriesSplit 교차 검증 수행
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
            
            # 주의: 데이터가 너무 적은 fold에서 클래스 1이 없으면 에러가 날 수 있음
            if y_tr.sum() == 0 or y_val.sum() == 0:
                continue 
                
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            # 분류는 조류 경보를 놓치지 않는 것(Recall)이 최우선 목표
            score = recall_score(y_val, preds, zero_division=0)
            cv_scores.append(score)
            
        # 정상적으로 계산된 Fold가 없으면 0 반환
        return np.mean(cv_scores) if cv_scores else 0

    # --- Optuna 스터디 실행 ---
    print("  [Optuna] 회귀 모델 최적화 중 (RMSE 최소화)...")
    reg_study = optuna.create_study(direction="minimize")
    reg_study.optimize(reg_objective, n_trials=n_trials)
    
    print("  [Optuna] 분류 모델 최적화 중 (Recall 최대화)...")
    cls_study = optuna.create_study(direction="maximize")
    cls_study.optimize(cls_objective, n_trials=n_trials)
    
    return reg_study.best_params, cls_study.best_params