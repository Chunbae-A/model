from typing import Any
from src.config import REGRESSION_MODEL_CANDIDATES, CLASSIFICATION_MODEL_CANDIDATES
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

def _expand_model_candidates(candidates: list[str]) -> list[str]:
    if "auto" in candidates:
        return ["lightgbm", "xgboost", "hist_gradient_boosting", "randomforest", "catboost"]
    return candidates


def build_regression_model_candidates(random_state: int = 42, optuna_best_params_reg: dict = None) -> dict[str, Any]:
    candidates = {}
    requested = _expand_model_candidates(REGRESSION_MODEL_CANDIDATES)

    if "lightgbm" in requested:
        try:
            from lightgbm import LGBMRegressor
            lgbm_args = {'objective': 'regression', 'random_state': random_state, 'verbose': -1}
            if optuna_best_params_reg and optuna_best_params_reg.get('regressor') == 'LightGBM':
                best_args = {k.replace('lgbm_reg_', ''): v for k, v in optuna_best_params_reg.items() if k.startswith('lgbm_reg_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr') 
                lgbm_args.update(best_args)
            candidates["lightgbm"] = LGBMRegressor(**lgbm_args)
        except ImportError: pass

    if "xgboost" in requested:
        try:
            from xgboost import XGBRegressor
            xgb_args = {'objective': 'reg:squarederror', 'random_state': random_state}
            if optuna_best_params_reg and optuna_best_params_reg.get('regressor') == 'XGBoost':
                best_args = {k.replace('xgb_reg_', ''): v for k, v in optuna_best_params_reg.items() if k.startswith('xgb_reg_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr') 
                xgb_args.update(best_args)
            candidates["xgboost"] = XGBRegressor(**xgb_args)
        except ImportError: pass

    if "hist_gradient_boosting" in requested:
        candidates["hist_gradient_boosting"] = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=500, random_state=random_state)

    if "randomforest" in requested:
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf_args = {'random_state': random_state}
            if optuna_best_params_reg and optuna_best_params_reg.get('regressor') == 'RandomForest':
                best_args = {k.replace('rf_reg_', ''): v for k, v in optuna_best_params_reg.items() if k.startswith('rf_reg_')}
                rf_args.update(best_args)
            candidates["randomforest"] = RandomForestRegressor(**rf_args)
        except ImportError: pass

    if "catboost" in requested:
        try:
            from catboost import CatBoostRegressor
            # verbose=False를 안 주면 터미널에 로그가 미친 듯이 도배되므로 필수 추가!
            cat_args = {'random_state': random_state, 'verbose': False}
            if optuna_best_params_reg and optuna_best_params_reg.get('regressor') == 'CatBoost':
                best_args = {k.replace('cat_reg_', ''): v for k, v in optuna_best_params_reg.items() if k.startswith('cat_reg_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr')
                cat_args.update(best_args)
            candidates["catboost"] = CatBoostRegressor(**cat_args)
        except ImportError: pass

    return candidates


def build_classification_model_candidates(random_state: int = 42, optuna_best_params: dict = None) -> dict[str, Any]:
    candidates = {}
    requested = _expand_model_candidates(CLASSIFICATION_MODEL_CANDIDATES)
    
    default_scale_pos_weight = 4.6 

    if "lightgbm" in requested:
        try:
            from lightgbm import LGBMClassifier
            lgbm_args = {'objective': 'binary', 'random_state': random_state, 'verbose': -1}
            lgbm_args['scale_pos_weight'] = default_scale_pos_weight 
            
            if optuna_best_params and optuna_best_params.get('classifier') == 'LightGBM':
                best_args = {k.replace('lgbm_', ''): v for k, v in optuna_best_params.items() if k.startswith('lgbm_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr')
                lgbm_args.update(best_args) 
            candidates["lightgbm"] = LGBMClassifier(**lgbm_args)
        except ImportError: pass

    if "xgboost" in requested:
        try:
            from xgboost import XGBClassifier
            xgb_args = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': random_state}
            xgb_args['scale_pos_weight'] = default_scale_pos_weight 
            
            if optuna_best_params and optuna_best_params.get('classifier') == 'XGBoost':
                best_args = {k.replace('xgb_', ''): v for k, v in optuna_best_params.items() if k.startswith('xgb_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr')
                xgb_args.update(best_args) 
            candidates["xgboost"] = XGBClassifier(**xgb_args)
        except ImportError: pass

    if "hist_gradient_boosting" in requested:
        candidates["hist_gradient_boosting"] = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=500, random_state=random_state)

    if "randomforest" in requested:
        try:
            from sklearn.ensemble import RandomForestClassifier
            # RF는 scale_pos_weight 인자가 없고 class_weight를 씁니다. 동일한 비율(1 : 4.6)을 딕셔너리로 넘겨줌.
            rf_args = {'random_state': random_state, 'class_weight': {0: 1, 1: default_scale_pos_weight}}
            if optuna_best_params and optuna_best_params.get('classifier') == 'RandomForest':
                best_args = {k.replace('rf_', ''): v for k, v in optuna_best_params.items() if k.startswith('rf_')}
                rf_args.update(best_args)
            candidates["randomforest"] = RandomForestClassifier(**rf_args)
        except ImportError: pass

    if "catboost" in requested:
        try:
            from catboost import CatBoostClassifier
            # CatBoost는 scale_pos_weight를 지원합니다.
            cat_args = {'random_state': random_state, 'verbose': False, 'scale_pos_weight': default_scale_pos_weight}
            if optuna_best_params and optuna_best_params.get('classifier') == 'CatBoost':
                best_args = {k.replace('cat_', ''): v for k, v in optuna_best_params.items() if k.startswith('cat_')}
                if 'lr' in best_args: best_args['learning_rate'] = best_args.pop('lr')
                cat_args.update(best_args)
            candidates["catboost"] = CatBoostClassifier(**cat_args)
        except ImportError: pass
    
    return candidates