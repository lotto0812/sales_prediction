import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import optuna
import lightgbm as lgb
from typing import Dict, Tuple

def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20, 
                          figsize: Tuple[int, int] = (10, 8)):
    """
    特徴量重要度のプロット
    """
    plt.figure(figsize=figsize)
    top_features = feature_importance.head(top_n)
    
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(y_test: pd.Series, y_pred: np.ndarray, 
                             figsize: Tuple[int, int] = (10, 8)):
    """
    予測値vs実際値のプロット
    """
    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_pred, alpha=0.6)
    
    # 完全予測の線
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('実際の売上')
    plt.ylabel('予測売上')
    plt.title('予測値 vs 実際値')
    
    # R²スコアを表示
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        n_trials: int = 100) -> Dict:
    """
    Optunaを使ったハイパーパラメータチューニング
    """
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'verbose': -1,
            'random_seed': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50)],
            verbose_eval=False
        )
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(((y_val - y_pred) ** 2).mean())
        
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt', 
        'verbose': -1,
        'random_seed': 42
    })
    
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    
    return best_params

def cross_validate_model(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                        params: Dict = None) -> Dict[str, float]:
    """
    クロスバリデーションでモデルの性能を評価
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_seed': 42
        }
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # データセットの作成
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        # モデルの訓練
        model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50)],
            verbose_eval=False
        )
        
        # 予測と評価
        y_pred = model.predict(X_val_fold)
        
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred)))
        mae_scores.append(mean_absolute_error(y_val_fold, y_pred))
        r2_scores.append(r2_score(y_val_fold, y_pred))
    
    cv_results = {
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores)
    }
    
    return cv_results 