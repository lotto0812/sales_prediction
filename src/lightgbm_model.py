import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from typing import Dict, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

class SalesPredictor:
    """
    LightGBMを使った売上予測モデル
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初期化
        """
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
        # デフォルトパラメータ
        self.default_params = {
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
        
        if params:
            self.default_params.update(params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              num_boost_round: int = 1000, early_stopping_rounds: int = 100,
              verbose_eval: int = 100) -> lgb.Booster:
        """
        モデルの訓練
        """
        # データセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # モデルの訓練
        self.model = lgb.train(
            params=self.default_params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(verbose_eval)
            ]
        )
        
        # 特徴量重要度の取得
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def predict(self, X: pd.DataFrame, num_iteration: int = None) -> np.ndarray:
        """
        予測の実行
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません。先にtrainメソッドを実行してください。")
        
        return self.model.predict(X, num_iteration=num_iteration)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        モデルの評価
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8), 
                               use_plotly: bool = False):
        """
        特徴量重要度のプロット
        """
        if self.feature_importance is None:
            raise ValueError("特徴量重要度が計算されていません。先にモデルを訓練してください。")
        
        if use_plotly:
            try:
                from plotly_visualization import PlotlyVisualizer
                visualizer = PlotlyVisualizer()
                fig = visualizer.plot_feature_importance(self.feature_importance, top_n)
                fig.show()
                return fig
            except ImportError:
                print("plotly_visualizationモジュールが見つかりません。matplotlibを使用します。")
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance (Gain)')
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_vs_actual(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                 figsize: Tuple[int, int] = (10, 8), use_plotly: bool = False):
        """
        予測値vs実際値のプロット
        """
        y_pred = self.predict(X_test)
        
        if use_plotly:
            try:
                from plotly_visualization import PlotlyVisualizer
                visualizer = PlotlyVisualizer()
                fig = visualizer.plot_prediction_vs_actual(y_test, y_pred)
                fig.show()
                return fig
            except ImportError:
                print("plotly_visualizationモジュールが見つかりません。matplotlibを使用します。")
        
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
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
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
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt', 
            'verbose': -1,
            'random_seed': 42
        })
        
        print(f"Best RMSE: {study.best_value:.4f}")
        print(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def save_model(self, filepath: str):
        """
        モデルの保存
        """
        if self.model is None:
            raise ValueError("保存するモデルがありません。")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'best_params': self.best_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"モデルを {filepath} に保存しました。")
    
    def load_model(self, filepath: str):
        """
        モデルの読み込み
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        self.best_params = model_data.get('best_params')
        
        print(f"モデルを {filepath} から読み込みました。")

def cross_validate_model(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5,
                        params: Optional[Dict] = None) -> Dict[str, float]:
    """
    クロスバリデーションでモデルの性能を評価
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{cv_folds}")
        
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # モデルの訓練
        predictor = SalesPredictor(params)
        predictor.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                       verbose_eval=0)
        
        # 評価
        metrics = predictor.evaluate(X_val_fold, y_val_fold)
        rmse_scores.append(metrics['RMSE'])
        mae_scores.append(metrics['MAE'])
        r2_scores.append(metrics['R2'])
    
    cv_results = {
        'RMSE_mean': np.mean(rmse_scores),
        'RMSE_std': np.std(rmse_scores),
        'MAE_mean': np.mean(mae_scores),
        'MAE_std': np.std(mae_scores),
        'R2_mean': np.mean(r2_scores),
        'R2_std': np.std(r2_scores)
    }
    
    return cv_results

if __name__ == "__main__":
    # テスト用のサンプルデータ作成
    from data_preprocessing import StoreDataPreprocessor
    
    preprocessor = StoreDataPreprocessor()
    sample_df = preprocessor.create_sample_data(1000)
    processed_df = preprocessor.preprocess_features(sample_df, is_training=True)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_df)
    
    # モデルの訓練とテスト
    predictor = SalesPredictor()
    predictor.train(X_train, y_train, X_test, y_test)
    
    # 評価
    metrics = predictor.evaluate(X_test, y_test)
    print("評価結果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 