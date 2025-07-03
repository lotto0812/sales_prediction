#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
店舗売上予測システム - メイン実行スクリプト

既存店舗の売上データと特徴量を使って、
取引のない店舗の売上を予測するLightGBMモデル
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from src.data_preprocessing import StoreDataPreprocessor
from src.lightgbm_model import SalesPredictor, cross_validate_model

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

def load_real_data(train_path: str, predict_path: str = None):
    """
    実際のデータファイルを読み込み
    
    Args:
        train_path: 訓練用データ（既存店舗の売上データ）のパス
        predict_path: 予測対象データ（取引のない店舗データ）のパス
    """
    preprocessor = StoreDataPreprocessor()
    
    # 訓練用データの読み込み
    print(f"訓練用データを読み込み中: {train_path}")
    train_df = preprocessor.load_data(train_path)
    print(f"訓練用データの形状: {train_df.shape}")
    
    # 予測対象データの読み込み（オプション）
    predict_df = None
    if predict_path:
        print(f"予測対象データを読み込み中: {predict_path}")
        predict_df = preprocessor.load_data(predict_path)
        print(f"予測対象データの形状: {predict_df.shape}")
    
    return train_df, predict_df, preprocessor

def create_sample_data(n_samples: int = 10000):
    """
    サンプルデータを作成（テスト用）
    """
    print(f"サンプルデータを作成中 (n={n_samples})")
    preprocessor = StoreDataPreprocessor()
    sample_df = preprocessor.create_sample_data(n_samples)
    
    return sample_df, None, preprocessor

def train_and_evaluate_model(train_df: pd.DataFrame, preprocessor: StoreDataPreprocessor,
                           use_tuning: bool = False, cv_folds: int = 5):
    """
    モデルの訓練と評価を実行
    """
    print("\n=== データ前処理 ===")
    processed_df = preprocessor.preprocess_features(train_df, is_training=True)
    print(f"前処理後のデータ形状: {processed_df.shape}")
    print(f"特徴量数: {len(preprocessor.feature_columns)}")
    
    # データ分割
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_df)
    print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}")
    
    print("\n=== クロスバリデーション ===")
    X_full = processed_df.drop(['売上', '店舗ID'], axis=1, errors='ignore')
    y_full = processed_df['売上']
    
    cv_results = cross_validate_model(X_full, y_full, cv_folds=cv_folds)
    print("クロスバリデーション結果:")
    for metric, value in cv_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # モデルの初期化
    predictor = SalesPredictor()
    
    # ハイパーパラメータチューニング（オプション）
    if use_tuning:
        print("\n=== ハイパーパラメータチューニング ===")
        # さらに検証用データを分割
        X_train_tune, X_val_tune, y_train_tune, y_val_tune = preprocessor.split_data(
            pd.concat([X_train, y_train], axis=1).assign(売上=y_train)
        )
        
        best_params = predictor.hyperparameter_tuning(
            X_train_tune, y_train_tune, X_val_tune, y_val_tune, n_trials=50
        )
        predictor.default_params = best_params
    
    print("\n=== モデル訓練 ===")
    # 最終モデルの訓練
    predictor.train(X_train, y_train, X_test, y_test)
    
    print("\n=== モデル評価 ===")
    metrics = predictor.evaluate(X_test, y_test)
    print("テストデータでの評価結果:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 可視化
    try:
        print("\n=== 結果の可視化 ===")
        # Plotly Express版を試す
        try:
            from src.plotly_visualization import create_quick_visualization
            create_quick_visualization(predictor.feature_importance, y_test, predictor.predict(X_test))
            print("Plotly Expressを使用したインタラクティブなグラフを作成しました。")
            print("outputs/dashboard.html をブラウザで開いてダッシュボードを確認できます。")
        except ImportError:
            print("Plotlyが利用できません。matplotlibを使用します。")
            predictor.plot_feature_importance(top_n=15)
            predictor.plot_prediction_vs_actual(X_test, y_test)
    except Exception as e:
        print(f"可視化でエラーが発生しました: {e}")
    
    return predictor, processed_df

def make_predictions(predictor: SalesPredictor, predict_df: pd.DataFrame,
                    preprocessor: StoreDataPreprocessor, output_path: str = None):
    """
    予測の実行と結果保存
    """
    if predict_df is None:
        print("予測対象データがありません。")
        return
    
    print("\n=== 予測の実行 ===")
    
    # 予測対象データの前処理
    predict_processed = preprocessor.preprocess_features(predict_df.copy(), is_training=False)
    print(f"予測対象データの前処理完了: {predict_processed.shape}")
    
    # 予測実行
    predictions = predictor.predict(predict_processed)
    
    # 結果の整理
    result_df = predict_df.copy()
    result_df['予測売上'] = predictions
    
    # 結果の表示
    print("\n予測結果の統計:")
    print(f"  平均予測売上: {predictions.mean():,.0f}円")
    print(f"  売上の中央値: {np.median(predictions):,.0f}円")
    print(f"  売上の標準偏差: {predictions.std():,.0f}円")
    print(f"  最小予測売上: {predictions.min():,.0f}円")
    print(f"  最大予測売上: {predictions.max():,.0f}円")
    
    # CSVファイルに保存
    if output_path:
        result_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n予測結果を保存しました: {output_path}")
    
    return result_df

def main():
    """
    メイン処理
    """
    parser = argparse.ArgumentParser(description='店舗売上予測システム')
    parser.add_argument('--train_data', type=str, help='訓練用データファイルのパス')
    parser.add_argument('--predict_data', type=str, help='予測対象データファイルのパス')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='予測結果の出力ファイル名')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='サンプルデータのサイズ（実データがない場合）')
    parser.add_argument('--use_tuning', action='store_true', 
                       help='ハイパーパラメータチューニングを実行')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='クロスバリデーションのフォールド数')
    parser.add_argument('--save_model', type=str, help='モデル保存先のパス')
    
    args = parser.parse_args()
    
    # データディレクトリの作成
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # データの準備
        if args.train_data:
            # 実データを使用
            train_df, predict_df, preprocessor = load_real_data(
                args.train_data, args.predict_data
            )
        else:
            # サンプルデータを作成
            print("実データが指定されていません。サンプルデータを使用します。")
            train_df, predict_df, preprocessor = create_sample_data(args.sample_size)
            
            # サンプルデータを保存
            train_df.to_csv('data/sample_train_data.csv', index=False, encoding='utf-8')
            print("サンプル訓練データを data/sample_train_data.csv に保存しました。")
        
        # データの基本情報表示
        print("\n=== データ概要 ===")
        print("訓練データの列:")
        for col in train_df.columns:
            print(f"  {col}: {train_df[col].dtype}")
        
        print(f"\n欠損値の確認:")
        missing_info = train_df.isnull().sum()
        if missing_info.sum() > 0:
            print(missing_info[missing_info > 0])
        else:
            print("  欠損値はありません")
        
        # モデルの訓練と評価
        predictor, processed_df = train_and_evaluate_model(
            train_df, preprocessor, args.use_tuning, args.cv_folds
        )
        
        # モデルの保存
        if args.save_model:
            predictor.save_model(args.save_model)
        else:
            predictor.save_model('models/sales_prediction_model.pkl')
        
        # 予測の実行（予測対象データがある場合）
        if predict_df is not None:
            predictions_df = make_predictions(
                predictor, predict_df, preprocessor, 
                f'outputs/{args.output}'
            )
        else:
            # サンプルの予測対象データを作成して予測
            print("\n予測対象データがないため、サンプルデータを作成して予測を実行します。")
            sample_predict_df = preprocessor.create_sample_data(100)
            sample_predict_df = sample_predict_df.drop('売上', axis=1)  # 売上列を削除
            
            predictions_df = make_predictions(
                predictor, sample_predict_df, preprocessor,
                f'outputs/sample_{args.output}'
            )
        
        print("\n=== 処理完了 ===")
        print("売上予測モデルの訓練と予測が完了しました。")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 