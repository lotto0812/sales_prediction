# 店舗売上予測システム

既存10000店舗の2024年売上データと店舗特徴量をもとに、取引のない店舗の売上を予測するLightGBMモデルです。

## 特徴量

- 都道府県 (string)
- 人流 (float)
- ウェブサイトありなし (0,1)
- 席数 (int)
- 予約可不可 (0,1)
- 住所 (string)
- 地域 (string)
- 最寄り駅 (string)
- 電話有り無し (0,1)
- 食べログスコア (float)
- 食べログスコア投票数 (int)

## プロジェクト構成

```
├── src/
│   ├── data_preprocessing.py    # データ前処理クラス
│   ├── lightgbm_model.py       # LightGBMモデルクラス
│   └── model_utils.py          # ユーティリティ関数
├── notebooks/
│   └── sales_prediction_analysis.ipynb  # 分析用ノートブック
├── data/                       # データファイル
├── models/                     # 訓練済みモデル
├── outputs/                    # 予測結果
├── main.py                     # メイン実行スクリプト
├── requirements.txt            # 必要なライブラリ
└── README.md                   # このファイル
```

## セットアップ

1. 必要なライブラリのインストール:
```bash
pip install -r requirements.txt
```

2. プロジェクトディレクトリに移動:
```bash
cd lightgbm
```

## 使用方法

### 1. サンプルデータでの実行（テスト用）

```bash
python main.py
```

### 2. 実データでの実行

```bash
python main.py --train_data data/train_data.csv --predict_data data/predict_data.csv
```

### 3. ハイパーパラメータチューニング付きで実行

```bash
python main.py --use_tuning
```

### 4. その他のオプション

```bash
python main.py --help
```

利用可能なオプション:
- `--train_data`: 訓練用データファイルのパス
- `--predict_data`: 予測対象データファイルのパス
- `--output`: 予測結果の出力ファイル名
- `--sample_size`: サンプルデータのサイズ
- `--use_tuning`: ハイパーパラメータチューニングを実行
- `--cv_folds`: クロスバリデーションのフォールド数
- `--save_model`: モデル保存先のパス

## データフォーマット

### 訓練用データ (CSV形式)
- 列名は日本語でも英語でも可
- 欠損値は自動で処理
- 文字列カテゴリカル変数は自動でエンコーディング

例:
```csv
店舗ID,都道府県,人流,ウェブサイトありなし,席数,予約可不可,住所,地域,最寄り駅,電話有り無し,食べログスコア,食べログスコア投票数,売上
1,東京都,5234.5,1,50,1,東京都新宿区...,都心部,新宿駅,1,3.8,150,450000
```

### 予測対象データ (CSV形式)
- 訓練用データと同じ特徴量列（売上列は不要）

## 出力結果

- 予測結果CSV: `outputs/predictions.csv`
- 訓練済みモデル: `models/sales_prediction_model.pkl`
- 評価レポート: コンソール出力

## 分析ノートブック

詳細な分析は `notebooks/sales_prediction_analysis.ipynb` で実行できます:

```bash
jupyter notebook notebooks/sales_prediction_analysis.ipynb
```

## モデルの特徴

- **アルゴリズム**: LightGBM (Gradient Boosting)
- **前処理**: Target Encoding, 特徴量エンジニアリング
- **評価指標**: RMSE, MAE, R², MAPE
- **クロスバリデーション**: 5-fold CV
- **ハイパーパラメータチューニング**: Optuna (オプション)

## パフォーマンス

サンプルデータ（10,000店舗）での性能例:
- RMSE: ~45,000円
- R²: ~0.85
- 処理時間: 約1-2分

## トラブルシューティング

### よくある問題

1. **日本語文字化け**: データファイルのエンコーディングを確認
2. **メモリ不足**: サンプルサイズを小さくする (`--sample_size 1000`)
3. **ライブラリエラー**: `pip install -r requirements.txt` を再実行

## ライセンス

MIT License
