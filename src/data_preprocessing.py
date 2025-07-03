import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from .geo_features import GeoFeatureEngineer
    GEO_FEATURES_AVAILABLE = True
except ImportError:
    GEO_FEATURES_AVAILABLE = False
    print("地理的特徴量エンジニアリング機能が利用できません。")

class StoreDataPreprocessor:
    """
    店舗データの前処理を行うクラス
    """
    
    def __init__(self, use_geo_features=True):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.use_geo_features = use_geo_features and GEO_FEATURES_AVAILABLE
        self.geo_engineer = GeoFeatureEngineer() if self.use_geo_features else None
        
        if use_geo_features and not GEO_FEATURES_AVAILABLE:
            print("警告: 地理的特徴量が要求されましたが、利用できません。")
        
    def load_data(self, file_path):
        """
        CSVファイルからデータを読み込み
        """
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='shift_jis')
        
        return df
    
    def create_sample_data(self, n_samples=10000):
        """
        サンプルデータを作成（実データがない場合のテスト用）
        """
        np.random.seed(42)
        
        prefectures = ['東京都', '大阪府', '愛知県', '神奈川県', '埼玉県', '千葉県', 
                      '兵庫県', '福岡県', '北海道', '宮城県', '広島県', '京都府']
        
        regions = ['都心部', '郊外', '住宅街', '商業地区', '観光地', '駅前']
        
        stations = ['新宿駅', '渋谷駅', '池袋駅', '横浜駅', '大阪駅', '名古屋駅', 
                   '天神駅', '札幌駅', '仙台駅', '広島駅', '京都駅', '神戸駅']
        
        data = {
            '店舗ID': range(1, n_samples + 1),
            '都道府県': np.random.choice(prefectures, n_samples),
            '人流': np.maximum(np.random.normal(5000, 2000, n_samples), 100),
            'ウェブサイトありなし': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            '席数': np.random.randint(10, 200, n_samples),
            '予約可不可': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            '住所': [f'{pref}市区町村{i}' for i, pref in enumerate(np.random.choice(prefectures, n_samples))],
            '地域': np.random.choice(regions, n_samples),
            '最寄り駅': np.random.choice(stations, n_samples),
            '電話有り無し': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
            '食べログスコア': np.clip(np.random.normal(3.5, 0.5, n_samples), 1.0, 5.0),
            '食べログスコア投票数': np.clip(np.random.poisson(100, n_samples), 0, 1000),
            # 緯度経度を追加（日本全国）
            '緯度': np.random.uniform(31.0, 45.0, n_samples),  # 沖縄〜北海道
            '経度': np.random.uniform(129.0, 146.0, n_samples),  # 西日本〜東日本
        }
        
        df = pd.DataFrame(data)
        
        # 売上を特徴量から生成（リアルな相関を持たせる）
        base_sales = (
            df['人流'] * 0.5 +
            df['席数'] * 100 +
            df['ウェブサイトありなし'] * 50000 +
            df['予約可不可'] * 30000 +
            df['電話有り無し'] * 20000 +
            df['食べログスコア'] * 100000 +
            df['食べログスコア投票数'] * 200
        )
        
        # 地理的要因を考慮（東京、大阪に近いほど売上が高い）
        tokyo_lat, tokyo_lon = 35.6762, 139.6503
        osaka_lat, osaka_lon = 34.6937, 135.5023
        
        # 東京・大阪からの距離を計算（簡易版）
        tokyo_distance = np.sqrt((df['緯度'] - tokyo_lat)**2 + (df['経度'] - tokyo_lon)**2)
        osaka_distance = np.sqrt((df['緯度'] - osaka_lat)**2 + (df['経度'] - osaka_lon)**2)
        
        # 最寄りの主要都市からの距離
        min_distance = np.minimum(tokyo_distance, osaka_distance)
        
        # 距離に応じた売上補正（近いほど高い）
        geo_factor = np.exp(-min_distance * 0.5)  # 距離に応じて指数的に減衰
        
        df['売上'] = base_sales * (1 + geo_factor) + np.random.normal(0, 50000, n_samples)
        
        df['売上'] = np.maximum(df['売上'], 10000)
        
        return df
    
    def preprocess_features(self, df, is_training=True):
        """
        特徴量の前処理
        """
        df_processed = df.copy()
        
        # 住所から都道府県情報を抽出（既に都道府県列がある場合はスキップ）
        if '住所' in df_processed.columns:
            df_processed = df_processed.drop('住所', axis=1)
        
        # 特徴量エンジニアリング
        df_processed = self._feature_engineering(df_processed)
        
        # 地理的特徴量エンジニアリング
        if self.use_geo_features:
            print("地理的特徴量エンジニアリングを実行中...")
            df_processed = self.geo_engineer.create_all_geo_features(
                df_processed, 
                lat_col='緯度', 
                lon_col='経度', 
                target_col='売上' if '売上' in df_processed.columns else None,
                include_spatial_lag=len(df_processed) <= 5000  # 大きなデータセットでは空間ラグを無効化
            )
        
        # 文字列型カテゴリカル変数の処理（地理的特徴量生成後）
        categorical_columns = ['都道府県', '地域', '最寄り駅']
        
        # 地理的特徴量のカテゴリカル変数も追加
        geo_categorical_columns = [col for col in df_processed.columns 
                                 if col.endswith('ゾーン') and df_processed[col].dtype == 'object']
        categorical_columns.extend(geo_categorical_columns)
        
        if is_training:
            # Label Encodingを使用
            for col in categorical_columns:
                if col in df_processed.columns:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(
                        df_processed[col].astype(str)
                    )
        else:
            # 推論時の処理
            for col in categorical_columns:
                if col in self.label_encoders and col in df_processed.columns:
                    # 未知のカテゴリは最頻値で埋める
                    known_labels = set(self.label_encoders[col].classes_)
                    df_processed[col] = df_processed[col].astype(str)
                    unknown_mask = ~df_processed[col].isin(known_labels)
                    if unknown_mask.any():
                        most_frequent = df_processed[col].mode()[0] if not df_processed[col].mode().empty else list(known_labels)[0]
                        df_processed.loc[unknown_mask, col] = most_frequent
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # 特徴量の選択
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['店舗ID', '売上'] and not col.startswith('target_')]
        
        if is_training:
            self.feature_columns = feature_cols
        
        return df_processed[self.feature_columns] if not is_training else df_processed
    
    def _feature_engineering(self, df):
        """
        特徴量エンジニアリング
        """
        df_eng = df.copy()
        
        # 人流と席数の相互作用項
        if '人流' in df_eng.columns and '席数' in df_eng.columns:
            df_eng['人流_席数_比'] = df_eng['人流'] / (df_eng['席数'] + 1)
        
        # 食べログスコアと投票数の相互作用項
        if '食べログスコア' in df_eng.columns and '食べログスコア投票数' in df_eng.columns:
            df_eng['食べログ_重み付きスコア'] = (
                df_eng['食べログスコア'] * np.log1p(df_eng['食べログスコア投票数'])
            )
        
        # サービス充実度スコア
        service_cols = ['ウェブサイトありなし', '予約可不可', '電話有り無し']
        available_service_cols = [col for col in service_cols if col in df_eng.columns]
        if available_service_cols:
            df_eng['サービス充実度'] = df_eng[available_service_cols].sum(axis=1)
        
        return df_eng
        
    def split_data(self, df, target_col='売上', test_size=0.2, random_state=42):
        """
        データを訓練用とテスト用に分割
        """
        X = df.drop([target_col, '店舗ID'], axis=1, errors='ignore')
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # サンプルデータでテスト
    preprocessor = StoreDataPreprocessor()
    sample_df = preprocessor.create_sample_data(1000)
    
    print("サンプルデータの形状:", sample_df.shape)
    print("\nサンプルデータの先頭5行:")
    print(sample_df.head())
    
    # 前処理のテスト
    processed_df = preprocessor.preprocess_features(sample_df, is_training=True)
    print(f"\n前処理後のデータ形状: {processed_df.shape}")
    print(f"特徴量カラム: {preprocessor.feature_columns}") 