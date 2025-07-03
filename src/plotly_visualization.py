import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PlotlyVisualizer:
    """
    Plotly Expressを使った売上予測結果の可視化クラス
    """
    
    def __init__(self, theme='plotly_white'):
        """
        初期化
        
        Args:
            theme: plotlyのテーマ ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white')
        """
        self.theme = theme
        
    def plot_feature_importance(self, feature_importance: pd.DataFrame, 
                              top_n: int = 20, 
                              title: str = "特徴量重要度",
                              width: int = 800,
                              height: int = 600) -> go.Figure:
        """
        特徴量重要度のインタラクティブなバープロット
        """
        top_features = feature_importance.head(top_n).copy()
        
        fig = px.bar(
            top_features,
            y='feature',
            x='importance',
            orientation='h',
            title=f'{title} (Top {top_n})',
            labels={'importance': '重要度 (Gain)', 'feature': '特徴量'},
            template=self.theme,
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            width=width,
            height=height,
            yaxis={'categoryorder': 'total ascending'},
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def plot_prediction_vs_actual(self, y_test: pd.Series, y_pred: np.ndarray,
                                 title: str = "予測値 vs 実際値",
                                 width: int = 800,
                                 height: int = 600) -> go.Figure:
        """
        予測値vs実際値のインタラクティブな散布図
        """
        # R²スコアの計算
        r2 = r2_score(y_test, y_pred)
        
        # データフレームの作成
        df_plot = pd.DataFrame({
            '実際値': y_test,
            '予測値': y_pred,
            '誤差': np.abs(y_test - y_pred),
            '誤差率': np.abs((y_test - y_pred) / y_test) * 100
        })
        
        fig = px.scatter(
            df_plot,
            x='実際値',
            y='予測値',
            color='誤差率',
            size='誤差',
            hover_data=['誤差'],
            title=f'{title} (R² = {r2:.3f})',
            labels={'誤差率': '誤差率 (%)', '誤差': '絶対誤差'},
            template=self.theme,
            color_continuous_scale='viridis'
        )
        
        # 完全予測の線を追加
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='完全予測ライン',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def plot_sales_distribution(self, sales_data: pd.Series,
                              title: str = "売上分布",
                              width: int = 800,
                              height: int = 500) -> go.Figure:
        """
        売上分布のヒストグラム
        """
        fig = px.histogram(
            x=sales_data,
            nbins=50,
            title=title,
            labels={'x': '売上', 'y': '頻度'},
            template=self.theme,
            marginal='box'  # ボックスプロットも表示
        )
        
        # 統計情報の追加
        mean_val = sales_data.mean()
        median_val = sales_data.median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                     annotation_text=f"平均: {mean_val:,.0f}円")
        fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                     annotation_text=f"中央値: {median_val:,.0f}円")
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, 
                                features: list = None,
                                title: str = "特徴量相関ヒートマップ",
                                width: int = 800,
                                height: int = 600) -> go.Figure:
        """
        特徴量間の相関ヒートマップ
        """
        if features:
            correlation_data = data[features].corr()
        else:
            # 数値列のみを選択
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            correlation_data = data[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title=title,
            template=self.theme,
            color_continuous_scale='RdBu'
        )
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=10),
            title_font_size=16
        )
        
        return fig
    
    def plot_residuals(self, y_test: pd.Series, y_pred: np.ndarray,
                      title: str = "残差プロット",
                      width: int = 800,
                      height: int = 500) -> go.Figure:
        """
        残差プロット
        """
        residuals = y_test - y_pred
        
        df_residuals = pd.DataFrame({
            '予測値': y_pred,
            '残差': residuals,
            '絶対残差': np.abs(residuals)
        })
        
        fig = px.scatter(
            df_residuals,
            x='予測値',
            y='残差',
            color='絶対残差',
            title=title,
            labels={'予測値': '予測値', '残差': '残差 (実際値 - 予測値)'},
            template=self.theme,
            color_continuous_scale='viridis'
        )
        
        # ゼロライン追加
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=12),
            title_font_size=16
        )
        
        return fig
    
    def plot_categorical_analysis(self, data: pd.DataFrame, 
                                 categorical_col: str,
                                 target_col: str = '売上',
                                 plot_type: str = 'box',
                                 title: str = None,
                                 width: int = 800,
                                 height: int = 600) -> go.Figure:
        """
        カテゴリカル変数別の売上分析
        
        Args:
            plot_type: 'box', 'violin', 'bar'のいずれか
        """
        if title is None:
            title = f'{categorical_col}別{target_col}分析'
        
        if plot_type == 'box':
            fig = px.box(
                data,
                x=categorical_col,
                y=target_col,
                title=title,
                template=self.theme
            )
        elif plot_type == 'violin':
            fig = px.violin(
                data,
                x=categorical_col,
                y=target_col,
                box=True,
                title=title,
                template=self.theme
            )
        elif plot_type == 'bar':
            grouped_data = data.groupby(categorical_col)[target_col].agg(['mean', 'count']).reset_index()
            fig = px.bar(
                grouped_data,
                x=categorical_col,
                y='mean',
                title=f'{title} (平均値)',
                labels={'mean': f'平均{target_col}', 'count': '店舗数'},
                template=self.theme,
                hover_data=['count']
            )
        
        fig.update_layout(
            width=width,
            height=height,
            font=dict(size=12),
            title_font_size=16,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def plot_geographic_distribution(self, df: pd.DataFrame, 
                                   lat_col: str = '緯度', 
                                   lon_col: str = '経度',
                                   value_col: str = '売上',
                                   title: str = "売上の地理的分布",
                                   width: int = 1000,
                                   height: int = 600) -> go.Figure:
        """
        地理的分布のマップ表示
        """
        if lat_col not in df.columns or lon_col not in df.columns:
            print(f"警告: {lat_col} または {lon_col} が見つかりません。")
            return go.Figure()
        
        # 値でカラーマップを作成
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=value_col,
            size=value_col,
            hover_data=[col for col in ['店舗ID', '都道府県', '地域'] if col in df.columns],
            color_continuous_scale='Viridis',
            mapbox_style='open-street-map',
            title=title,
            width=width,
            height=height
        )
        
        # 日本の中心にフォーカス
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=35.6762, lon=139.6503),
                zoom=5
            ),
            template=self.theme
        )
        
        return fig
    
    def plot_geo_feature_analysis(self, df: pd.DataFrame,
                                geo_features: list = None,
                                target_col: str = '売上',
                                title: str = "地理的特徴量分析",
                                width: int = 1200,
                                height: int = 800) -> go.Figure:
        """
        地理的特徴量の分析プロット
        """
        if geo_features is None:
            geo_features = [col for col in df.columns if any(keyword in col for keyword in 
                           ['距離', 'クラスタ', '近隣', '密度', '空間', 'ゾーン'])]
        
        if not geo_features:
            print("地理的特徴量が見つかりません。")
            return go.Figure()
        
        # 上位特徴量を選択（最大6つ）
        selected_features = geo_features[:6]
        
        # サブプロットを作成
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=selected_features,
            specs=[[{"secondary_y": False}]*3]*2
        )
        
        for i, feature in enumerate(selected_features):
            row = i // 3 + 1
            col = i % 3 + 1
            
            if df[feature].dtype in ['int64', 'float64']:
                # 数値特徴量: 散布図
                fig.add_trace(
                    go.Scatter(
                        x=df[feature],
                        y=df[target_col] if target_col in df.columns else [0]*len(df),
                        mode='markers',
                        name=feature,
                        showlegend=False,
                        opacity=0.6
                    ),
                    row=row, col=col
                )
            else:
                # カテゴリカル特徴量: ボックスプロット
                fig.add_trace(
                    go.Box(
                        x=df[feature],
                        y=df[target_col] if target_col in df.columns else [0]*len(df),
                        name=feature,
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=self.theme
        )
        
        return fig
    
    def plot_cluster_analysis(self, df: pd.DataFrame,
                            lat_col: str = '緯度',
                            lon_col: str = '経度',
                            cluster_col: str = '地理クラスタ',
                            value_col: str = '売上',
                            title: str = "地理的クラスタ分析",
                            width: int = 1200,
                            height: int = 600) -> go.Figure:
        """
        地理的クラスタリング結果の可視化
        """
        if any(col not in df.columns for col in [lat_col, lon_col, cluster_col]):
            print("必要な列が見つかりません。")
            return go.Figure()
        
        # クラスタごとの平均値を計算
        cluster_stats = df.groupby(cluster_col).agg({
            lat_col: 'mean',
            lon_col: 'mean',
            value_col: 'mean' if value_col in df.columns else lambda x: 0,
            cluster_col: 'count'
        }).reset_index()
        cluster_stats.columns = [cluster_col, 'avg_lat', 'avg_lon', 'avg_value', 'count']
        
        # サブプロット作成
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['クラスタ分布マップ', 'クラスタ別売上分布'],
            specs=[[{"type": "scattermapbox"}, {"type": "box"}]]
        )
        
        # 左: 地図上でのクラスタ表示
        fig.add_trace(
            go.Scattermapbox(
                lat=df[lat_col],
                lon=df[lon_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df[cluster_col],
                    colorscale='Set1',
                    showscale=True
                ),
                text=df[cluster_col],
                name='店舗',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # クラスタ中心点を追加
        fig.add_trace(
            go.Scattermapbox(
                lat=cluster_stats['avg_lat'],
                lon=cluster_stats['avg_lon'],
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star'
                ),
                text=cluster_stats[cluster_col],
                name='クラスタ中心',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 右: クラスタ別売上分布
        if value_col in df.columns:
            for cluster in sorted(df[cluster_col].unique()):
                cluster_data = df[df[cluster_col] == cluster][value_col]
                fig.add_trace(
                    go.Box(
                        y=cluster_data,
                        name=f'クラスタ {cluster}',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # レイアウト更新
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=35.6762, lon=139.6503),
                zoom=5
            ),
            template=self.theme
        )
        
        return fig
    
    def create_dashboard(self, feature_importance: pd.DataFrame,
                        y_test: pd.Series, y_pred: np.ndarray,
                        data: pd.DataFrame = None,
                        save_html: bool = True,
                        filename: str = "sales_prediction_dashboard.html") -> go.Figure:
        """
        総合ダッシュボードの作成
        """
        # サブプロットの作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('特徴量重要度', '予測値 vs 実際値', '残差プロット', '売上分布'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 特徴量重要度 (上位10個)
        top_features = feature_importance.head(10)
        fig.add_trace(
            go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                name='重要度',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # 2. 予測値 vs 実際値
        r2 = r2_score(y_test, y_pred)
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'予測 (R²={r2:.3f})',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # 完全予測ライン
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='完全予測',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. 残差プロット
        residuals = y_test - y_pred
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='残差',
                marker=dict(color='orange', opacity=0.6)
            ),
            row=2, col=1
        )
        
        # 4. 売上分布
        fig.add_trace(
            go.Histogram(
                x=y_test,
                name='実際値分布',
                opacity=0.7,
                marker_color='blue'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Histogram(
                x=y_pred,
                name='予測値分布',
                opacity=0.7,
                marker_color='red'
            ),
            row=2, col=2
        )
        
        # レイアウト更新
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="売上予測モデル 総合ダッシュボード",
            title_font_size=20,
            template=self.theme
        )
        
        # HTMLファイルとして保存
        if save_html:
            fig.write_html(filename)
            print(f"ダッシュボードを {filename} に保存しました。")
        
        return fig

def create_quick_visualization(feature_importance: pd.DataFrame,
                             y_test: pd.Series, y_pred: np.ndarray,
                             save_html: bool = True) -> None:
    """
    クイック可視化関数
    """
    visualizer = PlotlyVisualizer()
    
    # 1. 特徴量重要度
    fig1 = visualizer.plot_feature_importance(feature_importance)
    if save_html:
        fig1.write_html("outputs/feature_importance.html")
    fig1.show()
    
    # 2. 予測値 vs 実際値
    fig2 = visualizer.plot_prediction_vs_actual(y_test, y_pred)
    if save_html:
        fig2.write_html("outputs/prediction_vs_actual.html")
    fig2.show()
    
    # 3. 残差プロット
    fig3 = visualizer.plot_residuals(y_test, y_pred)
    if save_html:
        fig3.write_html("outputs/residuals.html")
    fig3.show()
    
    # 4. ダッシュボード
    dashboard = visualizer.create_dashboard(feature_importance, y_test, y_pred, 
                                          save_html=save_html, 
                                          filename="outputs/dashboard.html")
    dashboard.show()

if __name__ == "__main__":
    # テスト用のサンプルデータ
    import sys
    sys.path.append('.')
    
    from data_preprocessing import StoreDataPreprocessor
    from lightgbm_model import SalesPredictor
    
    # サンプルデータでテスト
    preprocessor = StoreDataPreprocessor()
    sample_df = preprocessor.create_sample_data(1000)
    processed_df = preprocessor.preprocess_features(sample_df, is_training=True)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(processed_df)
    
    # モデル訓練
    predictor = SalesPredictor()
    predictor.train(X_train, y_train, X_test, y_test, verbose_eval=0)
    
    # 予測
    y_pred = predictor.predict(X_test)
    
    # 可視化
    create_quick_visualization(predictor.feature_importance, y_test, y_pred) 