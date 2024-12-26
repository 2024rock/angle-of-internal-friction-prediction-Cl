# app.py
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = "soil_parameter_model.h5"
    DATA_DIR = "soil_parameter_model.h5"
    FEATURE_COLS = ['depth', 'void ratio', 'water content', 'X', 'Y', 'UU', 'CU', 'CUBar', 'CD']
    TARGET_COL = 'φ'
    PARAM_RANGES = {
        'depth': {'min': 0.0, 'max': 10000.0, 'default': 0.0, 'type': 'continuous', 'section': '一般'},
        'void ratio': {'min': 0.0, 'max': 10000.0, 'default': 0.0, 'type': 'continuous', 'section': '一般'},
        'water content': {'min': 0.0, 'max': 10000.0, 'default': 0.0, 'type': 'continuous', 'section': '一般'},
        'X': {'min': -1000.0, 'max': 1000.0, 'default': 0.0, 'type': 'continuous', 'section': '座標（土質により決定）'},
        'Y': {'min': -1000.0, 'max': 1000.0, 'default': 0.0, 'type': 'continuous', 'section': '座標（土質により決定）'},
        'UU': {'min': 0, 'max': 1, 'default': 0, 'type': 'binary', 'section': '三軸圧縮試験'},
        'CU': {'min': 0, 'max': 1, 'default': 0, 'type': 'binary', 'section': '三軸圧縮試験'},
        'CUBar': {'min': 0, 'max': 1, 'default': 0, 'type': 'binary', 'section': '三軸圧縮試験'},
        'CD': {'min': 0, 'max': 1, 'default': 0, 'type': 'binary', 'section': '三軸圧縮試験'}
    }

class PredictionApp:
    def __init__(self):
        self.model = None
        self.load_artifacts()

    @st.cache_resource
    def _load_model():
        """モデルの読み込み"""
        try:
            model_path = "soil_parameter_model.h5"
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            raise Exception(f"モデル読み込みエラー: {e}\nパス: {model_path}")

    def load_artifacts(self):
        """モデルの読み込み"""
        try:
            self.model = PredictionApp._load_model()
            return True
        except Exception as e:
            st.error(f"モデルの読み込みエラー: {e}")
            return False

    def validate_input(self, input_values):
        """入力値の検証"""
        for col, value in input_values.items():
            if col in Config.PARAM_RANGES:
                ranges = Config.PARAM_RANGES[col]
                if value < ranges['min'] or value > ranges['max']:
                    return False, f"{col}の値が範囲外です ({ranges['min']}~{ranges['max']})"
        return True, None

    def predict(self, input_values):
        """予測の実行"""
        if self.model is None:
            return None, "モデルが読み込まれていません"

        try:
            # 入力値の検証
            is_valid, error_message = self.validate_input(input_values)
            if not is_valid:
                return None, error_message

            # 予測用データの準備
            input_array = np.array([[input_values[name] for name in Config.FEATURE_COLS]])

            # 予測を実行
            with st.spinner('予測を実行中...'):
                prediction = self.model.predict(input_array, verbose=0)

            return prediction[0][0], None

        except Exception as e:
            return None, f"予測実行エラー: {str(e)}"

def main():
    st.set_page_config(page_title="内部摩擦角予測アプリ", layout="wide")

    st.title("内部摩擦角予測アプリ")

    # 説明文を更新
    st.markdown("""
    このアプリケーションでは、入力されたパラメータに基づいて内部摩擦角（φ）を予測します。
    サイドバーから必要な数値を入力し、**予測実行**ボタンをクリックしてください。

    なお、入力値のx,yは以下の組み合わせを入力してください：
    - 礫: (-75.62, 38.14)
    - 砂: (-122.47, -114.98)
    - シルト: (30.24, -162.03)
    - 粘土: (77.65, -9.26)
    """)

    # アプリケーションのインスタンス化
    app = PredictionApp()

    # サイドバーに入力フォームを配置
    with st.sidebar:
        st.header("システム情報")
        if app.model is not None:
            st.success("システム準備完了")
        else:
            st.error("システムの初期化に失敗しました")

        st.header("パラメータ入力")
        input_values = {}

        # セクションごとにパラメータをグループ化
        sections = {
            '一般': [],
            '座標（土質により決定）': [],
            '三軸圧縮試験': []
        }

        # パラメータをセクションごとに分類
        for feature in Config.FEATURE_COLS:
            section = Config.PARAM_RANGES[feature]['section']
            sections[section].append(feature)

        # セクションごとにパラメータを表示
        for section, params in sections.items():
            st.subheader(section)
            for feature in params:
                range_info = Config.PARAM_RANGES[feature]
                if range_info['type'] == 'continuous':
                    input_values[feature] = st.number_input(
                        f"{feature}",
                        min_value=range_info['min'],
                        max_value=range_info['max'],
                        value=range_info['default'],
                        step=0.1,
                        format="%.3f"
                    )
                else:  # binary type
                    value = st.radio(
                        f"{feature}",
                        options=[0, 1],
                        format_func=lambda x: "0" if x == 0 else "1",
                        horizontal=True
                    )
                    input_values[feature] = float(value)

        predict_button = st.button("予測実行", type="primary")

    # メインパネルに予測結果を表示
    if predict_button:
        prediction, error = app.predict(input_values)

        if error:
            st.error(error)
        else:
            st.header("予測結果")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label=f"予測された{Config.TARGET_COL}の値",
                    value=f"{prediction:.2f}"
                )

    # 注意事項
    st.markdown("---")
    st.subheader("使用上の注意")
    st.markdown("""
    - 入力値は適切な範囲内の値を使用してください
    - UU, CU, CUBar, CDは0または1のみ選択可能です
    - 予測結果は参考値であり、実際の値とは異なる可能性があります
    - 異常な値を入力した場合、予測結果の信頼性が低下します
    """)

if __name__ == "__main__":
    main()
