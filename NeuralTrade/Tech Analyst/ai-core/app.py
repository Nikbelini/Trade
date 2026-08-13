import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sktime.forecasting.base import ForecastingHorizon

st.set_page_config(page_title="SBER ML Forecast", layout="wide")
st.title("📈 Прогноз акций СБЕР (ML Гибрид)")

# Загрузка модели
@st.cache_resource
def load_model():
    try:
        return joblib.load('sber_best_model1.pkl')
    except Exception as exception:
        return None

artifacts = load_model()

if artifacts is None:
    st.error("Модель не найдена!`")
    st.stop()

model = artifacts['model']
y_train = artifacts['y_train']
last_price = artifacts['last_price']
score_smape = artifacts['score_smape']
score_rmse = artifacts['score_rmse']

# Сайдбар
st.sidebar.header("⚙️ Настройки")
forecast_days = st.sidebar.slider("Дней прогноза", 7, 60, 30)
show_metrics = st.sidebar.checkbox("Показать метрики качества", value=True)

# Метрики
if show_metrics:
    st.sidebar.info(f"**sMAPE:** {score_smape:.3f}\n\n**RMSE:** {score_rmse:.3f}")

# График истории
st.subheader("📊 История котировок")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=y_train.index, 
    y=y_train, 
    mode='lines', 
    name='История',
    line=dict(color='blue', width=2)
))
fig_hist.update_layout(
    height=400, 
    xaxis_title="Дата", 
    yaxis_title="Цена (RUB)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig_hist, use_container_width=True)

# Прогноз
st.subheader("🤖 ML Прогноз")

if st.button("🔮 Рассчитать прогноз", type="primary"):
    with st.spinner('Модель анализирует...'):
        try:
            # Горизонт прогноза
            fh = ForecastingHorizon(values=range(1, forecast_days + 1), is_relative=True)
            
            # Предсказание
            pred_values = model.predict(fh)
            
            # Даты
            last_date = y_train.index[-1]
            pred_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast_days, 
                freq='D'
            )
            
            df_pred = pd.DataFrame({
                'Дата': pred_dates, 
                'Прогноз': pred_values.values
            })
            
            # График с прогнозом
            fig_forecast = go.Figure()
            
            # История (последние 90 дней)
            hist_tail = y_train.tail(90)
            fig_forecast.add_trace(go.Scatter(
                x=hist_tail.index, 
                y=hist_tail, 
                mode='lines', 
                name='История',
                line=dict(color='blue', width=2)
            ))
            
            # Прогноз
            fig_forecast.add_trace(go.Scatter(
                x=df_pred['Дата'], 
                y=df_pred['Прогноз'], 
                mode='lines+markers', 
                name='Прогноз ML',
                line=dict(color='red', dash='dot', width=2),
                marker=dict(size=6)
            ))
            
            fig_forecast.update_layout(
                title=f"Прогноз на {forecast_days} дней (Гибрид: KNN-num + GBR-num)",
                xaxis_title="Дата",
                yaxis_title="Цена (RUB)",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Метрики
            current_price = y_train.iloc[-1]
            end_pred = df_pred['Прогноз'].iloc[-1]
            change = ((end_pred - current_price) / current_price) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Последняя цена", f"{current_price:.2f} ₽")
            c2.metric("Прогноз (конец)", f"{end_pred:.2f} ₽")
            c3.metric("Ожидаемое изменение", f"{change:.2f}%", delta_color="normal")
            
            # Таблица
            with st.expander("📋 Таблица прогноза по дням"):
                df_display = df_pred.copy()
                df_display['Дата'] = df_display['Дата'].dt.strftime('%Y-%m-%d')
                df_display['Прогноз'] = df_display['Прогноз'].round(2)
                st.dataframe(df_display, use_container_width=True)
            
        except Exception as exception:
            st.error(f"Ошибка: {exception}")
