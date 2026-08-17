import streamlit as st

st.set_page_config(page_title="Air Quality Intelligence", page_icon="🌫️", layout="wide")

st.title("Air Quality Intelligence Platform")
st.caption("Phase 1 MVP — Indian urban air quality analytics and forecasting")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current AQI", "—", help="Populated after the ingestion pipeline is connected.")
col2.metric("7-day average", "—")
col3.metric("Dominant pollutant", "—")
col4.metric("Stations online", "—")

st.info("The dashboard shell is ready. Connect PostgreSQL/PostGIS and run the ingestion jobs to populate live KPIs, charts, forecasts, and the station map.")

with st.expander("Phase 1 modules"):
    st.markdown("""
    - Live/near-real-time pollutant ingestion via OpenAQ
    - Hourly weather ingestion via Open-Meteo
    - PostGIS station geometry and time-series storage
    - CPCB-style AQI sub-index calculation
    - Diurnal, weekday, correlation and ranking analysis
    - Naive and SARIMA forecasting with MAE/RMSE
    - Dockerized PostgreSQL + PostGIS + Streamlit stack
    """)
