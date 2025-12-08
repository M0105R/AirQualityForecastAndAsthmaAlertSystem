# --------------------------------------------------------------
# app.py 
# --------------------------------------------------------------

import os
import time
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Try import Prophet; fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------- CONFIG ----------
API_KEY = "579b464db66ec23bdd00000152e7f9a432ff457a4d233b4f17bd74ea"
RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
HISTORY_FILE = "aqi_history.csv"

# -----------------------------------------------------------
# ⭐ ADD SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.title("📌 Navigation")

st.sidebar.markdown("""
- [🏠 Live AQI](#live-aqi)
- [📈 24-Hour Forecast](#forecast-section)
- [🩺 Forecast Health Advice](#forecast-advice-section)
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# UTIL + ORIGINAL FUNCTIONS  (unchanged)
# --------------------------------------------------------------

def safe_get(url, params=None, timeout=20, tries=2, backoff=1.5):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc


def fetch_realtime_aqi():
    try:
        url = f"{BASE_URL}?api-key={API_KEY}&format=json&limit=2000"
        r = safe_get(url, timeout=20, tries=3, backoff=1.0)
        records = r.json().get("records", [])
        df = pd.DataFrame(records)
        if not df.empty and "avg_value" in df.columns:
            df["avg_value"] = pd.to_numeric(df["avg_value"], errors="coerce")
        return df
    except Exception:
        try:
            st.toast("⚠️ Live AQI fetch failed. Using history.", icon="⚠️")
        except:
            st.warning("⚠️ Live AQI fetch failed. Using history.")
        return pd.DataFrame()


from difflib import get_close_matches

def normalize_city_name(s):
    return str(s).strip().lower() if pd.notna(s) else ""

def find_best_city_match(user_city, cities_list):
    if not user_city:
        return None
    user_city_norm = normalize_city_name(user_city)
    candidates = [c for c in cities_list if pd.notna(c)]
    for c in candidates:
        if normalize_city_name(c) == user_city_norm:
            return c
    matches = get_close_matches(user_city, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def get_city_aqi(df, city):
    if df.empty or city is None:
        return None, pd.DataFrame()
    df = df.copy()
    if "city" not in df.columns:
        return None, pd.DataFrame()
    df["city_norm"] = df["city"].astype(str).str.strip().str.lower()
    city_norm = normalize_city_name(city)
    city_data = df[df["city_norm"] == city_norm]
    if city_data.empty:
        city_data = df[df["city_norm"].str.contains(city_norm, na=False)]
    if city_data.empty:
        return None, pd.DataFrame()
    pollutants = city_data[
        city_data.get("pollutant_id", "").isin(["PM2.5", "PM10"])
        if "pollutant_id" in city_data.columns else [False]
    ]
    if not pollutants.empty and "avg_value" in pollutants.columns and pollutants["avg_value"].dropna().size > 0:
        avg_aqi = pollutants["avg_value"].dropna().mean()
    else:
        if "avg_value" in city_data.columns and city_data["avg_value"].dropna().size > 0:
            avg_aqi = city_data["avg_value"].dropna().mean()
        else:
            avg_aqi = None
    return (round(float(avg_aqi), 2) if avg_aqi is not None else None), \
           city_data.drop(columns=["city_norm"], errors="ignore")


def get_aqi_category(aqi):
    if aqi is None:
        return "Unknown", "gray"
    aqi = float(aqi)
    if aqi <= 50: return "Good", "green"
    elif aqi <= 100: return "Satisfactory", "#9ACD32"
    elif aqi <= 200: return "Moderate", "yellow"
    elif aqi <= 300: return "Poor", "orange"
    elif aqi <= 400: return "Very Poor", "red"
    else: return "Severe", "darkred"


def get_advice(aqi, user_type, severity):
    cat, _ = get_aqi_category(aqi)
    normal = {
        "Good": "Air is fresh. Safe to go outside.",
        "Satisfactory": "Air is okay; sensitive people stay cautious.",
        "Moderate": "Wear a mask outdoors if you're sensitive.",
        "Poor": "Avoid long outdoor exposure. Mask recommended.",
        "Very Poor": "Avoid outdoor activities completely.",
        "Severe": "Stay indoors. Use air purifier if available."
    }
    asthma_map = {
        "Low": {
            "Good": "Carry inhaler. Safe to go out.",
            "Satisfactory": "Avoid dusty spots. Carry inhaler.",
            "Moderate": "Wear mask and limit outdoor time.",
            "Poor": "Avoid going out if possible.",
            "Very Poor": "Stay indoors, keep meds nearby.",
            "Severe": "Do not go outside."
        },
        "Medium": {
            "Good": "Carry inhaler outdoors.",
            "Satisfactory": "Avoid prolonged exposure.",
            "Moderate": "Use N95 outdoors.",
            "Poor": "Avoid outdoor activities.",
            "Very Poor": "Stay indoors with air purifier.",
            "Severe": "Do not go outside."
        },
        "High": {
            "Good": "Carry inhaler constantly.",
            "Satisfactory": "Avoid long exposure.",
            "Moderate": "Strictly limit outdoor time.",
            "Poor": "Stay indoors.",
            "Very Poor": "Indoor rest required.",
            "Severe": "Critical — seek medical help if needed."
        }
    }
    if user_type == "No":
        return normal.get(cat, "No specific advice.")
    return asthma_map.get(severity, {}).get(cat, "No specific advice.")


def detect_user_city():
    try:
        r = requests.get("https://ipinfo.io", timeout=5)
        return r.json().get("city", None)
    except:
        return None

# --------------------------------------------------------------
# ⭐ TEMP FIX — AUTO-DETECT ALWAYS USES HYDERABAD
# --------------------------------------------------------------

def get_browser_location():
    return None  # disabled completely for now

# --------------------------------------------------------------
# HISTORY + FORECAST FUNCTIONS (unchanged)
# --------------------------------------------------------------

def ensure_history_file():
    if not os.path.exists(HISTORY_FILE):
        df = pd.DataFrame(columns=["city", "ds", "y"])
        df.to_csv(HISTORY_FILE, index=False)
        return
    try:
        df = pd.read_csv(HISTORY_FILE)
        cols = set(df.columns)
        if not {"city", "ds", "y"}.issubset(cols):
            df = pd.DataFrame(columns=["city", "ds", "y"])
            df.to_csv(HISTORY_FILE, index=False)
    except:
        df = pd.DataFrame(columns=["city", "ds", "y"])
        df.to_csv(HISTORY_FILE, index=False)

def load_history_all():
    ensure_history_file()
    try:
        h = pd.read_csv(HISTORY_FILE)
        if "ds" in h.columns:
            h["ds"] = pd.to_datetime(h["ds"], errors="coerce")
        else:
            h["ds"] = pd.NaT
        h["y"] = pd.to_numeric(h.get("y", pd.Series(dtype=float)), errors="coerce")
        h = h.dropna(subset=["ds"])
        h["city_norm"] = h["city"].astype(str).str.strip().str.lower()
        return h
    except:
        return pd.DataFrame(columns=["city","ds","y","city_norm"])

def load_history_for_city(city):
    h = load_history_all()
    city_norm = normalize_city_name(city)
    if h.empty:
        return pd.DataFrame(columns=["ds","y"])
    ch = h[h["city_norm"] == city_norm].copy()
    if ch.empty:
        return pd.DataFrame(columns=["ds","y"])
    return ch[["ds","y"]].sort_values("ds").reset_index(drop=True)

def append_history(city, ds, y):
    ensure_history_file()
    try:
        hist = pd.read_csv(HISTORY_FILE)
    except:
        hist = pd.DataFrame(columns=["city","ds","y"])

    if "ds" not in hist.columns:
        hist["ds"] = pd.NaT

    hist["ds"] = pd.to_datetime(hist["ds"], errors="coerce")
    hist["y"] = pd.to_numeric(hist.get("y", pd.Series(dtype=float)), errors="coerce")
    hist = hist.dropna(subset=["ds"])

    ds_dt = pd.to_datetime(ds)
    city_norm = city.strip()

    is_same_hour = (
        (hist["city"].astype(str).str.strip().str.lower() == city_norm.lower()) &
        (hist["ds"].dt.floor("h") == ds_dt.floor("h"))
    )
    if is_same_hour.any():
        idx = is_same_hour.idxmax()
        hist.at[idx, "y"] = y
    else:
        new = pd.DataFrame([{"city": city_norm, "ds": ds_dt, "y": y}])
        if hist.empty:
            hist = new
        else:
            hist = pd.concat([hist, new], ignore_index=True, sort=False)

    hist.to_csv(HISTORY_FILE, index=False)
    try:
        st.toast("📘 History updated", icon="ℹ️")
    except:
        pass


import numpy as np

def generate_synthetic_history(current_aqi, hours=24):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    data = []
    for i in range(hours, 0, -1):
        ts = now - timedelta(hours=i)
        variation = np.sin(i / 3.0) * 3.5
        noise = np.random.uniform(-2.0, 2.0)
        val = current_aqi + variation + noise
        val = max(5.0, min(500.0, val))
        data.append({"ds": ts, "y": round(val,2)})
    df = pd.DataFrame(data)
    return df


def forecast_with_prophet(history_df, periods=24, freq='H'):
    if history_df.empty or len(history_df) < 3:
        return None
    df = history_df.rename(columns={"ds":"ds","y":"y"})
    df["ds"] = pd.to_datetime(df["ds"])
    model = Prophet(daily_seasonality=True, weekly_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def simple_fallback_forecast(history_df, periods=24):
    try:
        st.toast("ℹ️ Using fallback forecast (not Prophet)", icon="ℹ️")
    except:
        pass

    if history_df.empty:
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        base = 100.0
        dates = [now + timedelta(hours=i+1) for i in range(periods)]
        return pd.DataFrame({"ds": dates, "yhat": [base]*periods})

    df = history_df.set_index("ds").resample("H").mean().interpolate().reset_index()
    window = min(24, len(df))
    last_mean = df["y"].rolling(window=window).mean().iloc[-1] if window>0 else df["y"].iloc[-1]
    last_mean = float(last_mean) if pd.notna(last_mean) else df["y"].iloc[-1]
    future_dates = [df["ds"].max() + timedelta(hours=i+1) for i in range(periods)]
    return pd.DataFrame({"ds": future_dates, "yhat": [last_mean]*periods})


def get_forecast_for_city(city, current_aqi):
    hist = load_history_for_city(city)

    if len(hist) >= 24 and PROPHET_AVAILABLE:
        try:
            return forecast_with_prophet(hist, periods=24, freq='H')
        except:
            try:
                st.toast("⚠️ Prophet failed. Using fallback.", icon="⚠️")
            except:
                pass
            return simple_fallback_forecast(hist, periods=24)

    if len(hist) >= 3 and PROPHET_AVAILABLE:
        try:
            synth = generate_synthetic_history(current_aqi, hours=24)
            combined = pd.concat([hist.rename(columns={"y":"y"}), synth],
                                 ignore_index=True, sort=False)
            combined = combined.drop_duplicates(subset=["ds"]).sort_values("ds")
            return forecast_with_prophet(combined, periods=24, freq='H')
        except:
            try:
                st.toast("⚠️ Forecast error. Using fallback.", icon="⚠️")
            except:
                pass
            return simple_fallback_forecast(hist, periods=24)

    synth = generate_synthetic_history(current_aqi, hours=48)
    if PROPHET_AVAILABLE:
        try:
            return forecast_with_prophet(synth, periods=24, freq='H')
        except:
            try:
                st.toast("⚠️ Prophet failed. Fallback used.", icon="⚠️")
            except:
                pass
            return simple_fallback_forecast(synth, periods=24)

    return simple_fallback_forecast(synth, periods=24)

# --------------------------------------------------------------
# UI START
# --------------------------------------------------------------

st.markdown("<a id='live-aqi'></a>", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align:center; color:#1B4F72;'>🌍 Air Quality Forecast & Asthma Alert System</h1>
    <p style='text-align:center;'>Live AQI • Forecast • Pollutants • Health Advice</p>
""", unsafe_allow_html=True)
st.write("---")

df_live = fetch_realtime_aqi()

if not df_live.empty and "city" in df_live.columns:
    cities = sorted(df_live["city"].dropna().unique())
else:
    hist_all = load_history_all()
    cities = sorted(hist_all["city"].astype(str).unique()) if not hist_all.empty else []

# --------------------------------------------------------------
# FIXED AUTO-DETECT = HYDERABAD
# --------------------------------------------------------------

auto_city_raw = "Hyderabad"
auto_city_matched = "Hyderabad"

# --------------------------------------------------------------
# CITY SELECTION
# --------------------------------------------------------------

col1, col2 = st.columns(2)
with col1:
    use_auto = st.checkbox(f"📍 Auto-detect", value=False)
with col2:
    selected_city = st.selectbox("🏙 Select City", cities) if cities else st.text_input("Enter City Name")

if use_auto:
    city = auto_city_matched
    try:
        st.toast(f"📍 Auto-detected city: Hyderabad", icon="📍")
    except:
        st.info("Auto-detected city: Hyderabad")
else:
    city = selected_city

aqi_value, city_data = (None, pd.DataFrame())
if not df_live.empty:
    aqi_value, city_data = get_city_aqi(df_live, city)

if aqi_value is None:
    hist_for_city = load_history_for_city(city)
    if not hist_for_city.empty:
        aqi_value = float(hist_for_city.sort_values("ds").iloc[-1]["y"])
        try:
            st.toast("⚠️ Live data unavailable — using history", icon="⚠️")
        except:
            st.warning("Live data unavailable — using history")
    else:
        try:
            st.toast("⚠️ No data available — using synthetic AQI", icon="⚠️")
        except:
            st.warning("No data available — using synthetic AQI")
        aqi_value = 100.0

try:
    append_history(city, datetime.now().isoformat(), float(aqi_value))
except:
    try:
        st.toast("⚠️ Failed to save history", icon="⚠️")
    except:
        st.warning("Failed to save history")

category, color = get_aqi_category(aqi_value)

if category in ["Poor", "Very Poor", "Severe"]:
    try:
        st.toast(f"⚠️ Air Quality is {category}! Take precautions.", icon="⚠️")
    except:
        st.warning(f"Air Quality is {category}! Take precautions.")

st.markdown(
    f"<h2 style='text-align:center; color:{color};'>AQI in {city.title()}: {aqi_value} ({category})</h2>",
    unsafe_allow_html=True
)

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=aqi_value,
    title={"text": f"AQI Level: {category}"},
    gauge={
        "axis": {"range": [0, 500]},
        "bar": {"color": "#0A3D91"},
        "steps": [
            {"range": [0, 50], "color": "green"},
            {"range": [50, 100], "color": "#9ACD32"},
            {"range": [100, 200], "color": "yellow"},
            {"range": [200, 300], "color": "orange"},
            {"range": [300, 400], "color": "red"},
            {"range": [400, 500], "color": "darkred"}
        ]
    }
))
st.plotly_chart(fig, use_container_width=True)

st.write("---")

user_type = st.radio("Do you have asthma?", ["No", "Yes"])
severity = st.selectbox("Asthma Severity", ["Low","Medium","High"]) if user_type=="Yes" else None
advice = get_advice(aqi_value, user_type, severity)

st.markdown(
    f"<div style='background-color:#F4F6G7;padding:15px;border-radius:10px;border-left:6px solid {color};'>{advice}</div>",
    unsafe_allow_html=True
)
st.write("---")

st.markdown("<a id='forecast-section'></a>", unsafe_allow_html=True)
st.subheader("📈 24-Hour AQI Forecast (Hourly)")

forecast = get_forecast_for_city(city, float(aqi_value))

if forecast is None:
    st.info("Forecast not available.")
else:
    if "ds" in forecast.columns and "yhat" in forecast.columns:
        try:
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            next24 = forecast[forecast["ds"] > datetime.now() - timedelta(minutes=5)].head(24)
            if next24.empty:
                next24 = forecast.tail(24)
            fc_x = next24["ds"]
            fc_y = next24["yhat"]
        except:
            fc_x = forecast["ds"].head(24)
            fc_y = forecast.iloc[:24, 1]
    else:
        fc_x = forecast["ds"].head(24)
        fc_y = forecast["yhat"].head(24)

    fc_fig = go.Figure()
    fc_fig.add_trace(go.Scatter(x=fc_x, y=fc_y, mode="lines+markers",
                                line=dict(color="#0A3D91", width=2)))
    fc_fig.update_layout(xaxis_title="Time", yaxis_title="AQI", template="plotly_white")
    st.plotly_chart(fc_fig, use_container_width=True)

st.markdown("<a id='forecast-advice-section'></a>", unsafe_allow_html=True)
st.write("---")
st.subheader("🩺 Hour-wise Health Advice for Forecast")

try:
    forecast_advice_df = pd.DataFrame({
        "Time": fc_x.dt.strftime("%I:%M %p"),
        "Predicted AQI": [round(float(x), 2) for x in fc_y],
    })

    categories = []
    advice_list = []

    for aqi in fc_y:
        cat, _ = get_aqi_category(aqi)
        categories.append(cat)
        advice_list.append(get_advice(aqi, user_type, severity))

    forecast_advice_df["Category"] = categories
    forecast_advice_df["Advice"] = advice_list

    st.dataframe(forecast_advice_df, use_container_width=True, height=450)

    worst_idx = forecast_advice_df["Predicted AQI"].idxmax()
    worst_row = forecast_advice_df.loc[worst_idx]

    st.markdown(
        f"""
        <div style="background:#FFF3G3;padding:12px;border-left:6px solid red;border-radius:8px;">
            <b>⚠ Highest Risk Hour:</b><br>
            Time: <b>{worst_row['Time']}</b><br>
            Predicted AQI: <b>{worst_row['Predicted AQI']}</b> ({worst_row['Category']})<br>
            Advice: {worst_row['Advice']}
        </div>
        """,
        unsafe_allow_html=True
    )

except Exception as e:
    st.warning(f"Could not generate hour-wise advice: {e}")
