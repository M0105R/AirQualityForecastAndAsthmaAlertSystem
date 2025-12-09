Air Quality Forecast & Asthma Alert System

A real-time Air Quality Monitoring, 24-Hour AQI Forecasting, and Asthma Health Advisory application built using Streamlit, Prophet, and live air-quality data from CPCB (data.gov.in API).
This project helps users, especially asthma patients, make informed decisions about outdoor exposure.

How It Works:
1. Real-Time AQI Fetching:
Pulls fresh air-quality readings from the data.gov.in endpoint for Indian cities.

2. Historical AQI Storage:
Every time user checks AQI, the app stores it in aqi_history.csv.

3. Forecasting:
If enough historical data → Prophet model
If not enough → synthetic interpolation + fallback model

4. Personalized Health Advice:
Based on:

AQI category

Asthma status

Asthma severity

5. In-App Alerts
Shown automatically when AQI falls into:

Poor

Very Poor

Severe


Tech Stack:

Technology	      -            Purpose

Python	          -       Core programming

Streamlit         -      UI + in-app alerts

Prophet	          -        Time-series forecasting

Pandas, NumPy	    -          Data processing

Plotly	          -        Interactive charts

Requests	        -            API calls

data.gov.in API	  -          Real-time AQI



Installation & Setup:

1️. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py
