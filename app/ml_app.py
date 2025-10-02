import streamlit as st
import pandas as pd
import joblib
import json

# === Load model pipeline ===
model = joblib.load("app/xgb_pipeline.joblib")

# === Load urutan fitur ===
with open("app/features_order.json", "r") as f:
    features_order = json.load(f)

# === Load lookup tables ===
airline_stats = pd.read_csv("app/airline_stats.csv")
route_stats = pd.read_csv("app/route_stats.csv")
time_stats = pd.read_csv("app/time_stats.csv")
duration_stats = pd.read_csv("app/duration_stats.csv")
arrv_per_day = pd.read_csv("app/arrv_per_day.csv")
dept_per_day = pd.read_csv("app/dept_per_day.csv")

# === Helper functions ===
def categorize_time(t):
    if 0 <= t < 600:
        return "Night"
    elif 600 <= t < 1200:
        return "Morning"
    elif 1200 <= t < 1800:
        return "Afternoon"
    else:
        return "Evening"

def categorize_duration(d):
    if d < 120:
        return "Short"
    elif d < 300:
        return "Medium"
    else:
        return "Long"

# === Main App Function ===
def run_ml_app():
    st.subheader("🔍 Flight Delay Prediction")

    # === Input UI sederhana ===
    airline = st.text_input("Airline (kode, contoh: AA)")
    airport_from = st.text_input("Airport From (contoh: LAX)")
    airport_to = st.text_input("Airport To (contoh: DFW)")
    day_of_week = st.number_input("Day of Week (1=Mon, 7=Sun)", min_value=1, max_value=7, value=1)
    time = st.number_input("Departure Time (minutes of day, 0–2359)", min_value=0, max_value=2359, value=600)
    length = st.number_input("Flight Length (minutes)", min_value=30, max_value=1000, value=120)

    if st.button("Prediksi"):
        try:
            # === Feature engineering ===
            dep_time_cat = categorize_time(time)
            duration_cat = categorize_duration(length)

            # === Lookup ===
            airline_row = airline_stats[airline_stats["Airline"] == airline]
            route_row = route_stats[(route_stats["AirportFrom"] == airport_from) & (route_stats["AirportTo"] == airport_to)]
            time_row = time_stats[time_stats["DepTimeCategory"] == dep_time_cat]
            duration_row = duration_stats[duration_stats["Duration_cat"] == duration_cat]
            arrv_row = arrv_per_day[(arrv_per_day["AirportTo"] == airport_to) & (arrv_per_day["DayOfWeek"] == day_of_week)]
            dept_row = dept_per_day[(dept_per_day["AirportFrom"] == airport_from) & (dept_per_day["DayOfWeek"] == day_of_week)]

            # === Build feature dict (16 kolom sesuai features_order.json) ===
            feat = {
                "Airline": airline,
                "DayOfWeek": day_of_week,
                "Time": time,
                "Length": length,
                "DepTimeCategory": dep_time_cat,
                "Duration_cat": duration_cat,
                "DelayRate_Airline": airline_row["DelayRate_Airline"].values[0] if not airline_row.empty else 0,
                "FreqRate_Airline": airline_row["FreqRate_Airline"].values[0] if not airline_row.empty else 0,
                "DelayRate_Route": route_row["DelayRate_Route"].values[0] if not route_row.empty else 0,
                "FreqRate_Route": route_row["FreqRate_Route"].values[0] if not route_row.empty else 0,
                "DelayRate_DeptTime": time_row["DelayRate_DeptTime"].values[0] if not time_row.empty else 0,
                "FreqRate_DeptTime": time_row["FreqRate_DeptTime"].values[0] if not time_row.empty else 0,
                "DelayRate_Duration": duration_row["DelayRate_Duration"].values[0] if not duration_row.empty else 0,
                "FreqRate_Duration": duration_row["FreqRate_Duration"].values[0] if not duration_row.empty else 0,
                "Arrv_perDay": arrv_row["Arrv_perDay"].values[0] if not arrv_row.empty else 0,
                "Dept_perDay": dept_row["Dept_perDay"].values[0] if not dept_row.empty else 0,
            }

            # === Buat dataframe sesuai urutan fitur ===
            input_df = pd.DataFrame([[feat[f] for f in features_order]], columns=features_order)

            # === Prediksi pakai pipeline ===
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            # === Output ===
            if prediction == 1:
                st.error(f"⚠️ Prediksi: **DELAY** (Probabilitas {prob:.2f})")
            else:
                st.success(f"✅ Prediksi: **TEPAT WAKTU** (Probabilitas {prob:.2f})")

        except Exception as e:
            st.warning(f"Terjadi error: {e}")
