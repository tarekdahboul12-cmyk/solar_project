import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Lebanon Solar Forecast", page_icon="☀️")

TOWNS = {
    "Anjar": (33.739, 35.93),
    "Batroun": (34.255, 35.659),
    "Beirut-RaficHaririAirport.csv": (33.834, 35.496),
    "Bekaa": (33.675, 35.833),
    "Daraya": (33.64, 35.506),
    "Faraya": (34.026, 35.838),
    "Port-Of-Beirut": (33.903, 35.525),
    "Port-Of-Tripoli": (34.451, 35.828),
    "ReneMouawad-Airport.csv": (34.591, 36.011),
    "baalbek": (34.01, 36.212),
    "bcharre": (34.252, 36.042),
    "beirut": (33.896, 35.478),
    "chehime": (33.624, 35.484),
    "debbieh": (33.666, 35.487),
    "halba": (34.544, 36.078),
    "jbeil": (34.127, 35.822),
    "mountlebanon": (33.737, 35.6),
    "nabatieh": (33.284, 35.49),
    "saida": (33.556, 35.395),
    "tripoli": (34.437, 35.835),
    "tyre": (33.272, 35.196),
    "zahle": (33.848, 35.903),
}


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solar_model.joblib")
    return joblib.load(path)


@st.cache_data(ttl=3600)
def get_forecast(lat, lon, tilt):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
        "latitude": lat, "longitude": lon, "timezone": "GMT", "forecast_days": 3,
        "hourly": "temperature_2m,wind_speed_10m,global_tilted_irradiance",
        "tilt": tilt, "azimuth": 0, "wind_speed_unit": "ms"})
    r.raise_for_status()
    js = r.json()
    h = js["hourly"]
    fc = pd.DataFrame({
        "time": pd.to_datetime(h["time"]).tz_localize("UTC") - pd.Timedelta(minutes=30),
        "G": h["global_tilted_irradiance"],
        "T2m": h["temperature_2m"],
        "WS10m": h["wind_speed_10m"],
    }).dropna()
    fc[["G", "T2m", "WS10m"]] = fc[["G", "T2m", "WS10m"]].astype(float)
    fc["lat"], fc["lon"], fc["elevation"] = lat, lon, js["elevation"]
    return fc


def add_features(fc):
    fc = fc.copy()
    solar_hour = (fc["time"].dt.hour + fc["time"].dt.minute / 60 + fc["lon"] / 15) % 24
    doy = fc["time"].dt.dayofyear
    fc["hour_sin"] = np.sin(2 * np.pi * solar_hour / 24)
    fc["hour_cos"] = np.cos(2 * np.pi * solar_hour / 24)
    fc["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    fc["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (solar_hour - 12))
    lat_r = np.radians(fc["lat"])
    fc["sun_elev"] = np.degrees(np.arcsin(
        np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(hour_angle)))
    return fc


bundle = load_model()
model, mu, sd = bundle["model"], bundle["mu"], bundle["sd"]
FEATURES, TILT = bundle["features"], bundle["tilt"]

st.title("☀️ Lebanon Solar Forecast")
st.write("How much will your solar panels produce? Set up your system on the left.")

# ---------------- inputs (sidebar) ----------------
with st.sidebar:
    st.header("Your system")
    town = st.selectbox("Your area", list(TOWNS))
    n_panels = st.number_input("Number of panels", min_value=1, max_value=200, value=10, step=1)
    panel_w = st.number_input("Panel power (W), written on the panel label",
                              min_value=100, max_value=800, value=550, step=5)
    system_kw = n_panels * panel_w / 1000
    st.write(f"Your system: **{system_kw:.2f} kW**")
    day = st.radio("Day", ["Today", "Tomorrow"], index=1, horizontal=True)
    factor = st.slider("Your system's real performance (%)", 50, 100, 100,
                       help="Dust, shade or a bad angle make real systems produce less. "
                            "If your inverter app shows less than this forecast, lower this value.")

# ---------------- forecast ----------------
lat, lon = TOWNS[town]
try:
    fc = get_forecast(lat, lon, TILT)
except Exception as e:
    st.error(f"Could not get the weather forecast right now. Try again later. ({e})")
    st.stop()

now = pd.Timestamp.now(tz="Asia/Beirut")
target_date = (now + pd.Timedelta(days=1 if day == "Tomorrow" else 0)).date()
fc["local"] = fc["time"].dt.tz_convert("Asia/Beirut")
fc = fc[fc["local"].dt.date == target_date].reset_index(drop=True)
if fc.empty:
    st.error("No forecast available for this day.")
    st.stop()

fc = add_features(fc)
X = (fc[FEATURES].values - mu) / sd
fc["kW"] = np.clip(model.predict(X), 0, 1) * system_kw * factor / 100
fc.loc[fc["G"] <= 0, "kW"] = 0.0

total_kwh = fc["kW"].sum()
start = fc["kW"].rolling(3).sum().shift(-2).idxmax()
start_hour = int(fc["local"].dt.hour[start])
sun_hours = total_kwh / system_kw

# ---------------- tabs ----------------
tab1, tab2 = st.tabs(["☀️ Forecast", "🔌 Appliance planner"])

with tab1:
    c1, c2 = st.columns(2)
    c1.metric("Expected energy", f"{total_kwh:.1f} kWh")
    c2.metric("Best 3 hours", f"{start_hour}:00 – {start_hour + 3}:00")

    if sun_hours >= 4.5:
        st.success("Strong day: run the washing machine and water heater in the best hours.")
    elif sun_hours >= 2.5:
        st.info("Average day: run heavy appliances one at a time in the best hours.")
    else:
        st.warning("Weak day: save your battery and postpone heavy loads if you can.")

    chart = pd.DataFrame({"hour": fc["local"].dt.hour, "kW": fc["kW"]}).set_index("hour")
    st.subheader(f"Hour by hour, {target_date}")
    st.bar_chart(chart)

    st.caption("This is an estimate based on the weather forecast and a machine learning model trained on "
               "PVGIS data for Lebanon. Real output depends on your panels, shading, dust and battery.")

with tab2:
    st.subheader("Your appliances")
    st.write("Add the heavy appliances you want to run today. You can edit, add or delete rows.")

    default_appliances = pd.DataFrame({
        "Appliance": ["Washing machine", "Water heater", "Dishwasher", "Iron"],
        "Power (kW)": [2.0, 1.5, 1.8, 1.2],
        "Hours": [2, 3, 1, 1],
    })
    appliances = st.data_editor(
        default_appliances,
        num_rows="dynamic",
        
        column_config={
            "Appliance": st.column_config.TextColumn(required=True),
            "Power (kW)": st.column_config.NumberColumn(min_value=0.1, max_value=10.0, step=0.1, required=True),
            "Hours": st.column_config.NumberColumn(min_value=1, max_value=12, step=1, required=True),
        },
        key="appliances",
    )
    appliances = appliances.dropna()

    st.subheader("Your house and battery")
    c1, c2, c3 = st.columns(3)
    base_kw = c1.number_input("Always-on load (kW)", min_value=0.0, max_value=5.0, value=0.4, step=0.1,
                              help="Fridge, lights, router, TV... what the house uses all the time.")
    battery_kwh = c2.number_input("Battery size (kWh)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    battery_start = c3.slider("Battery at midnight (%)", 0, 100, 50)

    st.write(f"**{len(appliances)}** appliances, using "
             f"**{(appliances['Power (kW)'] * appliances['Hours']).sum():.1f} kWh** in total.")
        # ---------------- step 3: schedule the appliances ----------------
    hours = np.arange(24)
    solar = np.zeros(24)
    solar[fc["local"].dt.hour.values] = fc["kW"].values      # solar kW for each hour of the day
    load = np.full(24, base_kw)                               # always-on load

    plan = []
    for _, a in appliances.sort_values("Power (kW)", ascending=False).iterrows():
        p, h = float(a["Power (kW)"]), int(a["Hours"])
        best_start, best_score = 0, -np.inf
        for s in range(0, 24 - h + 1):
            surplus = solar[s:s + h] - load[s:s + h]            # sun left after what is already running
            covered = np.minimum(np.clip(surplus, 0, None), p).sum()
            score = covered + 0.001 * surplus.sum()
            if score > best_score:
                best_start, best_score = s, score
        load[best_start:best_start + h] += p
        plan.append({"Appliance": a["Appliance"], "Start": f"{best_start}:00",
                     "End": f"{best_start + h}:00", "Sun covers": f"{100 * best_score / (p * h):.0f}%"})

    # ---------------- step 4: simulate the battery hour by hour ----------------
    soc = battery_kwh * battery_start / 100
    soc_hist, gen_kwh, gen_hours = [], 0.0, []
    for t in hours:
        net = solar[t] - load[t]
        if net >= 0:
            soc = min(battery_kwh, soc + net)                  # extra sun charges the battery
        else:
            take = min(-net, soc)
            soc -= take
            if -net - take > 1e-6:                             # battery empty -> generator needed
                gen_kwh += -net - take
                gen_hours.append(int(t))
        soc_hist.append(100 * soc / battery_kwh if battery_kwh > 0 else 0)

    night_hours = soc / base_kw if base_kw > 0 else 99
    lasts_until = int((24 + night_hours) % 24) if night_hours < 24 else None

    # ---------------- step 5: show the result ----------------
    st.subheader("Your plan for " + str(target_date))
    st.dataframe(pd.DataFrame(plan).set_index("Appliance"))

    m1, m2, m3 = st.columns(3)
    m1.metric("Battery at midnight tonight", f"{soc_hist[-1]:.0f}%")
    m2.metric("Battery lasts until", f"{lasts_until}:00" if lasts_until is not None else "Next day")
    m3.metric("Generator needed today", f"{len(gen_hours)} h")

    if gen_hours:
        st.warning("You will need the generator at: " + ", ".join(f"{t}:00" for t in gen_hours))
    else:
        st.success("No generator needed today with this plan.")

    st.write("**Sun vs. house use (kW)**")
    st.bar_chart(pd.DataFrame({"Solar": solar, "House use": load}, index=hours))
    st.write("**Battery level (%)**")
    st.line_chart(pd.DataFrame({"Battery %": soc_hist}, index=hours))