"""Flight price estimator - Streamlit app."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

HERE = Path(__file__).parent
st.set_page_config(page_title="Flight Price Estimator", page_icon="✈️", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(HERE / "flight_price_model.joblib")
    meta = joblib.load(HERE / "app_metadata.joblib")
    return model, meta


model, meta = load()
opts, route_duration = meta["options"], meta["route_duration"]

st.title("✈️ Flight Price Estimator")
st.caption("Gradient boosting model trained on 300k domestic flights in India (test R² 0.985, ~12% average error). "
           "Prices in Indian rupees (INR).")

with st.sidebar:
    st.header("Your flight")
    source = st.selectbox("From", opts["source_city"], index=opts["source_city"].index("Delhi"))
    dest_options = [c for c in opts["destination_city"] if c != source]
    dest = st.selectbox("To", dest_options, index=dest_options.index("Mumbai") if "Mumbai" in dest_options else 0)
    airline = st.selectbox("Airline", opts["airline"], index=opts["airline"].index("Vistara"))
    cls = st.radio("Class", opts["class"], index=opts["class"].index("Economy"), horizontal=True)
    stops = st.selectbox("Stops", opts["stops"], index=opts["stops"].index("zero"))
    dep = st.selectbox("Departure time", opts["departure_time"], index=opts["departure_time"].index("Morning"))
    arr = st.selectbox("Arrival time", opts["arrival_time"], index=opts["arrival_time"].index("Afternoon"))
    days_left = st.slider("Days before departure", 1, 49, 7)

key = (source, dest, stops)
duration = float(route_duration.get(key, route_duration.loc[(source, dest)].median()
                                    if (source, dest) in route_duration.index.droplevel(2) else route_duration.median()))


def frame(days):
    return pd.DataFrame({
        "airline": airline, "source_city": source, "departure_time": dep, "stops": stops,
        "arrival_time": arr, "destination_city": dest, "class": cls, "duration": duration,
        "days_left": days, "route": f"{source} - {dest}"}, index=range(len(days)))


price = float(model.predict(frame([days_left]))[0])
curve = pd.DataFrame({"days_left": range(1, 50)})
curve["price"] = model.predict(frame(list(curve["days_left"])))
best = curve.loc[curve["price"].idxmin()]

c1, c2, c3 = st.columns(3)
c1.metric("Estimated price", f"₹{price:,.0f}")
c2.metric("Cheapest day to book", f"{int(best.days_left)} days before", f"₹{best.price:,.0f}")
c3.metric("You could save", f"₹{max(price - best.price, 0):,.0f}",
          f"{max(price - best.price, 0) / price:.0%} vs your date", delta_color="off")

fig = px.line(curve, x="days_left", y="price", markers=True,
              labels={"days_left": "Days before departure", "price": "Estimated price (INR)"},
              title=f"Estimated price by booking day: {source} → {dest}, {airline}, {cls}")
fig.update_xaxes(autorange="reversed")
fig.add_vline(x=days_left, line_dash="dash", line_color="red")
st.plotly_chart(fig, width="stretch")

st.info("Insight from the data: Economy fares typically jump about two weeks before departure. "
        "Booking 22–49 days ahead is about 65% cheaper than booking 1–2 days ahead.")
st.caption(f"Typical flight duration used for this route and number of stops: {duration:.1f} hours. "
           "Bootcamp final project (team Yggdrasil) rebuilt by Muhammad Irsyad Dzulfaqor · "
           "[GitHub](https://github.com/mirsyaddzulfaqor-tech/flight-price-prediction)")
