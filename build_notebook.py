"""Builds and executes flight_price_prediction.ipynb (run: python build_notebook.py)."""
import sys
sys.path.insert(0, ".")
from nbrun import Notebook

nb = Notebook(".")

nb.md("""
# Flight Price Prediction

**Business question:** what drives the price of a domestic flight ticket, and can we predict it well enough to
guide travellers ("book now or wait?") and pricing teams?

This notebook rebuilds the Final Project of the Digital Skola Data Science Bootcamp (team Yggdrasil / Kel 2), where
I worked on data cleaning, EDA, feature engineering, modelling, evaluation and the presentation.

**Data:** Airlines Flights Data (public), 300,153 one-way flights between six major Indian cities, booked
1–49 days before departure (`data/airlines_flights_data.csv.gz`). Prices are in Indian rupees (INR).
""")

nb.code("""
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
""")

nb.md("## 1. Load and check the data")
nb.code("""
raw = pd.read_csv("data/airlines_flights_data.csv.gz")
print(raw.shape, "| missing values:", raw.isna().sum().sum(), "| duplicate rows:", raw.duplicated().sum())
raw.head()
""")

nb.md("""
## 2. Clean and engineer features

* Drop `index` (row number) and `flight` (1,561 flight codes, too specific to generalise and impractical for users to enter).
* Add `route` = source city + destination city, because price depends on the city pair, not on each city alone.
* Remove rows that become exact duplicates after dropping those columns.
""")
nb.code("""
df = raw.drop(columns=["index", "flight"])
df["route"] = df["source_city"] + " - " + df["destination_city"]
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before - len(df):,} duplicates -> {len(df):,} rows")
cat_cols = ["airline", "source_city", "departure_time", "stops", "arrival_time",
            "destination_city", "class", "route"]
num_cols = ["duration", "days_left"]
df[cat_cols].nunique()
""")

nb.md("## 3. What drives the price?")
nb.code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.histplot(df["price"], bins=60, ax=axes[0], color="#4C72B0")
axes[0].set(title="Ticket price distribution", xlabel="Price (INR)")
cls = df.groupby("class")["price"].median()
cls.plot(kind="bar", ax=axes[1], color=["#C44E52", "#4C72B0"], rot=0)
axes[1].set(title="Median price by class", ylabel="INR")
for i, v in enumerate(cls.values):
    axes[1].text(i, v * 1.02, f"{v:,.0f}", ha="center")
air = df[df["class"] == "Economy"].groupby("airline")["price"].median().sort_values()
air.plot(kind="barh", ax=axes[2], color="#4C72B0")
axes[2].set(title="Median ECONOMY price by airline", xlabel="INR", ylabel="")
plt.tight_layout()
print((df.groupby("class")["price"].agg(["count", "median", "mean"])).round(0).to_string())
""")
nb.md("""
Prices form two separate markets. **the median Business ticket costs about 9 times the median Economy ticket**, so `class` will
dominate any model. Within Economy, airlines still differ clearly, with Vistara and Air India priced above the low-cost carriers.
""")

nb.code("""
booking = (df.groupby(["class", "days_left"])["price"].median().reset_index())
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, c in zip(axes, ["Economy", "Business"]):
    sub = booking[booking["class"] == c]
    ax.plot(sub["days_left"], sub["price"], marker=".", color="#4C72B0")
    ax.set(title=f"{c}: median price vs days before departure", xlabel="Days left", ylabel="INR")
    ax.invert_xaxis()
plt.tight_layout()

eco = df[df["class"] == "Economy"]
win = eco.groupby(pd.cut(eco["days_left"], [0, 2, 7, 14, 21, 49],
                         labels=["1-2 days", "3-7 days", "8-14 days", "15-21 days", "22-49 days"]),
                  observed=True)["price"].median()
print("Economy median price by booking window (INR):")
print(win.round(0).to_string())
print(f"\\nBooking 22-49 days ahead vs 1-2 days ahead: {1 - win.iloc[-1] / win.iloc[0]:.0%} cheaper")
""")
nb.code("""
stops = df.groupby(["class", "stops"])["price"].median().unstack(0).round(0)
print("Median price by number of stops (INR):")
print(stops.to_string())
print("\\nCorrelation with price:")
print(df[["duration", "days_left", "price"]].corr()["price"].round(2).to_string())
""")
nb.md("""
**Findings**

1. **Booking window matters most for Economy travellers.** Prices climb sharply in the last two weeks before departure, and the last few days are by far the most expensive.
2. **Stops are not a discount.** One-stop flights are typically priced *above* non-stop flights, because they serve longer routes and different airlines.
3. **Duration has only a weak linear link to price (r = 0.20)**, so a non-linear model is needed to combine class, airline, route and timing.
""")

nb.md("""
## 4. Model comparison

The data is split 80/20 into train and test sets. Four models are compared:

* **Ridge regression**: linear baseline with one-hot encoding.
* **Decision tree**: simple non-linear model.
* **Random forest**: an ensemble of trees. It is trained on a 60,000-row sample to keep the training time short.
* **Histogram gradient boosting**: scikit-learn's fast boosting model, which handles the categorical features natively.

The metrics are MAE and RMSE in rupees, R², and MAPE (average percentage error).
""")
nb.code("""
X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

onehot = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                            ("num", StandardScaler(), num_cols)])
ordinal = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                             ("num", "passthrough", num_cols)])

candidates = {
    "Ridge (baseline)": Pipeline([("prep", onehot), ("model", Ridge(alpha=1.0))]),
    "Decision Tree": Pipeline([("prep", ordinal), ("model", DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_STATE))]),
    "Random Forest (60k sample)": Pipeline([("prep", ordinal), ("model", RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))]),
    "Hist Gradient Boosting": Pipeline([("prep", ordinal), ("model", HistGradientBoostingRegressor(
        categorical_features=list(range(len(cat_cols))), max_iter=500, learning_rate=0.1,
        max_leaf_nodes=63, random_state=RANDOM_STATE))]),
}

def score(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": np.mean(np.abs(y_true - y_pred) / y_true) * 100}

rows, fitted = [], {}
sample_idx = X_train.sample(60000, random_state=RANDOM_STATE).index
for name, pipe in candidates.items():
    t0 = time.time()
    Xt, yt = (X_train.loc[sample_idx], y_train.loc[sample_idx]) if "sample" in name else (X_train, y_train)
    pipe.fit(Xt, yt)
    fitted[name] = pipe
    rows.append({"model": name, **score(y_test, pipe.predict(X_test)),
                 "train_R2": r2_score(yt, pipe.predict(Xt)), "train_seconds": time.time() - t0})
results = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
results.round({"MAE": 0, "RMSE": 0, "R2": 3, "MAPE_%": 1, "train_R2": 3, "train_seconds": 1})
""")
nb.md("""
**Histogram gradient boosting** has the lowest RMSE, the highest R² and the smallest train-test gap (0.987 vs 0.985), so it is the final model.
The decision tree has a lower MAE and MAPE, so it is very accurate on typical cheap tickets, but its RMSE is higher. That means it makes
more large misses, and its train-test gap is wider. For a pricing tool, avoiding large misses on expensive tickets matters more.
""")

nb.md("## 5. Final model: where is it accurate, where not?")
nb.code("""
best = fitted["Hist Gradient Boosting"]
pred = best.predict(X_test)
eval_df = X_test.assign(actual=y_test.values, pred=pred)
by_class = eval_df.groupby("class").apply(lambda g: pd.Series(score(g["actual"], g["pred"])), include_groups=False)
print(by_class.round({"MAE": 0, "RMSE": 0, "R2": 3, "MAPE_%": 1}).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
s = eval_df.sample(6000, random_state=RANDOM_STATE)
sns.scatterplot(data=s, x="actual", y="pred", hue="class", s=10, alpha=0.4, ax=axes[0])
lim = [0, s[["actual", "pred"]].max().max()]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set(title="Actual vs predicted price (test sample)", xlabel="Actual (INR)", ylabel="Predicted (INR)")
err = (eval_df["pred"] - eval_df["actual"]) / eval_df["actual"] * 100
sns.histplot(err.clip(-60, 60), bins=60, ax=axes[1], color="#4C72B0")
axes[1].set(title="Percentage error distribution", xlabel="Error (%)")
plt.tight_layout()
""")
nb.code("""
from sklearn.inspection import permutation_importance
sample = X_test.sample(5000, random_state=RANDOM_STATE)
imp = permutation_importance(best, sample, y_test.loc[sample.index], n_repeats=3,
                             random_state=RANDOM_STATE, scoring="r2")
importance = pd.Series(imp.importances_mean, index=sample.columns).sort_values()
ax = importance.plot(kind="barh", figsize=(7, 4), color="#4C72B0")
ax.set(title="Permutation importance (drop in R2 when a feature is shuffled)", xlabel="Drop in R2")
importance.sort_values(ascending=False).round(3)
""")

nb.md("## 6. Save the model for the Streamlit app")
nb.code("""
joblib.dump(best, "app/flight_price_model.joblib", compress=3)
options = {c: sorted(df[c].unique().tolist()) for c in ["airline", "source_city", "destination_city",
                                                        "departure_time", "arrival_time", "stops", "class"]}
route_duration = df.groupby(["source_city", "destination_city", "stops"])["duration"].median().round(2)
joblib.dump({"options": options, "route_duration": route_duration}, "app/app_metadata.joblib", compress=3)
final_scores = results.loc["Hist Gradient Boosting"]
print(f"Saved. Test R2 {final_scores['R2']:.3f}, MAE {final_scores['MAE']:,.0f} INR, MAPE {final_scores['MAPE_%']:.1f}%")
""")

nb.md("""
## 7. Conclusions and recommendations

* **For travellers:** book Economy at least two to three weeks ahead. Last-minute fares are much higher.
* **For pricing and revenue teams:** class, route, airline and booking window explain almost all of the price variation, and the model predicts prices with about 12% average error. It can be used to flag fares that look too high or too low for their route and timing.
* **Limitations:** the data covers one season and six cities in India. Business fares have a much higher absolute error in rupees (but a lower percentage error). Retrain on newer data before any production use.
""")

nb.save("flight_price_prediction.ipynb")
print("saved")
