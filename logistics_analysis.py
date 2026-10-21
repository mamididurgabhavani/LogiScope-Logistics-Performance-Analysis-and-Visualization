"""Week 3 - Advanced Data Analysis and Visualization in Logistics
Simulates a logistics dataset, performs EDA, and produces visualizations."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", context="notebook")
OUT = "charts/"

# ------------------------------------------------------------------
# 1. DATA SIMULATION
# ------------------------------------------------------------------
N = 1500
dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D")
hubs = np.random.choice(["Hyderabad", "Mumbai", "Delhi", "Chennai", "Kolkata"], N,
                        p=[0.22, 0.28, 0.25, 0.15, 0.10])
modes = np.random.choice(["Road", "Rail", "Air", "Sea"], N, p=[0.55, 0.20, 0.15, 0.10])
carriers = np.random.choice(["Carrier A", "Carrier B", "Carrier C", "Carrier D"], N,
                            p=[0.35, 0.30, 0.20, 0.15])
priority = np.random.choice(["Standard", "Express"], N, p=[0.75, 0.25])

mode_dist = {"Road": (50, 1800), "Rail": (300, 2500), "Air": (500, 2800), "Sea": (800, 3000)}
distance = np.array([np.random.uniform(*mode_dist[m]) for m in modes])
weight = np.random.lognormal(mean=6.0, sigma=0.8, size=N).clip(20, 5000)        # kg
volume = (weight / np.random.uniform(120, 250, N)).clip(0.1, 40)                # m3

speed = {"Road": 55, "Rail": 45, "Air": 600, "Sea": 30}  # effective km/h
transit_h = np.array([d / speed[m] for d, m in zip(distance, modes)])
planned_days = np.ceil(transit_h / 24 + {"Road": 0.5, "Rail": 1, "Air": 0.5, "Sea": 2}["Road"]).clip(1, None)
planned_days = np.array([np.ceil(t / 24 + {"Road": 0.5, "Rail": 1.0, "Air": 0.5, "Sea": 2.0}[m])
                         for t, m in zip(transit_h, modes)])

# Delay drivers: carrier, hub congestion, peak season, Express handling
carrier_delay = {"Carrier A": 0.2, "Carrier B": 0.5, "Carrier C": 1.2, "Carrier D": 0.35}
hub_delay = {"Hyderabad": 0.2, "Mumbai": 0.9, "Delhi": 0.7, "Chennai": 0.3, "Kolkata": 0.5}
month = dates.month.values
peak = np.isin(month, [10, 11, 12]).astype(float)
delay = (np.array([carrier_delay[c] for c in carriers])
         + np.array([hub_delay[h] for h in hubs])
         + 0.9 * peak
         - 0.3 * (priority == "Express")
         + np.random.gamma(1.5, 0.5, N) - 1.0)
delay = np.where(delay < 0, 0, delay)
actual_days = np.round(planned_days + delay, 1)

rate = {"Road": 0.9, "Rail": 0.55, "Air": 6.5, "Sea": 0.35}  # USD per kg-100km base
cost = np.array([(w / 100) * (d / 100) * rate[m] for w, d, m in zip(weight, distance, modes)])
cost = cost * np.random.normal(1.0, 0.08, N) + 25
cost = cost * np.where(priority == "Express", 1.35, 1.0) * (1 + 0.12 * peak)
cost = cost.round(2)

df = pd.DataFrame({
    "shipment_id": [f"SH{i:05d}" for i in range(1, N + 1)],
    "ship_date": dates, "origin_hub": hubs, "transport_mode": modes, "carrier": carriers,
    "priority": priority, "distance_km": distance.round(1), "weight_kg": weight.round(1),
    "volume_m3": volume.round(2), "planned_days": planned_days, "actual_days": actual_days,
    "cost_usd": cost,
})
df["delay_days"] = (df.actual_days - df.planned_days).round(1)
df["on_time"] = df.delay_days <= 1.0
df["cost_per_kg"] = df.cost_usd / df.weight_kg
df["month"] = df.ship_date.dt.month
df["month_name"] = df.ship_date.dt.strftime("%b")
df["weekday"] = df.ship_date.dt.day_name()
df.to_csv("logistics_data.csv", index=False)

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
stats = {}
num = ["distance_km", "weight_kg", "volume_m3", "planned_days", "actual_days",
       "delay_days", "cost_usd", "cost_per_kg"]
desc = df[num].describe().T
desc["median"] = df[num].median()
desc["skew"] = df[num].skew()
stats["describe"] = desc.round(2).reset_index().rename(columns={"index": "variable"}).to_dict("records")
stats["n"] = N
stats["missing"] = int(df.isna().sum().sum())
stats["on_time_rate"] = round(df.on_time.mean() * 100, 1)
stats["mean_delay"] = round(df.delay_days.mean(), 2)
stats["total_cost"] = round(df.cost_usd.sum(), 0)

corr = df[["distance_km", "weight_kg", "volume_m3", "actual_days", "delay_days", "cost_usd"]].corr()
stats["corr"] = corr.round(2).to_dict()

mode_tbl = df.groupby("transport_mode").agg(
    shipments=("shipment_id", "count"), avg_days=("actual_days", "mean"),
    avg_cost=("cost_usd", "mean"), cost_per_kg=("cost_per_kg", "mean"),
    on_time=("on_time", "mean")).round(2)
mode_tbl["on_time"] = (mode_tbl.on_time * 100).round(1)
stats["mode"] = mode_tbl.reset_index().to_dict("records")

carr_tbl = df.groupby("carrier").agg(
    shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
    on_time=("on_time", "mean"), avg_cost=("cost_usd", "mean")).round(2)
carr_tbl["on_time"] = (carr_tbl.on_time * 100).round(1)
stats["carrier"] = carr_tbl.reset_index().to_dict("records")

hub_tbl = df.groupby("origin_hub").agg(
    shipments=("shipment_id", "count"), avg_delay=("delay_days", "mean"),
    on_time=("on_time", "mean")).round(2)
hub_tbl["on_time"] = (hub_tbl.on_time * 100).round(1)
stats["hub"] = hub_tbl.reset_index().to_dict("records")

peak_df = df.assign(season=np.where(df.month.isin([10, 11, 12]), "Peak (Oct-Dec)", "Off-peak"))
stats["season"] = peak_df.groupby("season").agg(
    avg_delay=("delay_days", "mean"), on_time=("on_time", "mean"),
    avg_cost=("cost_usd", "mean")).round(2).reset_index().to_dict("records")
stats["priority"] = df.groupby("priority").agg(
    avg_delay=("delay_days", "mean"), avg_cost=("cost_usd", "mean"),
    on_time=("on_time", "mean")).round(2).reset_index().to_dict("records")

# Cost regression (distance*weight interaction)
x = (df.distance_km * df.weight_kg) / 1e5
slope, intercept = np.polyfit(x, df.cost_usd, 1)
r2 = np.corrcoef(x, df.cost_usd)[0, 1] ** 2
stats["reg"] = {"slope": round(slope, 2), "intercept": round(intercept, 2), "r2": round(r2, 3)}
stats["head"] = df.head(5)[["shipment_id", "origin_hub", "transport_mode", "carrier", "distance_km",
                            "weight_kg", "planned_days", "actual_days", "cost_usd"]].to_dict("records")
stats["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

# ------------------------------------------------------------------
# 3. VISUALIZATIONS
# ------------------------------------------------------------------
# Fig 1: delivery-time distribution
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(df.actual_days, bins=30, kde=True, color="#2E6F9E", ax=ax)
ax.axvline(df.actual_days.mean(), color="#C0392B", ls="--", label=f"Mean = {df.actual_days.mean():.1f} d")
ax.axvline(df.actual_days.median(), color="#27AE60", ls=":", label=f"Median = {df.actual_days.median():.1f} d")
ax.set(title="Distribution of Actual Delivery Time", xlabel="Delivery time (days)", ylabel="Shipments")
ax.legend(); fig.tight_layout(); fig.savefig(OUT + "fig1_hist.png", dpi=160); plt.close(fig)

# Fig 2: box plot delivery days & cost by mode
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
order = ["Road", "Rail", "Air", "Sea"]
sns.boxplot(data=df, x="transport_mode", y="actual_days", order=order, palette="Blues", ax=axes[0])
axes[0].set(title="Delivery Time by Transport Mode", xlabel="Mode", ylabel="Days")
sns.boxplot(data=df, x="transport_mode", y="cost_per_kg", order=order, palette="Oranges", ax=axes[1])
axes[1].set(title="Cost per kg by Transport Mode", xlabel="Mode", ylabel="USD / kg", yscale="log")
fig.tight_layout(); fig.savefig(OUT + "fig2_box.png", dpi=160); plt.close(fig)

# Fig 3: correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlation Matrix of Key Logistics Variables")
fig.tight_layout(); fig.savefig(OUT + "fig3_corr.png", dpi=160); plt.close(fig)

# Fig 4: scatter cost vs distance, coloured by mode, size by weight
fig, ax = plt.subplots(figsize=(8.5, 5))
sns.scatterplot(data=df, x="distance_km", y="cost_usd", hue="transport_mode", size="weight_kg",
                sizes=(10, 150), alpha=0.6, hue_order=order, ax=ax)
ax.set(title="Transportation Cost vs Distance (bubble size = weight)", xlabel="Distance (km)",
       ylabel="Cost (USD)", yscale="log")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig4_scatter.png", dpi=160); plt.close(fig)

# Fig 5: monthly trend (volume of shipments, avg delay)
monthly = df.groupby("month").agg(shipments=("shipment_id", "count"),
                                  avg_delay=("delay_days", "mean"),
                                  avg_cost=("cost_usd", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.bar(monthly.month, monthly.shipments, color="#BFD7EA", label="Shipments")
ax1.set(xlabel="Month (2025)", ylabel="Shipments", title="Monthly Shipment Volume and Average Delay")
ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.plot(monthly.month, monthly.avg_delay, color="#C0392B", marker="o", lw=2, label="Avg delay (days)")
ax2.set_ylabel("Avg delay (days)"); ax2.grid(False)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig5_trend.png", dpi=160); plt.close(fig)

# Fig 6: on-time rate by carrier
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ct = carr_tbl.reset_index().sort_values("on_time")
bars = ax.barh(ct.carrier, ct.on_time, color=sns.color_palette("viridis", 4))
for b, v in zip(bars, ct.on_time):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center")
ax.axvline(df.on_time.mean() * 100, color="grey", ls="--", label="Network average")
ax.set(title="On-Time Delivery Rate by Carrier", xlabel="On-time rate (%)", xlim=(0, 100))
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(OUT + "fig6_carrier.png", dpi=160); plt.close(fig)

# Fig 7: bottleneck heatmap hub x month
pv = df.pivot_table(index="origin_hub", columns="month", values="delay_days", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(pv, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={"label": "Avg delay (days)"}, ax=ax)
ax.set(title="Bottleneck Map: Average Delay by Hub and Month", xlabel="Month", ylabel="Origin hub")
fig.tight_layout(); fig.savefig(OUT + "fig7_heat.png", dpi=160); plt.close(fig)

# Fig 8: pairplot-lite: cost per kg vs weight (economies of scale)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.regplot(data=df, x="weight_kg", y="cost_per_kg", scatter_kws={"alpha": 0.3, "s": 14},
            line_kws={"color": "#C0392B"}, logx=True, ax=ax)
ax.set(title="Economies of Scale: Cost per kg vs Shipment Weight", xlabel="Weight (kg)",
       ylabel="USD / kg", xscale="log", yscale="log")
fig.tight_layout(); fig.savefig(OUT + "fig8_scale.png", dpi=160); plt.close(fig)

stats["monthly"] = monthly.round(2).to_dict("records")
stats["worst_cell"] = {
    "hub": pv.stack().idxmax()[0], "month": int(pv.stack().idxmax()[1]),
    "delay": round(pv.stack().max(), 2)}
stats["scale_corr"] = round(np.corrcoef(np.log(df.weight_kg), np.log(df.cost_per_kg))[0, 1], 2)
with open("stats.json", "w") as f:
    json.dump(stats, f, indent=1, default=str)
print("done")
