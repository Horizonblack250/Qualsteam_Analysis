import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ======================
# Paths (relative to repo root)
# ======================
DATA_PATH = "data/df_clean.csv"
BATCH_SUMMARY_PATH = "data/batch_summary_qualsteam.csv"

# ======================
# Helper: duration formatting
# ======================
def format_duration(start_time, end_time):
    delta = end_time - start_time
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes >= 60:
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours} h {minutes} min"
    else:
        return f"{total_minutes} min"

# ======================
# Load data (cached)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    batch_df = pd.read_csv(BATCH_SUMMARY_PATH)
    batch_df["start_time"] = pd.to_datetime(batch_df["start_time"], errors="coerce")
    batch_df["end_time"] = pd.to_datetime(batch_df["end_time"], errors="coerce")
    batch_df = batch_df.sort_values("start_time").reset_index(drop=True)

    return df, batch_df

df, batch_df = load_data()

# ======================
# Streamlit Layout
# ======================
st.set_page_config(
    page_title="QualSteam – Real Dairy Dashboard",
    layout="wide"
)

st.title("QualSteam Performance – Real Dairy")

st.markdown(
    """
    Interactive dashboard to explore **QualSteam** batches:
    - Temperature tracking vs setpoint  
    - Pressure profile (P1, P2, pressure setpoint)  
    - Steam flow rate  
    - QualSteam valve opening  
    """
)

# Sidebar: batch selection
st.sidebar.header("Batch Selection")

if batch_df.empty:
    st.error("No batches found in `batch_summary_qualsteam.csv`.")
    st.stop()

batch_ids = batch_df["batch_id"].tolist()
selected_batch_id = st.sidebar.selectbox("Select Batch ID", batch_ids, index=0)

binfo = batch_df.loc[batch_df["batch_id"] == selected_batch_id].iloc[0]
start = binfo["start_time"]
end = binfo["end_time"]
duration_str = format_duration(start, end)

# Sidebar batch info
st.sidebar.subheader("Selected Batch Info")
st.sidebar.write(f"**Batch ID:** {int(binfo['batch_id'])}")
st.sidebar.write(f"**Start:** {start}")
st.sidebar.write(f"**End:** {end}")
st.sidebar.write(f"**Duration:** {duration_str}")
if "process_temp_sp_median" in binfo:
    st.sidebar.write(f"**Temp SP (median):** {binfo['process_temp_sp_median']:.1f} °C")
if "pressure_sp_median" in binfo and not pd.isna(binfo["pressure_sp_median"]):
    st.sidebar.write(f"**Pressure SP (median):** {binfo['pressure_sp_median']:.2f} barg")

# Main: show batch summary table
with st.expander("Show all detected batches"):
    show_cols = [c for c in batch_df.columns if c not in ("core_start_time", "core_end_time")]
    st.dataframe(batch_df[show_cols], use_container_width=True)

# ======================
# Filter data for selected batch
# ======================
mask = (df["Timestamp"] >= start) & (df["Timestamp"] <= end)
db = df.loc[mask].copy()

if db.empty:
    st.error("No data found for this batch window in df_clean.csv.")
    st.stop()

st.subheader(
    f"Batch {int(binfo['batch_id'])} – Duration: {duration_str} "
    f"(Site: Real Dairy, Valve: QualSteam)"
)

# ======================
# Plot – same style as before
# ======================
plt.style.use("default")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
ax_temp, ax_press, ax_flow, ax_valve = axes

# 1) Process Temp vs SP
ax_temp.plot(db["Timestamp"], db["Process Temp"],
             color="red", label="Process Temp")
ax_temp.fill_between(db["Timestamp"], db["Process Temp"],
                     alpha=0.20, color="red")
ax_temp.plot(db["Timestamp"], db["Process Temp SP"],
             color="black", linestyle="--", label="Process Temp SP")
ax_temp.set_ylabel("Temp (°C)")
ax_temp.legend(loc="upper right")

# 2) Pressures: P2, P2 SP, P1
ax_press.plot(db["Timestamp"], db["Outlet Steam Pressure"],
              color="blue", label="Outlet Pressure P2")
ax_press.fill_between(db["Timestamp"], db["Outlet Steam Pressure"],
                      alpha=0.20, color="blue")

if "Pressure SP" in db.columns:
    ax_press.plot(db["Timestamp"], db["Pressure SP"],
                  color="green", linestyle="--", label="Outlet Pressure SP")

ax_press.plot(db["Timestamp"], db["Inlet Steam Pressure"],
              color="cyan", linestyle="-.", label="Inlet Pressure P1")
ax_press.set_ylabel("Pressure (barg)")
ax_press.legend(loc="upper right")

# 3) Steam Flow Rate
ax_flow.plot(db["Timestamp"], db["Steam Flow Rate"],
             color="purple", label="Steam Flow Rate")
ax_flow.fill_between(db["Timestamp"], db["Steam Flow Rate"],
                     alpha=0.20, color="purple")
ax_flow.set_ylabel("Flow (kg/h)")
ax_flow.legend(loc="upper right")

# 4) Valve Opening
ax_valve.plot(db["Timestamp"], db["QualSteam Valve Opening"],
              color="orange", label="QualSteam Valve Opening")
ax_valve.fill_between(db["Timestamp"], db["QualSteam Valve Opening"],
                      alpha=0.20, color="orange")
ax_valve.set_ylabel("Valve Opening (%)")
ax_valve.set_xlabel("Timestamp")
ax_valve.legend(loc="upper right")

# Common formatting
for ax in axes:
    ax.grid(True, linestyle=":", alpha=0.4)

date_format = DateFormatter("%Y-%m-%d\n%H:%M")
ax_valve.xaxis.set_major_formatter(date_format)

fig.suptitle(
    f"QualSteam – Real Dairy – Batch {int(binfo['batch_id'])} | Duration: {duration_str}",
    fontsize=13,
    y=0.98
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

st.pyplot(fig)
