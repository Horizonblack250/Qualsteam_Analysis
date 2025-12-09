import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def format_duration_from_delta(delta: timedelta):
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
    if "core_start_time" in batch_df.columns:
        batch_df["core_start_time"] = pd.to_datetime(batch_df["core_start_time"], errors="coerce")
    if "core_end_time" in batch_df.columns:
        batch_df["core_end_time"] = pd.to_datetime(batch_df["core_end_time"], errors="coerce")

    # Remove batch 16 explicitly
    if "batch_id" in batch_df.columns:
        batch_df = batch_df[batch_df["batch_id"] != 16]

    batch_df = batch_df.sort_values("start_time").reset_index(drop=True)

    return df, batch_df


df, batch_df = load_data()

# ======================
# Streamlit Layout
# ======================
st.set_page_config(
    page_title="QualSteam – Real Dairy Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
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

# ======================
# Sidebar: batch selection
# ======================
st.sidebar.header("Batch Selection")

if batch_df.empty:
    st.error("No batches found in `batch_summary_qualsteam.csv` (after removing batch 16).")
    st.stop()

batch_ids = batch_df["batch_id"].tolist()
selected_batch_id = st.sidebar.selectbox("Select Batch ID", batch_ids, index=0)

binfo = batch_df.loc[batch_df["batch_id"] == selected_batch_id].iloc[0]
start = binfo["start_time"]
end = binfo["end_time"]
duration_str = format_duration(start, end)

core_start = binfo.get("core_start_time", pd.NaT)
core_end = binfo.get("core_end_time", pd.NaT)

# Sidebar batch info
st.sidebar.subheader("Selected Batch Info")
st.sidebar.write(f"**Batch ID:** {int(binfo['batch_id'])}")
st.sidebar.write(f"**Start:** {start}")
st.sidebar.write(f"**End:** {end}")
st.sidebar.write(f"**Duration:** {duration_str}")
if "process_temp_sp_median" in binfo and not pd.isna(binfo["process_temp_sp_median"]):
    st.sidebar.write(f"**Temp SP (median):** {binfo['process_temp_sp_median']:.1f} °C")
if "pressure_sp_median" in binfo and not pd.isna(binfo.get("pressure_sp_median", np.nan)):
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

# Core slice (for KPIs) – only if we have valid core times
core_db = None
if isinstance(core_start, pd.Timestamp) and isinstance(core_end, pd.Timestamp):
    core_mask = (df["Timestamp"] >= core_start) & (df["Timestamp"] <= core_end)
    core_db = df.loc[core_mask].copy()
    if core_db.empty:
        core_db = None

# ======================
# Compute KPIs (no valve KPIs)
# ======================
if core_db is not None:
    core_duration_str = format_duration(core_start, core_end)
else:
    core_duration_str = "N/A"

mean_abs_temp_dev = None
max_overshoot = None  # keep variable defined but we won't compute/display it

mean_flow_core = None
mean_abs_press_dev = None

if core_db is not None:
    if "Process Temp" in core_db.columns and "Process Temp SP" in core_db.columns:
        temp_diff = core_db["Process Temp"] - core_db["Process Temp SP"]
        mean_abs_temp_dev = temp_diff.abs().mean()
        # Max temp overshoot calculation intentionally disabled per request:
        # max_overshoot = temp_diff.clip(lower=0).max()
    if "Steam Flow Rate" in core_db.columns:
        mean_flow_core = core_db["Steam Flow Rate"].mean()
    if "Outlet Steam Pressure" in core_db.columns and "Pressure SP" in core_db.columns:
        press_diff = core_db["Outlet Steam Pressure"] - core_db["Pressure SP"]
        mean_abs_press_dev = press_diff.abs().mean()

# Estimated total steam during batch
total_steam_batch = None
if "Steam Flow Rate" in db.columns and len(db) > 1:
    db_sorted = db.sort_values("Timestamp")
    dt_hours = db_sorted["Timestamp"].diff().dt.total_seconds() / 3600.0
    flow_avg = (db_sorted["Steam Flow Rate"].shift(1) + db_sorted["Steam Flow Rate"]) / 2.0
    mass = (flow_avg * dt_hours).fillna(0.0)
    total_steam_batch = mass.sum()

# ======================
# Show KPIs
# ======================
st.subheader(
    f"Batch {int(binfo['batch_id'])} – Duration: {duration_str} "
    f"(Site: Real Dairy, Valve: QualSteam)"
)

kpi_row1 = st.columns(3)
kpi_row1[0].metric("Batch Duration", duration_str)
kpi_row1[1].metric("Core Duration", core_duration_str)
if total_steam_batch is not None:
    kpi_row1[2].metric("Estimated Total Steam (batch)", f"{total_steam_batch:,.0f} kg")
else:
    kpi_row1[2].metric("Estimated Total Steam (batch)", "N/A")

kpi_row2 = st.columns(3)
if mean_abs_temp_dev is not None:
    kpi_row2[0].metric("Mean |Temp - SP| (core)", f"{mean_abs_temp_dev:.2f} °C")
else:
    kpi_row2[0].metric("Mean |Temp - SP| (core)", "N/A")

# Max Temp Overshoot metric removed/commented out per request:
# if max_overshoot is not None:
#     kpi_row2[1].metric("Max Temp Overshoot (core)", f"{max_overshoot:.2f} °C")
# else:
#     kpi_row2[1].metric("Max Temp Overshoot (core)", "N/A")

if mean_flow_core is not None:
    kpi_row2[2].metric("Mean Steam Flow (core)", f"{mean_flow_core:.1f} kg/h")
else:
    kpi_row2[2].metric("Mean Steam Flow (core)", "N/A")

kpi_row3 = st.columns(3)
if mean_abs_press_dev is not None:
    kpi_row3[0].metric("Mean |P2 - P_SP| (core)", f"{mean_abs_press_dev:.3f} barg")
else:
    kpi_row3[0].metric("Mean |P2 - P_SP| (core)", "N/A")

# ======================
# Interactive Plot – same template & colors, Plotly
# ======================
fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=(
        "Process Temp vs SP",
        "Pressures (P2, P2 SP, P1)",
        "Steam Flow Rate",
        "QualSteam Valve Opening",
    ),
)

# 1) Process Temp & SP
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["Process Temp"],
        mode="lines",
        name="Process Temp",
        line=dict(color="red"),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.2)",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["Process Temp SP"],
        mode="lines",
        name="Process Temp SP",
        line=dict(color="#4A4A4A", dash="dash"),  # bright gold
    ),
    row=1,
    col=1,
)

# 2) Pressures
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["Outlet Steam Pressure"],
        mode="lines",
        name="Outlet Pressure P2",
        line=dict(color="blue"),
        fill="tozeroy",
        fillcolor="rgba(0,0,255,0.2)",
    ),
    row=2,
    col=1,
)
if "Pressure SP" in db.columns:
    fig.add_trace(
        go.Scatter(
            x=db["Timestamp"],
            y=db["Pressure SP"],
            mode="lines",
            name="Outlet Pressure SP",
            line=dict(color="green", dash="dash"),
        ),
        row=2,
        col=1,
    )
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["Inlet Steam Pressure"],
        mode="lines",
        name="Inlet Pressure P1",
        line=dict(color="cyan", dash="dashdot"),
    ),
    row=2,
    col=1,
)

# 3) Steam Flow Rate
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["Steam Flow Rate"],
        mode="lines",
        name="Steam Flow Rate",
        line=dict(color="purple"),
        fill="tozeroy",
        fillcolor="rgba(128,0,128,0.2)",
    ),
    row=3,
    col=1,
)

# 4) Valve Opening
fig.add_trace(
    go.Scatter(
        x=db["Timestamp"],
        y=db["QualSteam Valve Opening"],
        mode="lines",
        name="QualSteam Valve Opening",
        line=dict(color="orange"),
        fill="tozeroy",
        fillcolor="rgba(255,165,0,0.2)",
    ),
    row=4,
    col=1,
)

fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
fig.update_yaxes(title_text="Pressure (barg)", row=2, col=1)
fig.update_yaxes(title_text="Flow (kg/h)", row=3, col=1)
fig.update_yaxes(title_text="Valve Opening (%)", row=4, col=1)
fig.update_xaxes(title_text="Timestamp", row=4, col=1)

fig.update_layout(
    height=900,
    showlegend=True,
    title_text=f"QualSteam – Real Dairy – Batch {int(binfo['batch_id'])} | Duration: {duration_str}",
    margin=dict(l=40, r=40, t=80, b=40),
)

st.plotly_chart(fig, use_container_width=True)
