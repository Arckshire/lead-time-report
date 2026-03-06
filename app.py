# app.py
import io
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------
MILESTONES = ["CEP", "CGI", "CLL", "VDL", "VAD", "CDD", "CGO", "CER"]
P44_FALLBACKS = {"VDL": "VDL_P44", "VAD": "VAD_P44"}

REQUIRED_BASE_COLS = [
    "TENANT_NAME",
    "MASTER_SHIPMENT_ID",
    "POL",
    "POD",
    "CARRIER_NAME",
    "CARRIER_SCAC",
]

DISPLAY_COLS = {
    "TENANT_NAME": "Tenant Name",
    "LANE": "Lane",
    "CARRIER_NAME": "Carrier Name",
    "CARRIER_SCAC": "Carrier SCAC",
    "VOLUME": "Volume (Shipments)",

    "TOTAL_H": "Total Lead Time (Hours)",
    "TOTAL_D": "Total Lead Time (Days)",

    "MIN_H": "Min Lead Time (Hours)",
    "MIN_D": "Min Lead Time (Days)",

    "MED_H": "Median Lead Time (Hours)",
    "MED_D": "Median Lead Time (Days)",

    "PCT_H": "P{p} Lead Time (Hours)",
    "PCT_D": "P{p} Lead Time (Days)",

    "MAX_H": "Max Lead Time (Hours)",
    "MAX_D": "Max Lead Time (Days)",
}


# ----------------------------
# File ingest (robust encoding + excel)
# ----------------------------
def _read_input(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
        last_err = None
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError as e:
                last_err = e
                continue
        raise ValueError(
            f"Unable to decode CSV with {encodings}. Last error: {last_err}. "
            f"Try exporting as UTF-8, or upload Excel instead."
        )

    if name.endswith(".xlsx") or name.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


# ----------------------------
# Datetime handling (tz-aware vs tz-naive)
# ----------------------------
def _coerce_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Coerce timestamp columns into consistent tz-naive datetimes.
    Fixes: 'Cannot compare tz-naive and tz-aware timestamps'
    """
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_convert(None)  # tz-naive
    return df


def _resolve_timestamp(row: pd.Series, milestone: str) -> pd.Timestamp:
    """
    Resolve milestone timestamp with fallback only for VDL/VAD.
    Prefer carrier milestone over P44 fallback.
    """
    val = row.get(milestone, pd.NaT)
    if pd.notna(val):
        return val
    fb = P44_FALLBACKS.get(milestone)
    if fb:
        return row.get(fb, pd.NaT)
    return pd.NaT


def _round_hours(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.round(x, 2))


def _round_days_from_hours(x_hours: Optional[float]) -> Optional[int]:
    """
    Days must be clean integers (rounded).
    """
    if x_hours is None or (isinstance(x_hours, float) and np.isnan(x_hours)):
        return None
    return int(np.round(x_hours / 24.0))


def _safe_quantile(values: pd.Series, q: float) -> Optional[float]:
    values = values.dropna()
    if values.empty:
        return None
    return float(values.quantile(q, interpolation="linear"))


def _make_lane(pol, pod) -> str:
    pol_s = "" if pd.isna(pol) else str(pol)
    pod_s = "" if pd.isna(pod) else str(pod)
    return f"{pol_s} → {pod_s}"


# ----------------------------
# Core computation
# ----------------------------
def compute_shipment_leadtimes(
    raw: pd.DataFrame,
    start_ms: str,
    end_ms: str,
    shipment_agg: str,  # "Earliest" or "Latest"
) -> pd.DataFrame:
    """
    Container-level qualification -> container lead time -> shipment-level aggregation.

    Returns one row per:
      TENANT_NAME, POL, POD, CARRIER_NAME, CARRIER_SCAC, MASTER_SHIPMENT_ID
    with LEAD_TIME_HOURS and LANE.
    """
    df = raw.copy()

    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    dt_cols = [start_ms, end_ms]
    for ms in [start_ms, end_ms]:
        if ms in P44_FALLBACKS:
            dt_cols.append(P44_FALLBACKS[ms])
    dt_cols = list(dict.fromkeys(dt_cols))
    df = _coerce_datetimes(df, dt_cols)

    df["_START_TS"] = df.apply(lambda r: _resolve_timestamp(r, start_ms), axis=1)
    df["_END_TS"] = df.apply(lambda r: _resolve_timestamp(r, end_ms), axis=1)

    # Normalize resolved too (extra safety)
    df["_START_TS"] = pd.to_datetime(df["_START_TS"], errors="coerce", utc=True).dt.tz_convert(None)
    df["_END_TS"] = pd.to_datetime(df["_END_TS"], errors="coerce", utc=True).dt.tz_convert(None)

    df["_QUALIFIED"] = (
        pd.notna(df["_START_TS"]) & pd.notna(df["_END_TS"]) & (df["_END_TS"] >= df["_START_TS"])
    )

    # SAFE lead time computation (float hours)
    df["_LEAD_HOURS"] = np.nan
    valid_mask = df["_QUALIFIED"]
    df.loc[valid_mask, "_LEAD_HOURS"] = (
        (df.loc[valid_mask, "_END_TS"] - df.loc[valid_mask, "_START_TS"]).dt.total_seconds() / 3600.0
    )

    group_cols = ["TENANT_NAME", "POL", "POD", "CARRIER_NAME", "CARRIER_SCAC", "MASTER_SHIPMENT_ID"]
    qdf = df[df["_QUALIFIED"]].copy()

    if qdf.empty:
        return pd.DataFrame(columns=group_cols + ["LEAD_TIME_HOURS", "LANE"])

    agg_fn = "min" if shipment_agg.lower().startswith("ear") else "max"

    ship = (
        qdf.groupby(group_cols, dropna=False)["_LEAD_HOURS"]
        .agg(agg_fn)
        .reset_index()
        .rename(columns={"_LEAD_HOURS": "LEAD_TIME_HOURS"})
    )

    ship["LANE"] = ship.apply(lambda r: _make_lane(r["POL"], r["POD"]), axis=1)
    return ship


def _stats_from_group(
    lead_hours: pd.Series,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
) -> Dict[str, Optional[float]]:
    s = lead_hours.dropna()
    vol = int(s.shape[0])

    if vol == 0:
        return {
            "VOLUME": 0,
            "TOTAL_H": None, "TOTAL_D": None,
            "MIN_H": None, "MIN_D": None,
            "MED_H": None, "MED_D": None,
            "PCT_H": None, "PCT_D": None,
            "MAX_H": None, "MAX_D": None,
        }

    total_h = float(s.sum())
    min_h = float(s.min())
    med_h = float(s.median())
    max_h = float(s.max())

    pct_h = None
    if include_percentile and vol >= int(min_volume_for_percentile):
        pct_h = _safe_quantile(s, percentile_p / 100.0)

    total_h_r = _round_hours(total_h)
    min_h_r = _round_hours(min_h)
    med_h_r = _round_hours(med_h)
    max_h_r = _round_hours(max_h)
    pct_h_r = _round_hours(pct_h)

    return {
        "VOLUME": vol,

        "TOTAL_H": total_h_r,
        "TOTAL_D": _round_days_from_hours(total_h_r),

        "MIN_H": min_h_r,
        "MIN_D": _round_days_from_hours(min_h_r),

        "MED_H": med_h_r,
        "MED_D": _round_days_from_hours(med_h_r),

        "PCT_H": pct_h_r,
        "PCT_D": _round_days_from_hours(pct_h_r),

        "MAX_H": max_h_r,
        "MAX_D": _round_days_from_hours(max_h_r),
    }


def build_carrier_lane_report(
    shipment_lt: pd.DataFrame,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
) -> pd.DataFrame:
    """
    For each lane:
      - Lane summary row (ALL CARRIERS) with lane name filled (bold later)
      - Carrier rows below with lane column blank
    """
    cols = [
        "TENANT_NAME",
        "LANE",
        "CARRIER_NAME",
        "CARRIER_SCAC",
        "VOLUME",

        "TOTAL_H", "TOTAL_D",
        "MIN_H", "MIN_D",
        "MED_H", "MED_D",
        "PCT_H", "PCT_D",
        "MAX_H", "MAX_D",

        "_IS_LANE_ROW",
        "_POL",
        "_POD",
    ]

    if shipment_lt.empty:
        return pd.DataFrame(columns=cols)

    # Lane stats (ALL carriers combined)
    lane_cols = ["TENANT_NAME", "POL", "POD", "LANE"]
    lane_stats = (
        shipment_lt.groupby(lane_cols, dropna=False)
        .apply(lambda g: pd.Series(_stats_from_group(g["LEAD_TIME_HOURS"], percentile_p, include_percentile, min_volume_for_percentile)))
        .reset_index()
    )
    lane_stats["CARRIER_NAME"] = "ALL CARRIERS"
    lane_stats["CARRIER_SCAC"] = ""

    # Carrier stats
    carrier_cols = ["TENANT_NAME", "POL", "POD", "LANE", "CARRIER_NAME", "CARRIER_SCAC"]
    carrier_stats = (
        shipment_lt.groupby(carrier_cols, dropna=False)
        .apply(lambda g: pd.Series(_stats_from_group(g["LEAD_TIME_HOURS"], percentile_p, include_percentile, min_volume_for_percentile)))
        .reset_index()
    )

    # Order lanes by volume desc
    lane_stats = lane_stats.sort_values(["TENANT_NAME", "VOLUME", "LANE"], ascending=[True, False, True])

    rows = []
    for _, lr in lane_stats.iterrows():
        tenant, lane, pol, pod = lr["TENANT_NAME"], lr["LANE"], lr["POL"], lr["POD"]

        # Lane summary row
        rows.append({
            "TENANT_NAME": tenant,
            "LANE": lane,
            "CARRIER_NAME": lr["CARRIER_NAME"],
            "CARRIER_SCAC": lr["CARRIER_SCAC"],
            "VOLUME": lr["VOLUME"],

            "TOTAL_H": lr["TOTAL_H"], "TOTAL_D": lr["TOTAL_D"],
            "MIN_H": lr["MIN_H"], "MIN_D": lr["MIN_D"],
            "MED_H": lr["MED_H"], "MED_D": lr["MED_D"],
            "PCT_H": lr["PCT_H"], "PCT_D": lr["PCT_D"],
            "MAX_H": lr["MAX_H"], "MAX_D": lr["MAX_D"],

            "_IS_LANE_ROW": True,
            "_POL": pol,
            "_POD": pod,
        })

        # Carrier rows under lane (lane blank)
        csub = carrier_stats[
            (carrier_stats["TENANT_NAME"] == tenant)
            & (carrier_stats["POL"].astype(str) == str(pol))
            & (carrier_stats["POD"].astype(str) == str(pod))
        ].sort_values(["VOLUME", "CARRIER_NAME"], ascending=[False, True])

        for _, cr in csub.iterrows():
            rows.append({
                "TENANT_NAME": tenant,
                "LANE": "",
                "CARRIER_NAME": cr["CARRIER_NAME"],
                "CARRIER_SCAC": cr["CARRIER_SCAC"],
                "VOLUME": cr["VOLUME"],

                "TOTAL_H": cr["TOTAL_H"], "TOTAL_D": cr["TOTAL_D"],
                "MIN_H": cr["MIN_H"], "MIN_D": cr["MIN_D"],
                "MED_H": cr["MED_H"], "MED_D": cr["MED_D"],
                "PCT_H": cr["PCT_H"], "PCT_D": cr["PCT_D"],
                "MAX_H": cr["MAX_H"], "MAX_D": cr["MAX_D"],

                "_IS_LANE_ROW": False,
                "_POL": pol,
                "_POD": pod,
            })

    return pd.DataFrame(rows, columns=cols)


def compute_lane_and_carrier_counts(shipment_lt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses shipment-level rows (already one per master shipment per carrier+lane) to compute:
      - Lane Counts: unique shipments per TENANT + LANE (POL->POD)
      - Carrier Counts: unique shipments per TENANT + CARRIER
    """
    if shipment_lt.empty:
        lane_counts = pd.DataFrame(columns=["Tenant Name", "Lane", "Shipments"])
        carrier_counts = pd.DataFrame(columns=["Tenant Name", "Carrier Name", "Carrier SCAC", "Shipments"])
        return lane_counts, carrier_counts

    lane_counts = (
        shipment_lt.groupby(["TENANT_NAME", "LANE"], dropna=False)["MASTER_SHIPMENT_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"TENANT_NAME": "Tenant Name", "LANE": "Lane", "MASTER_SHIPMENT_ID": "Shipments"})
        .sort_values("Shipments", ascending=False)
    )

    carrier_counts = (
        shipment_lt.groupby(["TENANT_NAME", "CARRIER_NAME", "CARRIER_SCAC"], dropna=False)["MASTER_SHIPMENT_ID"]
        .nunique()
        .reset_index()
        .rename(columns={
            "TENANT_NAME": "Tenant Name",
            "CARRIER_NAME": "Carrier Name",
            "CARRIER_SCAC": "Carrier SCAC",
            "MASTER_SHIPMENT_ID": "Shipments",
        })
        .sort_values("Shipments", ascending=False)
    )

    return lane_counts, carrier_counts


def apply_top_n_lanes_filter(shipment_lt: pd.DataFrame, top_n_lanes: int) -> pd.DataFrame:
    """
    If top_n_lanes > 0, keep only shipments in the top N lanes by volume (per tenant).
    """
    if shipment_lt.empty or top_n_lanes <= 0:
        return shipment_lt

    # lane volume per tenant
    lane_vol = (
        shipment_lt.groupby(["TENANT_NAME", "LANE"], dropna=False)["MASTER_SHIPMENT_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"MASTER_SHIPMENT_ID": "SHIPMENTS"})
    )

    keep_pairs = []
    for tenant, sub in lane_vol.groupby("TENANT_NAME", dropna=False):
        sub_sorted = sub.sort_values(["SHIPMENTS", "LANE"], ascending=[False, True]).head(top_n_lanes)
        keep_pairs.extend(list(zip(sub_sorted["TENANT_NAME"], sub_sorted["LANE"])))

    keep_set = set(keep_pairs)
    filtered = shipment_lt[shipment_lt.apply(lambda r: (r["TENANT_NAME"], r["LANE"]) in keep_set, axis=1)].copy()
    return filtered


# ----------------------------
# Excel output (openpyxl only) — Excel Online safe
# ----------------------------
def write_excel_final(raw_df: pd.DataFrame, report_df: pd.DataFrame, percentile_p: int) -> bytes:
    """
    Final Excel output:
      - Raw Data
      - Carrier Lane Lead (with min + bold lane cell on lane rows)
    """
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)

        if report_df.empty:
            empty_cols = [
                DISPLAY_COLS["TENANT_NAME"],
                DISPLAY_COLS["LANE"],
                DISPLAY_COLS["CARRIER_NAME"],
                DISPLAY_COLS["CARRIER_SCAC"],
                DISPLAY_COLS["VOLUME"],

                DISPLAY_COLS["TOTAL_H"], DISPLAY_COLS["TOTAL_D"],
                DISPLAY_COLS["MIN_H"], DISPLAY_COLS["MIN_D"],
                DISPLAY_COLS["MED_H"], DISPLAY_COLS["MED_D"],
                DISPLAY_COLS["PCT_H"].format(p=percentile_p), DISPLAY_COLS["PCT_D"].format(p=percentile_p),
                DISPLAY_COLS["MAX_H"], DISPLAY_COLS["MAX_D"],
            ]
            pd.DataFrame(columns=empty_cols).to_excel(writer, sheet_name="Carrier Lane Lead", index=False)
        else:
            df = report_df.copy()
            lane_flags = df["_IS_LANE_ROW"].astype(bool).to_list()
            df = df.drop(columns=["_IS_LANE_ROW", "_POL", "_POD"])

            export_cols = {
                "TENANT_NAME": DISPLAY_COLS["TENANT_NAME"],
                "LANE": DISPLAY_COLS["LANE"],
                "CARRIER_NAME": DISPLAY_COLS["CARRIER_NAME"],
                "CARRIER_SCAC": DISPLAY_COLS["CARRIER_SCAC"],
                "VOLUME": DISPLAY_COLS["VOLUME"],

                "TOTAL_H": DISPLAY_COLS["TOTAL_H"],
                "TOTAL_D": DISPLAY_COLS["TOTAL_D"],
                "MIN_H": DISPLAY_COLS["MIN_H"],
                "MIN_D": DISPLAY_COLS["MIN_D"],
                "MED_H": DISPLAY_COLS["MED_H"],
                "MED_D": DISPLAY_COLS["MED_D"],
                "PCT_H": DISPLAY_COLS["PCT_H"].format(p=percentile_p),
                "PCT_D": DISPLAY_COLS["PCT_D"].format(p=percentile_p),
                "MAX_H": DISPLAY_COLS["MAX_H"],
                "MAX_D": DISPLAY_COLS["MAX_D"],
            }

            df = df[list(export_cols.keys())].rename(columns=export_cols)
            df.to_excel(writer, sheet_name="Carrier Lane Lead", index=False)

            ws = writer.book["Carrier Lane Lead"]
            bold_font = Font(bold=True)

            # Header bold
            for cell in ws[1]:
                cell.font = bold_font

            # Bold lane cell only for lane summary rows
            lane_col_idx = list(df.columns).index(DISPLAY_COLS["LANE"]) + 1
            for i, is_lane in enumerate(lane_flags, start=2):
                if is_lane:
                    ws.cell(row=i, column=lane_col_idx).font = bold_font

            # Column widths
            width_map = {
                DISPLAY_COLS["TENANT_NAME"]: 22,
                DISPLAY_COLS["LANE"]: 28,
                DISPLAY_COLS["CARRIER_NAME"]: 28,
                DISPLAY_COLS["CARRIER_SCAC"]: 14,
                DISPLAY_COLS["VOLUME"]: 18,
            }
            for idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[get_column_letter(idx)].width = width_map.get(col_name, 22)

    output.seek(0)
    return output.getvalue()


def write_excel_counts(lane_counts: pd.DataFrame, carrier_counts: pd.DataFrame) -> bytes:
    """
    Excel download with 2 sheets:
      - Lane Counts
      - Carrier Counts
    Includes ALL rows (not truncated).
    """
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        lane_counts.to_excel(writer, sheet_name="Lane Counts", index=False)
        carrier_counts.to_excel(writer, sheet_name="Carrier Counts", index=False)

        bold = Font(bold=True)

        # Format both sheets
        for sheet_name in ["Lane Counts", "Carrier Counts"]:
            ws = writer.book[sheet_name]
            for cell in ws[1]:
                cell.font = bold
            # basic widths
            for col in range(1, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(col)].width = 26

    output.seek(0)
    return output.getvalue()


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Ocean Lead Time Analyzer", layout="wide")
st.title("Ocean Lead Time Analyzer (Carrier + Lane Lead Time)")

st.markdown(
    """
Upload a **CSV or Excel** extract. Pick journey **start** and **end** milestones.
Output is an **Excel report**:
- **Raw Data**
- **Carrier Lane Lead** (lane summary + carrier rows)
"""
)

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.stop()

try:
    raw_df = _read_input(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns")

# Sidebar controls
st.sidebar.header("Journey Settings")

start_ms = st.sidebar.selectbox(
    "Journey start milestone",
    MILESTONES,
    index=MILESTONES.index("VDL") if "VDL" in MILESTONES else 0,
)
end_ms = st.sidebar.selectbox(
    "Journey end milestone",
    MILESTONES,
    index=MILESTONES.index("VAD") if "VAD" in MILESTONES else 1,
)

shipment_agg_label = st.sidebar.radio(
    "Shipment aggregation (multiple containers per master shipment)",
    options=["Earliest (MIN lead time)", "Latest (MAX lead time)"],
    index=0,
)
shipment_agg = "Earliest" if shipment_agg_label.startswith("Earliest") else "Latest"

st.sidebar.divider()
st.sidebar.header("Lane Filter")

top_n_lanes = st.sidebar.number_input(
    "Limit analysis to Top N lanes by volume (0 = all lanes)",
    min_value=0,
    max_value=1000,
    value=0,
    step=5,
)

st.sidebar.divider()
st.sidebar.header("Percentile Settings")

include_percentile = st.sidebar.checkbox("Include additional percentile (PXX)", value=True)

percentile_p = st.sidebar.number_input(
    "Percentile value (e.g., 80)",
    min_value=1,
    max_value=99,
    value=80,
    step=1,
    disabled=not include_percentile,
)

limit_by_volume = st.sidebar.checkbox(
    "Only compute percentile if volume ≥ threshold",
    value=False,
    disabled=not include_percentile,
)

min_volume = st.sidebar.number_input(
    "Min volume threshold",
    min_value=0,
    max_value=10_000_000,
    value=10,
    step=1,
    disabled=(not include_percentile) or (not limit_by_volume),
)

min_volume_for_pct = int(min_volume) if (include_percentile and limit_by_volume) else 0

# Compute shipment-level lead times
try:
    shipment_lt_all = compute_shipment_leadtimes(
        raw=raw_df,
        start_ms=start_ms,
        end_ms=end_ms,
        shipment_agg=shipment_agg,
    )
except Exception as e:
    st.error(f"Error computing lead times: {e}")
    st.stop()

# Apply Top-N lanes filter to the whole exercise
shipment_lt = apply_top_n_lanes_filter(shipment_lt_all, int(top_n_lanes))

# Metrics (total shipments refers to raw file uniqueness; eligible refers to selection)
total_shipments_raw = raw_df["MASTER_SHIPMENT_ID"].nunique() if "MASTER_SHIPMENT_ID" in raw_df.columns else None
eligible_shipments = shipment_lt["MASTER_SHIPMENT_ID"].nunique() if not shipment_lt.empty else 0
coverage = (eligible_shipments / total_shipments_raw * 100.0) if total_shipments_raw else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total Shipments (unique MASTER_SHIPMENT_ID in file)", f"{total_shipments_raw:,}" if total_shipments_raw is not None else "N/A")
c2.metric("Eligible Shipments (lead time computed; after lane filter)", f"{eligible_shipments:,}")
c3.metric("Coverage vs total file shipments", f"{coverage:.1f}%")

# Lane & Carrier counts (for on-screen summary + separate download)
lane_counts, carrier_counts = compute_lane_and_carrier_counts(shipment_lt)

st.subheader("Lane & Carrier Counts (shipment volume)")

lc, cc = st.columns(2)

with lc:
    st.markdown("**Lane Counts**")
    st.caption(f"Unique lanes: {lane_counts.shape[0]:,}")
    if lane_counts.shape[0] <= 25:
        st.dataframe(lane_counts, use_container_width=True)
    else:
        st.dataframe(lane_counts.head(25), use_container_width=True)
        st.caption("Showing Top 25 lanes by shipment volume.")

with cc:
    st.markdown("**Carrier Counts**")
    st.caption(f"Unique carriers: {carrier_counts.shape[0]:,}")
    if carrier_counts.shape[0] <= 25:
        st.dataframe(carrier_counts, use_container_width=True)
    else:
        st.dataframe(carrier_counts.head(25), use_container_width=True)
        st.caption("Showing Top 25 carriers by shipment volume.")

counts_excel = write_excel_counts(lane_counts=lane_counts, carrier_counts=carrier_counts)
st.download_button(
    label="Download Lane + Carrier Counts (Excel)",
    data=counts_excel,
    file_name="lane_and_carrier_counts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.subheader("Shipment-level lead times (preview)")
st.caption("One row per Tenant + Lane + Carrier + Master Shipment ID (after container qualification, shipment aggregation, and lane filter).")
st.dataframe(shipment_lt.head(200), use_container_width=True)

# Build final report (Carrier Lane Lead)
report_df = build_carrier_lane_report(
    shipment_lt=shipment_lt,
    percentile_p=int(percentile_p),
    include_percentile=bool(include_percentile),
    min_volume_for_percentile=int(min_volume_for_pct),
)

st.subheader("Carrier Lane Lead (preview)")
st.dataframe(
    report_df.drop(columns=["_POL", "_POD"]) if not report_df.empty else report_df,
    use_container_width=True,
)

# Final export (always available)
final_excel = write_excel_final(raw_df=raw_df, report_df=report_df, percentile_p=int(percentile_p))
st.download_button(
    label="Download Final Excel Report",
    data=final_excel,
    file_name=f"carrier_lane_lead_{start_ms}_to_{end_ms}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
