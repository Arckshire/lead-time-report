# app.py
import io
from typing import Dict, Tuple, Optional, List

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
    "MED_H": "Median Lead Time (Hours)",
    "MED_D": "Median Lead Time (Days)",
    "PCT_H": "P{p} Lead Time (Hours)",
    "PCT_D": "P{p} Lead Time (Days)",
    "MAX_H": "Max Lead Time (Hours)",
    "MAX_D": "Max Lead Time (Days)",
}

# ----------------------------
# Helpers
# ----------------------------
def _read_input(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def _coerce_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
    return df


def _resolve_timestamp(row: pd.Series, milestone: str) -> pd.Timestamp:
    """
    Resolve milestone timestamp with fallback only for VDL/VAD.
    Prefer carrier milestone over P44 fallback.
    """
    val = row.get(milestone, pd.NaT)
    if pd.notna(val):
        return val
    fallback = P44_FALLBACKS.get(milestone)
    if fallback:
        return row.get(fallback, pd.NaT)
    return pd.NaT


def _round_hours(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.round(x, 2))


def _round_days_from_hours(x_hours: Optional[float]) -> Optional[int]:
    if x_hours is None or (isinstance(x_hours, float) and np.isnan(x_hours)):
        return None
    return int(np.round(x_hours / 24.0))


def _safe_quantile(values: pd.Series, q: float) -> Optional[float]:
    values = values.dropna()
    if values.empty:
        return None
    return float(values.quantile(q, interpolation="linear"))


def _make_lane(pol: str, pod: str) -> str:
    pol_s = "" if pd.isna(pol) else str(pol)
    pod_s = "" if pd.isna(pod) else str(pod)
    return f"{pol_s} → {pod_s}"


def compute_shipment_leadtimes(
    raw: pd.DataFrame,
    start_ms: str,
    end_ms: str,
    shipment_agg: str,  # "Earliest" (min) or "Latest" (max) lead time
) -> pd.DataFrame:
    """
    Container-level qualification -> container lead time -> shipment-level aggregation.

    Returns one row per:
      TENANT_NAME, POL, POD, CARRIER_NAME, CARRIER_SCAC, MASTER_SHIPMENT_ID
    with LEAD_TIME_HOURS and LANE.
    """
    df = raw.copy()

    # Ensure required columns exist
    missing = [c for c in REQUIRED_BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # Coerce relevant datetime columns (start/end and possible fallbacks)
    dt_cols = [start_ms, end_ms]
    for ms in [start_ms, end_ms]:
        if ms in P44_FALLBACKS:
            dt_cols.append(P44_FALLBACKS[ms])
    dt_cols = list(dict.fromkeys(dt_cols))
    df = _coerce_datetimes(df, dt_cols)

    # Resolve per-row timestamps (container row)
    df["_START_TS"] = df.apply(lambda r: _resolve_timestamp(r, start_ms), axis=1)
    df["_END_TS"] = df.apply(lambda r: _resolve_timestamp(r, end_ms), axis=1)

    # Qualification: both timestamps present and end >= start
    df["_QUALIFIED"] = (
        pd.notna(df["_START_TS"]) & pd.notna(df["_END_TS"]) & (df["_END_TS"] >= df["_START_TS"])
    )

    # Lead time in hours at container level
    df["_LEAD_HOURS"] = np.where(
        df["_QUALIFIED"],
        (df["_END_TS"] - df["_START_TS"]).astype("timedelta64[s]") / 3600.0,
        np.nan,
    )

    # Group to shipment level within tenant+lane+carrier+master shipment
    group_cols = ["TENANT_NAME", "POL", "POD", "CARRIER_NAME", "CARRIER_SCAC", "MASTER_SHIPMENT_ID"]

    # Only consider qualified containers
    qdf = df[df["_QUALIFIED"]].copy()

    if qdf.empty:
        # Return empty with expected schema
        out = pd.DataFrame(columns=group_cols + ["LEAD_TIME_HOURS", "LANE"])
        return out

    if shipment_agg.lower().startswith("ear"):
        agg_fn = "min"
    else:
        agg_fn = "max"

    ship = (
        qdf.groupby(group_cols, dropna=False)["_LEAD_HOURS"]
        .agg(agg_fn)
        .reset_index()
        .rename(columns={"_LEAD_HOURS": "LEAD_TIME_HOURS"})
    )

    ship["LANE"] = ship.apply(lambda r: _make_lane(r["POL"], r["POD"]), axis=1)
    return ship


def build_carrier_lane_report(
    shipment_lt: pd.DataFrame,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
) -> pd.DataFrame:
    """
    Builds a hierarchical-ish table:
      Lane summary row (ALL carriers) then carrier rows
    with lane column blank on carrier rows.
    """
    if shipment_lt.empty:
        # Create empty report with expected columns
        cols = [
            "TENANT_NAME",
            "LANE",
            "CARRIER_NAME",
            "CARRIER_SCAC",
            "VOLUME",
            "TOTAL_H",
            "TOTAL_D",
            "MED_H",
            "MED_D",
            "PCT_H",
            "PCT_D",
            "MAX_H",
            "MAX_D",
        ]
        return pd.DataFrame(columns=cols)

    # Lane-level stats (ALL carriers)
    lane_cols = ["TENANT_NAME", "POL", "POD", "LANE"]
    lane_stats = shipment_lt.groupby(lane_cols, dropna=False).apply(
        lambda g: pd.Series(_stats_from_group(g["LEAD_TIME_HOURS"], percentile_p, include_percentile, min_volume_for_percentile))
    ).reset_index()

    lane_stats["CARRIER_NAME"] = "ALL CARRIERS"
    lane_stats["CARRIER_SCAC"] = ""

    # Carrier-level stats
    carrier_cols = ["TENANT_NAME", "POL", "POD", "LANE", "CARRIER_NAME", "CARRIER_SCAC"]
    carrier_stats = shipment_lt.groupby(carrier_cols, dropna=False).apply(
        lambda g: pd.Series(_stats_from_group(g["LEAD_TIME_HOURS"], percentile_p, include_percentile, min_volume_for_percentile))
    ).reset_index()

    # Order lanes by lane volume desc then lane name
    lane_stats = lane_stats.sort_values(["TENANT_NAME", "VOLUME", "LANE"], ascending=[True, False, True])

    # Build final "stacked" rows
    rows = []
    for _, lane_row in lane_stats.iterrows():
        tenant = lane_row["TENANT_NAME"]
        lane = lane_row["LANE"]
        pol = lane_row["POL"]
        pod = lane_row["POD"]

        # Lane summary row (lane shown)
        rows.append({
            "TENANT_NAME": tenant,
            "LANE": lane,
            "CARRIER_NAME": lane_row["CARRIER_NAME"],
            "CARRIER_SCAC": lane_row["CARRIER_SCAC"],
            "VOLUME": lane_row["VOLUME"],
            "TOTAL_H": lane_row["TOTAL_H"],
            "TOTAL_D": lane_row["TOTAL_D"],
            "MED_H": lane_row["MED_H"],
            "MED_D": lane_row["MED_D"],
            "PCT_H": lane_row["PCT_H"],
            "PCT_D": lane_row["PCT_D"],
            "MAX_H": lane_row["MAX_H"],
            "MAX_D": lane_row["MAX_D"],
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

        for _, crow in csub.iterrows():
            rows.append({
                "TENANT_NAME": tenant,
                "LANE": "",  # hide lane on carrier rows
                "CARRIER_NAME": crow["CARRIER_NAME"],
                "CARRIER_SCAC": crow["CARRIER_SCAC"],
                "VOLUME": crow["VOLUME"],
                "TOTAL_H": crow["TOTAL_H"],
                "TOTAL_D": crow["TOTAL_D"],
                "MED_H": crow["MED_H"],
                "MED_D": crow["MED_D"],
                "PCT_H": crow["PCT_H"],
                "PCT_D": crow["PCT_D"],
                "MAX_H": crow["MAX_H"],
                "MAX_D": crow["MAX_D"],
                "_IS_LANE_ROW": False,
                "_POL": pol,
                "_POD": pod,
            })

    out = pd.DataFrame(rows)

    # Clean display columns
    out = out.drop(columns=["_POL", "_POD"])
    return out


def _stats_from_group(
    lead_hours: pd.Series,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
) -> Dict[str, Optional[float]]:
    s = lead_hours.dropna()
    vol = int(s.shape[0])

    total_h = float(s.sum()) if vol else None
    med_h = float(s.median()) if vol else None
    max_h = float(s.max()) if vol else None

    pct_h = None
    if include_percentile and vol:
        if vol >= int(min_volume_for_percentile):
            pct_h = _safe_quantile(s, percentile_p / 100.0)

    # Rounding rules
    total_h_r = _round_hours(total_h)
    med_h_r = _round_hours(med_h)
    max_h_r = _round_hours(max_h)
    pct_h_r = _round_hours(pct_h)

    return {
        "VOLUME": vol,
        "TOTAL_H": total_h_r,
        "TOTAL_D": _round_days_from_hours(total_h_r),
        "MED_H": med_h_r,
        "MED_D": _round_days_from_hours(med_h_r),
        "PCT_H": pct_h_r,
        "PCT_D": _round_days_from_hours(pct_h_r),
        "MAX_H": max_h_r,
        "MAX_D": _round_days_from_hours(max_h_r),
    }


def write_excel(
    raw_df: pd.DataFrame,
    report_df: pd.DataFrame,
    percentile_p: int,
) -> bytes:
    """
    Produces an .xlsx with formatting:
      - Raw Data sheet as-is
      - Carrier Lane Lead sheet with:
         - clean headers
         - lane rows: lane cell bold
    """
    output = io.BytesIO()

    # Use xlsxwriter for formatting
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Sheet 1: Raw Data
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)

        # Sheet 2: Carrier Lane Lead
        if report_df.empty:
            # Still write an empty sheet with headers
            empty_cols = [
                DISPLAY_COLS["TENANT_NAME"],
                DISPLAY_COLS["LANE"],
                DISPLAY_COLS["CARRIER_NAME"],
                DISPLAY_COLS["CARRIER_SCAC"],
                DISPLAY_COLS["VOLUME"],
                DISPLAY_COLS["TOTAL_H"],
                DISPLAY_COLS["TOTAL_D"],
                DISPLAY_COLS["MED_H"],
                DISPLAY_COLS["MED_D"],
                DISPLAY_COLS["PCT_H"].format(p=percentile_p),
                DISPLAY_COLS["PCT_D"].format(p=percentile_p),
                DISPLAY_COLS["MAX_H"],
                DISPLAY_COLS["MAX_D"],
            ]
            pd.DataFrame(columns=empty_cols).to_excel(writer, sheet_name="Carrier Lane Lead", index=False)
        else:
            df = report_df.copy()

            # Rename columns to clean export labels
            export_cols = {
                "TENANT_NAME": DISPLAY_COLS["TENANT_NAME"],
                "LANE": DISPLAY_COLS["LANE"],
                "CARRIER_NAME": DISPLAY_COLS["CARRIER_NAME"],
                "CARRIER_SCAC": DISPLAY_COLS["CARRIER_SCAC"],
                "VOLUME": DISPLAY_COLS["VOLUME"],
                "TOTAL_H": DISPLAY_COLS["TOTAL_H"],
                "TOTAL_D": DISPLAY_COLS["TOTAL_D"],
                "MED_H": DISPLAY_COLS["MED_H"],
                "MED_D": DISPLAY_COLS["MED_D"],
                "PCT_H": DISPLAY_COLS["PCT_H"].format(p=percentile_p),
                "PCT_D": DISPLAY_COLS["PCT_D"].format(p=percentile_p),
                "MAX_H": DISPLAY_COLS["MAX_H"],
                "MAX_D": DISPLAY_COLS["MAX_D"],
            }

            is_lane_row = df["_IS_LANE_ROW"].astype(bool).to_list()
            df = df.drop(columns=["_IS_LANE_ROW"])

            df = df[list(export_cols.keys())].rename(columns=export_cols)
            df.to_excel(writer, sheet_name="Carrier Lane Lead", index=False)

            workbook = writer.book
            ws = writer.sheets["Carrier Lane Lead"]

            # Formats
            bold = workbook.add_format({"bold": True})
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#F2F2F2"})
            # Apply header format
            for col_idx, col_name in enumerate(df.columns):
                ws.write(0, col_idx, col_name, header_fmt)

            # Bold lane cell on lane rows
            lane_col_idx = df.columns.get_loc(DISPLAY_COLS["LANE"])
            for i, lane_flag in enumerate(is_lane_row, start=1):  # +1 due to header
                if lane_flag:
                    # Bold only the lane cell (per your preference)
                    val = df.iloc[i - 1, lane_col_idx]
                    ws.write(i, lane_col_idx, val, bold)

            # Set some reasonable column widths
            widths = {
                DISPLAY_COLS["TENANT_NAME"]: 22,
                DISPLAY_COLS["LANE"]: 28,
                DISPLAY_COLS["CARRIER_NAME"]: 28,
                DISPLAY_COLS["CARRIER_SCAC"]: 14,
                DISPLAY_COLS["VOLUME"]: 18,
            }
            for col_idx, col_name in enumerate(df.columns):
                ws.set_column(col_idx, col_idx, widths.get(col_name, 22))

    return output.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ocean Lead Time Analyzer", layout="wide")
st.title("Ocean Lead Time Analyzer (Carrier + Lane Lead Time)")

st.markdown(
    """
Upload a CSV or Excel extract from `ocean_data_quality`, pick your **start** and **end** milestones, and export an Excel report:
- **Raw Data**
- **Carrier Lane Lead** (Lane summary + Carrier rows)
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

start_ms = st.sidebar.selectbox("Journey start milestone", MILESTONES, index=MILESTONES.index("VDL") if "VDL" in MILESTONES else 0)
end_ms = st.sidebar.selectbox("Journey end milestone", MILESTONES, index=MILESTONES.index("VAD") if "VAD" in MILESTONES else 1)

shipment_agg = st.sidebar.radio(
    "Shipment aggregation (when multiple containers exist)",
    options=["Earliest (MIN lead time)", "Latest (MAX lead time)"],
    index=0,
)

st.sidebar.divider()
st.sidebar.header("Percentile Settings")

include_percentile = st.sidebar.checkbox("Include additional percentile (PXX)", value=True)
percentile_p = st.sidebar.number_input("Percentile value (e.g., 80)", min_value=1, max_value=99, value=80, step=1, disabled=not include_percentile)

limit_by_volume = st.sidebar.checkbox("Only compute percentile if volume ≥ threshold", value=False, disabled=not include_percentile)
min_volume = st.sidebar.number_input("Min volume threshold", min_value=0, max_value=10_000_000, value=10, step=1, disabled=(not include_percentile) or (not limit_by_volume))

min_volume_for_pct = int(min_volume) if (include_percentile and limit_by_volume) else 0

# Compute
try:
    shipment_lt = compute_shipment_leadtimes(
        raw=raw_df,
        start_ms=start_ms,
        end_ms=end_ms,
        shipment_agg=shipment_agg.split()[0],  # "Earliest"/"Latest"
    )
except Exception as e:
    st.error(f"Error computing lead times: {e}")
    st.stop()

total_shipments = raw_df["MASTER_SHIPMENT_ID"].nunique() if "MASTER_SHIPMENT_ID" in raw_df.columns else None
eligible_shipments = shipment_lt["MASTER_SHIPMENT_ID"].nunique() if not shipment_lt.empty else 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Shipments (unique MASTER_SHIPMENT_ID)", f"{total_shipments:,}" if total_shipments is not None else "N/A")
c2.metric("Eligible Shipments (with computed lead time)", f"{eligible_shipments:,}")
coverage = (eligible_shipments / total_shipments * 100.0) if total_shipments else 0.0
c3.metric("Coverage", f"{coverage:.1f}%")

st.subheader("Shipment-level lead times (preview)")
st.caption("One row per TENANT + LANE + CARRIER + MASTER_SHIPMENT_ID (after container qualification and shipment aggregation).")
st.dataframe(shipment_lt.head(200), use_container_width=True)

report_df = build_carrier_lane_report(
    shipment_lt=shipment_lt,
    percentile_p=int(percentile_p),
    include_percentile=bool(include_percentile),
    min_volume_for_percentile=int(min_volume_for_pct),
)

st.subheader("Carrier Lane Lead (preview)")
st.dataframe(
    report_df.drop(columns=["_IS_LANE_ROW"]) if "_IS_LANE_ROW" in report_df.columns else report_df,
    use_container_width=True,
)

# Export
excel_bytes = write_excel(raw_df=raw_df, report_df=report_df, percentile_p=int(percentile_p))

st.download_button(
    label="Download Excel Report",
    data=excel_bytes,
    file_name=f"carrier_lane_lead_{start_ms}_to_{end_ms}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
