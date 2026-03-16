import io
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# Config
# ----------------------------
MILESTONES = ["CEP", "CGI", "CLL", "VDL", "VAD", "CDD", "CGO", "CER"]
ORDERED_MILESTONES = ["CEP", "CGI", "CLL", "VDL", "VAD", "CDD", "CGO", "CER"]
SEGMENTS = [
    ("CEP", "CGI"),
    ("CGI", "CLL"),
    ("CLL", "VDL"),
    ("VDL", "VAD"),
    ("VAD", "CDD"),
    ("CDD", "CGO"),
    ("CGO", "CER"),
]

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
# File ingest
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
# Validation
# ----------------------------
def get_missing_columns(
    df: pd.DataFrame,
    start_ms: str,
    end_ms: str,
    whole_journey: bool,
) -> List[str]:
    missing = []

    # Required base columns
    for col in REQUIRED_BASE_COLS:
        if col not in df.columns:
            missing.append(col)

    # Validate selected milestones for normal mode
    selected_milestones = {start_ms, end_ms}

    # Whole journey needs the full chain
    if whole_journey:
        selected_milestones.update(ORDERED_MILESTONES)

    for ms in sorted(selected_milestones):
        if ms in P44_FALLBACKS:
            fb = P44_FALLBACKS[ms]
            if ms not in df.columns and fb not in df.columns:
                missing.append(f"{ms} (or {fb})")
        else:
            if ms not in df.columns:
                missing.append(ms)

    return missing


# ----------------------------
# Datetime helpers
# ----------------------------
def _coerce_datetimes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_convert(None)
    return df


def _resolve_timestamp(row: pd.Series, milestone: str) -> pd.Timestamp:
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


def _segment_col_name(start_ms: str, end_ms: str) -> str:
    return f"SEG_{start_ms}_{end_ms}_HOURS"


def _segment_label(start_ms: str, end_ms: str) -> str:
    return f"{start_ms}-{end_ms}"


# ----------------------------
# Raw export prep
# ----------------------------
def add_shipment_month_year(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    dt_cols = ORDERED_MILESTONES + list(P44_FALLBACKS.values())
    dt_cols = list(dict.fromkeys(dt_cols))
    df = _coerce_datetimes(df, dt_cols)

    for ms in ORDERED_MILESTONES:
        df[f"_RES_{ms}"] = df.apply(lambda r: _resolve_timestamp(r, ms), axis=1)
        df[f"_RES_{ms}"] = pd.to_datetime(df[f"_RES_{ms}"], errors="coerce", utc=True).dt.tz_convert(None)

    resolved_cols = [f"_RES_{ms}" for ms in ORDERED_MILESTONES]
    first_ts = None
    for col in resolved_cols:
        if first_ts is None:
            first_ts = df[col]
        else:
            first_ts = first_ts.fillna(df[col])

    df["SHIPMENT_MONTH_YEAR"] = first_ts.dt.strftime("%b %Y")
    df["SHIPMENT_MONTH_YEAR"] = df["SHIPMENT_MONTH_YEAR"].fillna("")

    df = df.drop(columns=resolved_cols, errors="ignore")
    return df


# ----------------------------
# Core shipment computation
# ----------------------------
def compute_shipment_leadtimes(
    raw: pd.DataFrame,
    start_ms: str,
    end_ms: str,
    shipment_agg: str,
    whole_journey: bool,
) -> pd.DataFrame:
    df = raw.copy()

    dt_cols = [start_ms, end_ms] + ORDERED_MILESTONES + list(P44_FALLBACKS.values())
    dt_cols = list(dict.fromkeys(dt_cols))
    df = _coerce_datetimes(df, dt_cols)

    # Resolve all milestone timestamps once
    for ms in ORDERED_MILESTONES:
        df[f"_RES_{ms}"] = df.apply(lambda r: _resolve_timestamp(r, ms), axis=1)
        df[f"_RES_{ms}"] = pd.to_datetime(df[f"_RES_{ms}"], errors="coerce", utc=True).dt.tz_convert(None)

    df["_START_TS"] = df[f"_RES_{start_ms}"]
    df["_END_TS"] = df[f"_RES_{end_ms}"]

    group_cols = ["TENANT_NAME", "POL", "POD", "CARRIER_NAME", "CARRIER_SCAC", "MASTER_SHIPMENT_ID"]

    if whole_journey:
        # Must have all resolved milestones present
        all_present_mask = pd.Series(True, index=df.index)
        for ms in ORDERED_MILESTONES:
            all_present_mask = all_present_mask & pd.notna(df[f"_RES_{ms}"])

        # Every consecutive segment must be ordered correctly
        ordered_mask = pd.Series(True, index=df.index)
        for a, b in SEGMENTS:
            ordered_mask = ordered_mask & (df[f"_RES_{b}"] >= df[f"_RES_{a}"])

        # Selected journey must also be ordered correctly
        selected_mask = pd.notna(df["_START_TS"]) & pd.notna(df["_END_TS"]) & (df["_END_TS"] >= df["_START_TS"])

        df["_WHOLE_VALID"] = all_present_mask & ordered_mask & selected_mask
        qdf = df[df["_WHOLE_VALID"]].copy()

        if qdf.empty:
            return pd.DataFrame(columns=group_cols + ["LANE", "JOURNEY_LEAD_HOURS"] + [_segment_col_name(a, b) for a, b in SEGMENTS])

        # Compute selected journey duration
        qdf["JOURNEY_LEAD_HOURS"] = (qdf["_END_TS"] - qdf["_START_TS"]).dt.total_seconds() / 3600.0

        # Compute all segment durations
        segment_cols = []
        for a, b in SEGMENTS:
            col = _segment_col_name(a, b)
            qdf[col] = (qdf[f"_RES_{b}"] - qdf[f"_RES_{a}"]).dt.total_seconds() / 3600.0
            segment_cols.append(col)

        agg_fn = "min" if shipment_agg.lower().startswith("ear") else "max"
        ship = qdf.groupby(group_cols, dropna=False)[["JOURNEY_LEAD_HOURS"] + segment_cols].agg(agg_fn).reset_index()
        ship["LANE"] = ship.apply(lambda r: _make_lane(r["POL"], r["POD"]), axis=1)
        return ship

    # Normal mode
    df["_QUALIFIED"] = pd.notna(df["_START_TS"]) & pd.notna(df["_END_TS"]) & (df["_END_TS"] >= df["_START_TS"])
    qdf = df[df["_QUALIFIED"]].copy()

    if qdf.empty:
        return pd.DataFrame(columns=group_cols + ["LANE", "JOURNEY_LEAD_HOURS"])

    qdf["JOURNEY_LEAD_HOURS"] = (qdf["_END_TS"] - qdf["_START_TS"]).dt.total_seconds() / 3600.0
    agg_fn = "min" if shipment_agg.lower().startswith("ear") else "max"
    ship = qdf.groupby(group_cols, dropna=False)[["JOURNEY_LEAD_HOURS"]].agg(agg_fn).reset_index()
    ship["LANE"] = ship.apply(lambda r: _make_lane(r["POL"], r["POD"]), axis=1)
    return ship


# ----------------------------
# Counts
# ----------------------------
def compute_lane_and_carrier_counts(shipment_lt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if shipment_lt.empty:
        lane_counts = pd.DataFrame(columns=["Tenant Name", "Lane", "Shipments"])
        carrier_counts = pd.DataFrame(columns=["Tenant Name", "Carrier Name", "Carrier SCAC", "Shipments"])
        return lane_counts, carrier_counts

    lane_counts = (
        shipment_lt.groupby(["TENANT_NAME", "LANE"], dropna=False)["MASTER_SHIPMENT_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"TENANT_NAME": "Tenant Name", "LANE": "Lane", "MASTER_SHIPMENT_ID": "Shipments"})
        .sort_values(["Shipments", "Lane"], ascending=[False, True])
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
        .sort_values(["Shipments", "Carrier Name"], ascending=[False, True])
    )

    return lane_counts, carrier_counts


def apply_top_n_lanes_filter(shipment_lt: pd.DataFrame, top_n_lanes: int) -> pd.DataFrame:
    if shipment_lt.empty or top_n_lanes <= 0:
        return shipment_lt

    lane_vol = (
        shipment_lt.groupby(["TENANT_NAME", "LANE"], dropna=False)["MASTER_SHIPMENT_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"MASTER_SHIPMENT_ID": "SHIPMENTS"})
    )

    top_lanes = (
        lane_vol.sort_values(["TENANT_NAME", "SHIPMENTS", "LANE"], ascending=[True, False, True])
        .groupby("TENANT_NAME", dropna=False)
        .head(top_n_lanes)[["TENANT_NAME", "LANE"]]
        .drop_duplicates()
    )

    filtered = shipment_lt.merge(top_lanes, on=["TENANT_NAME", "LANE"], how="inner")
    return filtered


# ----------------------------
# Report metrics
# ----------------------------
def _stats_for_series(
    series: pd.Series,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
    prefix: str,
) -> Dict[str, Optional[float]]:
    s = series.dropna()
    vol = int(s.shape[0])

    out = {
        f"{prefix}_TOTAL_H": None,
        f"{prefix}_TOTAL_D": None,
        f"{prefix}_MIN_H": None,
        f"{prefix}_MIN_D": None,
        f"{prefix}_MED_H": None,
        f"{prefix}_MED_D": None,
        f"{prefix}_PCT_H": None,
        f"{prefix}_PCT_D": None,
        f"{prefix}_MAX_H": None,
        f"{prefix}_MAX_D": None,
    }

    if vol == 0:
        return out

    total_h = _round_hours(float(s.sum()))
    min_h = _round_hours(float(s.min()))
    med_h = _round_hours(float(s.median()))
    max_h = _round_hours(float(s.max()))

    out[f"{prefix}_TOTAL_H"] = total_h
    out[f"{prefix}_TOTAL_D"] = _round_days_from_hours(total_h)
    out[f"{prefix}_MIN_H"] = min_h
    out[f"{prefix}_MIN_D"] = _round_days_from_hours(min_h)
    out[f"{prefix}_MED_H"] = med_h
    out[f"{prefix}_MED_D"] = _round_days_from_hours(med_h)
    out[f"{prefix}_MAX_H"] = max_h
    out[f"{prefix}_MAX_D"] = _round_days_from_hours(max_h)

    if include_percentile and vol >= int(min_volume_for_percentile):
        pct_h = _round_hours(_safe_quantile(s, percentile_p / 100.0))
        out[f"{prefix}_PCT_H"] = pct_h
        out[f"{prefix}_PCT_D"] = _round_days_from_hours(pct_h)

    return out


def build_duration_configs(start_ms: str, end_ms: str, whole_journey: bool) -> List[Dict[str, str]]:
    configs = [
        {
            "data_col": "JOURNEY_LEAD_HOURS",
            "prefix": "JOURNEY",
            "label": f"{start_ms}-{end_ms}",
            "display_mode": "journey",
        }
    ]

    if whole_journey:
        for a, b in SEGMENTS:
            configs.append({
                "data_col": _segment_col_name(a, b),
                "prefix": f"SEG_{a}_{b}",
                "label": _segment_label(a, b),
                "display_mode": "segment",
            })

    return configs


def _group_stats(
    g: pd.DataFrame,
    duration_configs: List[Dict[str, str]],
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
) -> pd.Series:
    result = {"VOLUME": int(g["MASTER_SHIPMENT_ID"].nunique())}

    for cfg in duration_configs:
        result.update(
            _stats_for_series(
                g[cfg["data_col"]],
                percentile_p=percentile_p,
                include_percentile=include_percentile,
                min_volume_for_percentile=min_volume_for_percentile,
                prefix=cfg["prefix"],
            )
        )

    return pd.Series(result)


def build_carrier_lane_report(
    shipment_lt: pd.DataFrame,
    percentile_p: int,
    include_percentile: bool,
    min_volume_for_percentile: int,
    duration_configs: List[Dict[str, str]],
) -> pd.DataFrame:
    base_cols = ["TENANT_NAME", "LANE", "CARRIER_NAME", "CARRIER_SCAC", "VOLUME", "_IS_LANE_ROW", "_POL", "_POD"]

    metric_cols = []
    for cfg in duration_configs:
        pfx = cfg["prefix"]
        metric_cols.extend([
            f"{pfx}_TOTAL_H", f"{pfx}_TOTAL_D",
            f"{pfx}_MIN_H", f"{pfx}_MIN_D",
            f"{pfx}_MED_H", f"{pfx}_MED_D",
            f"{pfx}_PCT_H", f"{pfx}_PCT_D",
            f"{pfx}_MAX_H", f"{pfx}_MAX_D",
        ])

    cols = base_cols + metric_cols

    if shipment_lt.empty:
        return pd.DataFrame(columns=cols)

    lane_cols = ["TENANT_NAME", "POL", "POD", "LANE"]
    lane_stats = (
        shipment_lt.groupby(lane_cols, dropna=False)
        .apply(lambda g: _group_stats(g, duration_configs, percentile_p, include_percentile, min_volume_for_percentile))
        .reset_index()
    )
    lane_stats["CARRIER_NAME"] = "ALL CARRIERS"
    lane_stats["CARRIER_SCAC"] = ""

    carrier_cols = ["TENANT_NAME", "POL", "POD", "LANE", "CARRIER_NAME", "CARRIER_SCAC"]
    carrier_stats = (
        shipment_lt.groupby(carrier_cols, dropna=False)
        .apply(lambda g: _group_stats(g, duration_configs, percentile_p, include_percentile, min_volume_for_percentile))
        .reset_index()
    )

    lane_stats = lane_stats.sort_values(["TENANT_NAME", "VOLUME", "LANE"], ascending=[True, False, True])

    rows = []
    for _, lr in lane_stats.iterrows():
        tenant, lane, pol, pod = lr["TENANT_NAME"], lr["LANE"], lr["POL"], lr["POD"]

        row = {
            "TENANT_NAME": tenant,
            "LANE": lane,
            "CARRIER_NAME": lr["CARRIER_NAME"],
            "CARRIER_SCAC": lr["CARRIER_SCAC"],
            "VOLUME": lr["VOLUME"],
            "_IS_LANE_ROW": True,
            "_POL": pol,
            "_POD": pod,
        }
        for mc in metric_cols:
            row[mc] = lr.get(mc)
        rows.append(row)

        csub = carrier_stats[
            (carrier_stats["TENANT_NAME"] == tenant)
            & (carrier_stats["POL"].astype(str) == str(pol))
            & (carrier_stats["POD"].astype(str) == str(pod))
        ].sort_values(["VOLUME", "CARRIER_NAME"], ascending=[False, True])

        for _, cr in csub.iterrows():
            row = {
                "TENANT_NAME": tenant,
                "LANE": "",
                "CARRIER_NAME": cr["CARRIER_NAME"],
                "CARRIER_SCAC": cr["CARRIER_SCAC"],
                "VOLUME": cr["VOLUME"],
                "_IS_LANE_ROW": False,
                "_POL": pol,
                "_POD": pod,
            }
            for mc in metric_cols:
                row[mc] = cr.get(mc)
            rows.append(row)

    return pd.DataFrame(rows, columns=cols)


# ----------------------------
# Export helpers
# ----------------------------
def build_export_rename_map(duration_configs: List[Dict[str, str]], percentile_p: int) -> Dict[str, str]:
    export_cols = {
        "TENANT_NAME": DISPLAY_COLS["TENANT_NAME"],
        "LANE": DISPLAY_COLS["LANE"],
        "CARRIER_NAME": DISPLAY_COLS["CARRIER_NAME"],
        "CARRIER_SCAC": DISPLAY_COLS["CARRIER_SCAC"],
        "VOLUME": DISPLAY_COLS["VOLUME"],
    }

    for cfg in duration_configs:
        pfx = cfg["prefix"]
        label = cfg["label"]

        if cfg["display_mode"] == "journey":
            export_cols[f"{pfx}_TOTAL_H"] = DISPLAY_COLS["TOTAL_H"]
            export_cols[f"{pfx}_TOTAL_D"] = DISPLAY_COLS["TOTAL_D"]
            export_cols[f"{pfx}_MIN_H"] = DISPLAY_COLS["MIN_H"]
            export_cols[f"{pfx}_MIN_D"] = DISPLAY_COLS["MIN_D"]
            export_cols[f"{pfx}_MED_H"] = DISPLAY_COLS["MED_H"]
            export_cols[f"{pfx}_MED_D"] = DISPLAY_COLS["MED_D"]
            export_cols[f"{pfx}_PCT_H"] = DISPLAY_COLS["PCT_H"].format(p=percentile_p)
            export_cols[f"{pfx}_PCT_D"] = DISPLAY_COLS["PCT_D"].format(p=percentile_p)
            export_cols[f"{pfx}_MAX_H"] = DISPLAY_COLS["MAX_H"]
            export_cols[f"{pfx}_MAX_D"] = DISPLAY_COLS["MAX_D"]
        else:
            export_cols[f"{pfx}_TOTAL_H"] = f"{label} Total Lead Time (Hours)"
            export_cols[f"{pfx}_TOTAL_D"] = f"{label} Total Lead Time (Days)"
            export_cols[f"{pfx}_MIN_H"] = f"{label} Min Lead Time (Hours)"
            export_cols[f"{pfx}_MIN_D"] = f"{label} Min Lead Time (Days)"
            export_cols[f"{pfx}_MED_H"] = f"{label} Median Lead Time (Hours)"
            export_cols[f"{pfx}_MED_D"] = f"{label} Median Lead Time (Days)"
            export_cols[f"{pfx}_PCT_H"] = f"{label} P{percentile_p} Lead Time (Hours)"
            export_cols[f"{pfx}_PCT_D"] = f"{label} P{percentile_p} Lead Time (Days)"
            export_cols[f"{pfx}_MAX_H"] = f"{label} Max Lead Time (Hours)"
            export_cols[f"{pfx}_MAX_D"] = f"{label} Max Lead Time (Days)"

    return export_cols


def write_excel_counts(lane_counts: pd.DataFrame, carrier_counts: pd.DataFrame) -> bytes:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        lane_counts.to_excel(writer, sheet_name="Lane Counts", index=False)
        carrier_counts.to_excel(writer, sheet_name="Carrier Counts", index=False)

        bold = Font(bold=True)
        for sheet_name in ["Lane Counts", "Carrier Counts"]:
            ws = writer.book[sheet_name]
            for cell in ws[1]:
                cell.font = bold
            for col in range(1, ws.max_column + 1):
                ws.column_dimensions[get_column_letter(col)].width = 26

    output.seek(0)
    return output.getvalue()


def write_excel_final(
    raw_df: pd.DataFrame,
    report_df: pd.DataFrame,
    duration_configs: List[Dict[str, str]],
    percentile_p: int,
) -> bytes:
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    output = io.BytesIO()

    raw_export = add_shipment_month_year(raw_df)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        raw_export.to_excel(writer, sheet_name="Raw Data", index=False)

        export_rename_map = build_export_rename_map(duration_configs, percentile_p)

        if report_df.empty:
            pd.DataFrame(columns=list(export_rename_map.values())).to_excel(writer, sheet_name="Carrier Lane Lead", index=False)
        else:
            df = report_df.copy()
            lane_flags = df["_IS_LANE_ROW"].astype(bool).to_list()
            df = df.drop(columns=["_IS_LANE_ROW", "_POL", "_POD"], errors="ignore")

            ordered_export_keys = [k for k in export_rename_map.keys() if k in df.columns]
            df = df[ordered_export_keys].rename(columns=export_rename_map)
            df.to_excel(writer, sheet_name="Carrier Lane Lead", index=False)

            ws = writer.book["Carrier Lane Lead"]
            bold_font = Font(bold=True)

            for cell in ws[1]:
                cell.font = bold_font

            lane_col_idx = list(df.columns).index(DISPLAY_COLS["LANE"]) + 1
            for i, is_lane in enumerate(lane_flags, start=2):
                if is_lane:
                    ws.cell(row=i, column=lane_col_idx).font = bold_font

            width_map = {
                DISPLAY_COLS["TENANT_NAME"]: 22,
                DISPLAY_COLS["LANE"]: 28,
                DISPLAY_COLS["CARRIER_NAME"]: 28,
                DISPLAY_COLS["CARRIER_SCAC"]: 14,
                DISPLAY_COLS["VOLUME"]: 18,
                "SHIPMENT_MONTH_YEAR": 18,
            }
            for idx, col_name in enumerate(df.columns, start=1):
                ws.column_dimensions[get_column_letter(idx)].width = width_map.get(col_name, 24)

        ws_raw = writer.book["Raw Data"]
        bold_font = Font(bold=True)
        for cell in ws_raw[1]:
            cell.font = bold_font

    output.seek(0)
    return output.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Ocean Lead Time Analyzer", layout="wide")
st.title("Ocean Lead Time Analyzer (Carrier + Lane Lead Time)")

st.markdown(
    """
Upload a **CSV or Excel** extract. Pick journey **start** and **end** milestones.  
Output is an **Excel report** with:
- **Raw Data**
- **Carrier Lane Lead**
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
    index=MILESTONES.index("VDL"),
)

end_ms = st.sidebar.selectbox(
    "Journey end milestone",
    MILESTONES,
    index=MILESTONES.index("VAD"),
)

shipment_agg_label = st.sidebar.radio(
    "Shipment aggregation (multiple containers per master shipment)",
    options=["Earliest (MIN lead time)", "Latest (MAX lead time)"],
    index=0,
)
shipment_agg = "Earliest" if shipment_agg_label.startswith("Earliest") else "Latest"

whole_journey = st.sidebar.checkbox("Calculate for whole journey", value=False)

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

# Validation
missing_cols = get_missing_columns(
    df=raw_df,
    start_ms=start_ms,
    end_ms=end_ms,
    whole_journey=whole_journey,
)

if missing_cols:
    st.error(
        "The uploaded file is missing required columns:\n\n- " + "\n- ".join(missing_cols)
    )
    st.stop()

# Compute shipment-level data
try:
    shipment_lt_all = compute_shipment_leadtimes(
        raw=raw_df,
        start_ms=start_ms,
        end_ms=end_ms,
        shipment_agg=shipment_agg,
        whole_journey=whole_journey,
    )
except Exception as e:
    st.error(f"Error computing lead times: {e}")
    st.stop()

shipment_lt = apply_top_n_lanes_filter(shipment_lt_all, int(top_n_lanes))

# Metrics
total_shipments_raw = raw_df["MASTER_SHIPMENT_ID"].nunique() if "MASTER_SHIPMENT_ID" in raw_df.columns else None
eligible_shipments = shipment_lt["MASTER_SHIPMENT_ID"].nunique() if not shipment_lt.empty else 0
coverage = (eligible_shipments / total_shipments_raw * 100.0) if total_shipments_raw else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total Shipments (unique MASTER_SHIPMENT_ID in file)", f"{total_shipments_raw:,}" if total_shipments_raw is not None else "N/A")
c2.metric("Eligible Shipments (after current rules)", f"{eligible_shipments:,}")
c3.metric("Coverage vs total file shipments", f"{coverage:.1f}%")

if whole_journey:
    st.info("Whole journey mode is ON. Only shipments with all milestone timestamps are included.")

# Counts
lane_counts, carrier_counts = compute_lane_and_carrier_counts(shipment_lt)

st.subheader("Lane & Carrier Counts (shipment volume)")

lc, cc = st.columns(2)

with lc:
    st.markdown("**Lane Counts**")
    st.caption(f"Unique lanes: {lane_counts.shape[0]:,}")
    st.dataframe(lane_counts if lane_counts.shape[0] <= 25 else lane_counts.head(25), use_container_width=True)
    if lane_counts.shape[0] > 25:
        st.caption("Showing Top 25 lanes by shipment volume.")

with cc:
    st.markdown("**Carrier Counts**")
    st.caption(f"Unique carriers: {carrier_counts.shape[0]:,}")
    st.dataframe(carrier_counts if carrier_counts.shape[0] <= 25 else carrier_counts.head(25), use_container_width=True)
    if carrier_counts.shape[0] > 25:
        st.caption("Showing Top 25 carriers by shipment volume.")

counts_excel = write_excel_counts(lane_counts=lane_counts, carrier_counts=carrier_counts)
st.download_button(
    label="Download Lane + Carrier Counts (Excel)",
    data=counts_excel,
    file_name="lane_and_carrier_counts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Shipment preview
st.subheader("Shipment-level lead times (preview)")
preview_cols = [
    "TENANT_NAME", "MASTER_SHIPMENT_ID", "POL", "POD", "LANE", "CARRIER_NAME", "CARRIER_SCAC", "JOURNEY_LEAD_HOURS"
]
if whole_journey:
    preview_cols.extend([_segment_col_name(a, b) for a, b in SEGMENTS])

preview_cols = [c for c in preview_cols if c in shipment_lt.columns]
st.dataframe(shipment_lt[preview_cols].head(200), use_container_width=True)

# Final report
duration_configs = build_duration_configs(start_ms=start_ms, end_ms=end_ms, whole_journey=whole_journey)

report_df = build_carrier_lane_report(
    shipment_lt=shipment_lt,
    percentile_p=int(percentile_p),
    include_percentile=bool(include_percentile),
    min_volume_for_percentile=int(min_volume_for_pct),
    duration_configs=duration_configs,
)

st.subheader("Carrier Lane Lead (preview)")
preview_report = report_df.drop(columns=["_POL", "_POD", "_IS_LANE_ROW"], errors="ignore")
st.dataframe(preview_report, use_container_width=True)

final_excel = write_excel_final(
    raw_df=raw_df,
    report_df=report_df,
    duration_configs=duration_configs,
    percentile_p=int(percentile_p),
)
st.download_button(
    label="Download Final Excel Report",
    data=final_excel,
    file_name=f"carrier_lane_lead_{start_ms}_to_{end_ms}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

if st.button("Generate Insights"):
    st.info("Generate Insights is added as a placeholder. Share the insight logic next, and we’ll wire it in.")
