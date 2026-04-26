"""
app.py  —  Capacity Expansion Model  (Juan Senga)
=================================================
Layout:
  • Sticky header
  • Zone network canvas (full-width, drag-to-reposition, click for info)
  • Zone load row (compact, one input per zone)
  • Generation fleet table (all generators across all zones)
  • Config row: Transmission | Policy | Solver + Solve
  • Results section

Run:  streamlit run app.py
"""

from __future__ import annotations
import copy, os, uuid, threading, time
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import declare_component

# ── Custom canvas component ────────────────────────────────────────────────────
_CANVAS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "backend", "canvas_component")
_zone_canvas = declare_component("zone_canvas", path=_CANVAS_DIR)

from backend.profiles import (
    generate_solar_cf, generate_wind_cf,
    generate_load_profile, load_profile_stats,
    SOLAR_CLIMATES, WIND_CLIMATES,
    LOAD_PROFILES, LOAD_PROFILE_KEYS,
)
from backend.model import build_and_solve

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Capacity Expansion Model — Juan Senga",
    page_icon=None, layout="wide", initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Generator type registry
# ─────────────────────────────────────────────────────────────────────────────
GEN_CONFIG = {
    "solar":   {"label": "Solar",       "color": "#E8A020", "re": True},
    "wind":    {"label": "Wind",        "color": "#2D7DD2", "re": True},
    "gas":     {"label": "Natural Gas", "color": "#8E6EC6", "re": False},
    "nuclear": {"label": "Nuclear",     "color": "#E74C3C", "re": False},
    "coal":    {"label": "Coal",        "color": "#7F8C8D", "re": False},
    "battery": {"label": "Battery",     "color": "#27AE60", "re": False},
}
GEN_TYPES_DISPLAY = ["Solar", "Wind", "Natural Gas", "Nuclear", "Coal", "Battery"]
_DISPLAY_TO_KEY   = {v["label"]: k for k, v in GEN_CONFIG.items()}

# Zone color palette (banner + left-border in fleet table)
ZONE_PALETTE = [
    {"bg": "#eaecf4", "border": "#14213d", "text": "#14213d"},
    {"bg": "#f9eaea", "border": "#8B2020", "text": "#8B2020"},
    {"bg": "#eaf2fb", "border": "#1A5276", "text": "#1A5276"},
    {"bg": "#eafaf1", "border": "#1E6B3A", "text": "#1E6B3A"},
    {"bg": "#f5eef8", "border": "#6C3483", "text": "#6C3483"},
    {"bg": "#fef9e7", "border": "#9A7D0A", "text": "#9A7D0A"},
    {"bg": "#e8f8f5", "border": "#0E6655", "text": "#0E6655"},
]

# Short type labels for generator naming ("Type_#_Zone" format)
TYPE_SHORT_LABEL = {
    "solar": "Solar", "wind": "Wind", "gas": "Gas",
    "nuclear": "Nuclear", "coal": "Coal", "battery": "Battery",
}

# EPA combustion emission factors (kg CO₂/MMBtu of fuel burned)
# Applied as: dispatch_MWh × heat_rate_MMBtu/MWh × factor → kg CO₂
EMISSIONS_KG_MMBTU = {
    "solar":   0.0,
    "wind":    0.0,
    "gas":    53.07,   # EPA AP-42: natural gas
    "nuclear": 0.0,    # no combustion
    "coal":   95.35,   # EPA AP-42: bituminous coal
    "battery": 0.0,
}

_GEN_DEFAULTS: dict[str, dict] = {
    "solar":   dict(capex=1_200_000.0, max_mw=500.0,
                    wacc=0.08, lifetime=30.0, fom=17_000.0, vom=0.0,
                    existing_mw=0.0, expandable=True, retirable=False),
    "wind":    dict(capex=1_500_000.0, max_mw=500.0,
                    wacc=0.08, lifetime=25.0, fom=43_000.0, vom=0.0,
                    existing_mw=0.0, expandable=True, retirable=False),
    "gas":     dict(capex=900_000.0, max_mw=200.0,
                    heat_rate=6.5, fuel_cost=4.0, vom=4.0,
                    ramp=1_000_000.0, min_load=0.0,
                    co2_factor=53.07,
                    wacc=0.08, lifetime=30.0, fom=12_000.0,
                    existing_mw=0.0, expandable=True, retirable=False),
    "nuclear": dict(capex=7_000_000.0, max_mw=1_000.0,
                    heat_rate=10.5, fuel_cost=0.75, vom=2.0,
                    ramp=50.0, min_load=0.9,
                    wacc=0.08, lifetime=40.0, fom=100_000.0,
                    existing_mw=0.0, expandable=True, retirable=False),
    "coal":    dict(capex=4_000_000.0, max_mw=500.0,
                    heat_rate=9.5, fuel_cost=2.0, vom=3.0,
                    ramp=100.0, min_load=0.4,
                    co2_factor=95.35,
                    wacc=0.08, lifetime=30.0, fom=40_000.0,
                    existing_mw=0.0, expandable=True, retirable=False),
    "battery": dict(capex=300_000.0, max_mw=500.0,
                    duration=4.0, eta_charge=0.96, eta_discharge=0.96,
                    self_discharge=0.0002, vom_c=0.5, vom_d=0.5,
                    wacc=0.08, lifetime=15.0, fom=20_000.0,
                    existing_mw=0.0, expandable=True, retirable=False),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
    --red:#A31F34; --red-dk:#7d1728; --navy:#14213d;
    --gray-dk:#4a4a4a; --gray-md:#767676; --gray-lt:#e8e8e8;
    --gray-bg:#f5f5f5; --white:#ffffff; --border:#d0d0d0;
    --input-bg:#fdfdfd; --font:'Inter',system-ui,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{
    font-family:var(--font);font-size:13px;
    background:var(--gray-bg)!important;color:var(--gray-dk);
}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"]{display:none;}
[data-testid="stSidebar"]{display:none;}
[data-testid="stAppViewBlockContainer"]{padding-top:0!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* ── Header row ──────────────────────────────────────────────────────────────
   The header is a st.columns() row — the 2nd direct child of the top-level
   stVerticalBlock (after the module-level CSS block).                        */
.block-container > div > div:nth-child(2){
    background:var(--red)!important;
    position:sticky!important;top:0!important;z-index:1000!important;
    border-bottom:2px solid var(--red-dk)!important;
    padding:0 20px!important;
    min-height:52px!important;
    align-items:center!important;
    margin:0!important;
}
/* Each column in the header — vertically centred */
.block-container > div > div:nth-child(2) [data-testid="column"]{
    display:flex!important;
    align-items:center!important;
    min-height:52px!important;
    padding:0!important;
}
.block-container > div > div:nth-child(2) [data-testid="column"] > div{
    width:100%!important;
    margin:0!important;
    padding:0!important;
}
/* Kill default <p> margin so text sits centred, not top-padded */
.block-container > div > div:nth-child(2) p{
    margin:0!important;padding:0!important;
    color:white!important;
}
.block-container > div > div:nth-child(2) span,
.block-container > div > div:nth-child(2) label{color:white!important;}
/* Buttons */
.block-container > div > div:nth-child(2) button{
    background:rgba(255,255,255,.18)!important;
    color:white!important;
    border:1px solid rgba(255,255,255,.38)!important;
    font-size:11px!important;font-weight:600!important;
    height:30px!important;min-height:0!important;
    padding:0 12px!important;border-radius:3px!important;
    white-space:nowrap!important;
}
.block-container > div > div:nth-child(2) button:hover{
    background:rgba(255,255,255,.3)!important;
    border-color:rgba(255,255,255,.6)!important;
    color:white!important;
}
/* Logo / title helper classes */
.cem-logo{display:flex;align-items:center;gap:8px;
    border-right:1px solid rgba(255,255,255,.3);padding-right:14px;}
.cem-logo-box{width:32px;height:32px;background:rgba(255,255,255,.15);
    border-radius:4px;display:flex;align-items:center;
    justify-content:center;font-size:14px;font-weight:800;color:white;}
.cem-logo-title{font-size:12px;font-weight:700;color:white!important;
    line-height:1.3;margin:0!important;}
.cem-logo-sub{font-size:10px;color:rgba(255,255,255,.7)!important;margin:0!important;}
.cem-hdr-title{font-size:15px;font-weight:700;color:white!important;
    white-space:nowrap;margin:0!important;}

/* Section headers */
.sec-hdr{
    background:var(--gray-bg);
    border-top:1px solid var(--border);border-bottom:1px solid var(--border);
    padding:5px 14px;font-size:9px;font-weight:700;
    letter-spacing:1.4px;text-transform:uppercase;color:var(--gray-md);
    border-left:3px solid var(--navy);color:var(--navy);
    margin-bottom:0;
}

/* Fleet table header */
.fleet-hdr-row{
    display:flex;align-items:center;
    padding:3px 14px 3px 14px;
    background:var(--gray-bg);border-bottom:1px solid var(--border);
    font-size:9px;font-weight:700;text-transform:uppercase;
    letter-spacing:.8px;color:var(--gray-md);
}

/* Generator type badge */
.gen-badge{
    display:inline-flex;align-items:center;gap:5px;
    padding:2px 8px 2px 6px;border-radius:2px;
    font-size:10px;font-weight:700;letter-spacing:.3px;
}
.gen-dot{
    width:7px;height:7px;border-radius:50%;flex-shrink:0;
}

/* Zone badge */
.zone-chip{
    display:inline-block;padding:2px 8px;border-radius:2px;
    background:#e8eef5;color:var(--navy);
    font-size:10px;font-weight:700;letter-spacing:.2px;
}

/* Widget overrides */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input{
    border:1px solid var(--border)!important;border-radius:2px!important;
    background:var(--input-bg)!important;font-size:12px!important;
    padding:4px 8px!important;height:30px!important;color:var(--gray-dk)!important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus{
    border-color:var(--red)!important;outline:none!important;
    box-shadow:0 0 0 2px rgba(163,31,52,.12)!important;
}
[data-testid="stSelectbox"]>div>div{
    border:1px solid var(--border)!important;border-radius:2px!important;
    background:var(--input-bg)!important;font-size:12px!important;
    min-height:30px!important;
}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label{
    font-size:9px!important;font-weight:700!important;
    color:var(--gray-md)!important;text-transform:uppercase!important;
    letter-spacing:.6px!important;
}
[data-testid="stNumberInput"] button{
    background:var(--gray-bg)!important;border-color:var(--border)!important;
    color:var(--gray-md)!important;height:30px!important;width:28px!important;
}
[data-testid="stCheckbox"]{margin:0!important;}
[data-testid="stCheckbox"] label{
    font-size:9px!important;font-weight:700!important;
    color:var(--gray-md)!important;text-transform:uppercase!important;
}

/* Buttons */
.stButton>button[kind="primary"]{
    background:var(--red)!important;color:white!important;border:none!important;
    border-radius:3px!important;font-weight:700!important;
    font-size:13px!important;height:36px!important;
}
.stButton>button[kind="primary"]:hover{background:var(--red-dk)!important;}
.stButton>button{
    border-radius:2px!important;font-size:11px!important;
    border-color:var(--border)!important;color:var(--gray-dk)!important;
}

/* Metrics */
[data-testid="stMetric"]{
    background:var(--white);border:1px solid var(--border);
    border-top:2px solid var(--red);padding:8px 12px!important;border-radius:2px;
}
[data-testid="stMetricValue"]{font-size:16px!important;font-weight:700!important;color:var(--navy)!important;}
[data-testid="stMetricLabel"]{font-size:10px!important;color:var(--gray-md)!important;}

/* Cards */
.card{background:var(--white);border:1px solid var(--border);
    border-radius:2px;margin-bottom:12px;overflow:hidden;}
.card-hdr{background:var(--gray-bg);border-bottom:1px solid var(--border);
    padding:6px 14px;font-size:9px;font-weight:700;
    letter-spacing:1.4px;text-transform:uppercase;color:var(--gray-md);}

/* Results */
.results-hdr{
    background:var(--navy);border-left:4px solid var(--red);
    padding:10px 16px;margin-bottom:12px;border-radius:2px;
}
.results-eyebrow{font-size:9px;letter-spacing:1.4px;text-transform:uppercase;color:#4f6080;margin-bottom:3px;}
.results-title{font-size:14px;font-weight:700;color:white;margin-bottom:2px;}
.results-stats{font-size:11px;color:#8fa3c8;}
.results-stats strong{color:#e8697a;}

[data-testid="stDataFrame"]{border-radius:2px;}
[data-testid="stSlider"]>div{padding:0!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _new_generator(gen_type: str,
                   solar_climate: str = "Temperate / Midwest",
                   wind_climate:  str = "Temperate / Midwest",
                   label: str = None) -> dict:
    g = {"id": str(uuid.uuid4())[:8], "type": gen_type,
         "label": label if label else GEN_CONFIG[gen_type]["label"]}
    g.update(_GEN_DEFAULTS[gen_type])
    if gen_type == "solar": g["climate"] = solar_climate
    if gen_type == "wind":  g["climate"] = wind_climate
    return g


def _default_zone(name: str) -> dict:
    return dict(
        id=name.replace(" ", "_"), name=name, load_mw=100.0,
        load_profile="flat",   # "flat"|"residential"|"commercial"|"industrial"|"custom"
        load_data=None,        # list[float] for CSV upload; None otherwise
        x=0.5, y=0.5,
        solar_climate="Temperate / Midwest",
        wind_climate="Temperate / Midwest",
        generators=[
            _new_generator("solar",   label=f"Solar_1_{name.replace(' ','')}"),
            _new_generator("wind",    label=f"Wind_1_{name.replace(' ','')}"),
            _new_generator("gas",     label=f"Gas_1_{name.replace(' ','')}"),
            _new_generator("battery", label=f"Battery_1_{name.replace(' ','')}"),
        ],
    )


def _default_policy() -> dict:
    return dict(min_re=0.0, voll=10_000.0, lifetime=25.0)


def _migrate_zone(z: dict) -> dict:
    """Upgrade zones from older flat-field format to generators-list format."""
    if "generators" in z:
        z.setdefault("solar_climate", "Temperate / Midwest")
        z.setdefault("wind_climate",  "Temperate / Midwest")
        z.setdefault("load_profile",  "flat")
        z.setdefault("load_data",     None)
        for g in z["generators"]:
            g.setdefault("existing_mw", 0.0)
            g.setdefault("expandable",  True)
            g.setdefault("retirable",   False)
        return z
    # Legacy format migration (old flat keys → generators list)
    gens = []
    if z.get("solar_enabled", False):
        g = _new_generator("solar", z.get("solar_climate", "Temperate / Midwest"))
        g.update(capex=z.get("solar_capex", 1_200_000.0),
                 max_mw=z.get("solar_max_mw", 500.0),
                 climate=z.get("solar_climate", "Temperate / Midwest"))
        gens.append(g)
    if z.get("wind_enabled", False):
        g = _new_generator("wind", wind_climate=z.get("wind_climate", "Temperate / Midwest"))
        g.update(capex=z.get("wind_capex", 1_500_000.0),
                 max_mw=z.get("wind_max_mw", 500.0),
                 climate=z.get("wind_climate", "Temperate / Midwest"))
        gens.append(g)
    if z.get("gas_enabled", True) and z.get("gas_max_mw", 0) > 0:
        g = _new_generator("gas")
        g.update(capex=z.get("gas_capex", 900_000.0), max_mw=z.get("gas_max_mw", 200.0),
                 heat_rate=z.get("gas_heat_rate", 6.5), fuel_cost=z.get("gas_fuel_cost", 4.0),
                 vom=z.get("gas_vom", 4.0), ramp=z.get("gas_ramp", 1e9),
                 min_load=z.get("gas_min_load", 0.0))
        gens.append(g)
    if z.get("batt_enabled", True):
        g = _new_generator("battery")
        g.update(capex=z.get("batt_capex", 300_000.0), max_mw=z.get("batt_max_mw", 500.0),
                 duration=z.get("batt_duration", 4.0),
                 eta_charge=z.get("batt_eta_charge", 0.96),
                 eta_discharge=z.get("batt_eta_discharge", 0.96),
                 self_discharge=z.get("batt_self_discharge", 0.0002))
        gens.append(g)
    if not gens:
        gens = [_new_generator("solar"), _new_generator("gas")]
    return dict(
        id=z.get("id", z.get("name", "Zone").replace(" ", "_")),
        name=z.get("name", "Zone"), load_mw=float(z.get("load_mw", 100.0)),
        x=z.get("x", 0.5), y=z.get("y", 0.5),
        solar_climate=z.get("solar_climate", "Temperate / Midwest"),
        wind_climate=z.get("wind_climate",  "Temperate / Midwest"),
        generators=gens,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def init_state():
    if "zones" not in st.session_state:
        st.session_state.zones = [
            {**_default_zone("Zone A"), "x": 0.25, "y": 0.45},
            {**_default_zone("Zone B"), "x": 0.72, "y": 0.45},
        ]
    else:
        st.session_state.zones = [_migrate_zone(z) for z in st.session_state.zones]
    for k in ("batteries", "gas", "grid"):
        st.session_state.pop(k, None)
    if "transmission" not in st.session_state:
        st.session_state.transmission = [{"from": "Zone_A", "to": "Zone_B", "cap_mw": 50.0}]
    if "policy" not in st.session_state:
        st.session_state.policy = _default_policy()
    else:
        for k, v in _default_policy().items():
            st.session_state.policy.setdefault(k, v)
    if "scenarios"    not in st.session_state: st.session_state.scenarios    = {}
    if "last_results" not in st.session_state: st.session_state.last_results = None
    # Invalidate results from old model structure
    res = st.session_state.last_results
    if res is not None and "generators" not in res:
        st.session_state.last_results = None
        st.session_state.scenarios    = {}
    if "solver"     not in st.session_state: st.session_state.solver     = "highs"
    if "T"          not in st.session_state: st.session_state.T          = 24
    if "time_limit" not in st.session_state: st.session_state.time_limit = 300
    if "canvas_key" not in st.session_state: st.session_state.canvas_key = 0
    # Solver threading state
    st.session_state.setdefault("_solving",        False)
    st.session_state.setdefault("_solve_result",   [])
    st.session_state.setdefault("_solve_error",    [])
    st.session_state.setdefault("_solver_thread",  None)
    st.session_state.setdefault("_stop_event",     None)
    st.session_state.setdefault("_pending_scenario", "")
    st.session_state.setdefault("uncap_all", False)

# ─────────────────────────────────────────────────────────────────────────────
# Widget seeding helper
# ─────────────────────────────────────────────────────────────────────────────

def _seed(key: str, value):
    st.session_state.setdefault(key, value)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble model inputs
# ─────────────────────────────────────────────────────────────────────────────

def assemble_inputs() -> dict:
    zones  = st.session_state.zones
    policy = st.session_state.policy
    xmit   = st.session_state.transmission
    T      = st.session_state.T
    zone_ids = [z["id"] for z in zones]

    model_gens = []
    for z in zones:
        for g in z["generators"]:
            mg = {
                "id":          g["id"],
                "zone":        z["id"],
                "type":        g["type"],
                "label":       g.get("label", g["type"]),
                "capex":       float(g.get("capex",       0)),
                "max_mw":      1e6 if st.session_state.get("uncap_all", False) else float(g.get("max_mw", 0)),
                "existing_mw": float(g.get("existing_mw", 0)),
                "expandable":  bool(g.get("expandable",   True)),
                "retirable":   bool(g.get("retirable",    False)),
                "wacc":        float(g.get("wacc",        0.08)),
                "lifetime":    float(g.get("lifetime",    25.0)),
                "fom":         float(g.get("fom",         0.0)),
            }
            typ = g["type"]
            if typ == "solar":
                mg["cf"]  = generate_solar_cf(g.get("climate", "Temperate / Midwest"), T, 42)
                mg["vom"] = float(g.get("vom", 0.0))
            elif typ == "wind":
                mg["cf"]  = generate_wind_cf(g.get("climate", "Temperate / Midwest"), T, 99)
                mg["vom"] = float(g.get("vom", 0.0))
            elif typ in ("gas", "nuclear", "coal"):
                hr = float(g.get("heat_rate", 6.5))
                vc = float(g.get("fuel_cost", 4.0)) * hr + float(g.get("vom", 4.0))
                mg["heat_rate"]  = hr
                mg["co2_factor"] = float(g.get("co2_factor",
                                               EMISSIONS_KG_MMBTU.get(typ, 0.0)))
                mg["vcost"]    = np.full(T, vc)
                mg["ramp"]     = float(g.get("ramp",     1e9))
                mg["min_load"] = float(g.get("min_load", 0.0))
            elif typ == "battery":
                mg["duration"] = float(g.get("duration",       4.0))
                mg["eta_c"]    = float(g.get("eta_charge",     0.96))
                mg["eta_d"]    = float(g.get("eta_discharge",  0.96))
                mg["sigma"]    = float(g.get("self_discharge", 0.0002))
                mg["vom_c"]    = float(g.get("vom_c",          0.5))
                mg["vom_d"]    = float(g.get("vom_d",          0.5))
            model_gens.append(mg)

    id_set = set(zone_ids)
    pairs  = []; seen = set()
    for lnk in xmit:
        z1, z2 = lnk["from"], lnk["to"]
        cap = float(lnk.get("cap_mw", 0))
        key = frozenset({z1, z2})
        if z1 in id_set and z2 in id_set and z1 != z2 and key not in seen:
            seen.add(key); pairs.append((z1, z2, cap))

    # Generate time-varying load profile per zone
    dc_load = {}
    for z in zones:
        profile = generate_load_profile(
            profile_type=z.get("load_profile", "flat"),
            peak_mw=float(z["load_mw"]),
            T=T,
            custom_data=z.get("load_data"),
        )
        dc_load[z["id"]] = profile

    return dict(
        zone_ids=zone_ids, T=T,
        solver=st.session_state.solver,
        time_limit=float(st.session_state.time_limit),
        dc_load=dc_load,
        generators=model_gens,
        min_re=policy["min_re"],
        voll=policy["voll"],
        lifetime=policy["lifetime"],
        transfer_pairs=pairs,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Canvas data (includes per-zone tech list for popup)
# ─────────────────────────────────────────────────────────────────────────────

def _zone_emissions_tco2(res: dict, zone_id: str) -> float:
    """Operational CO₂ emissions for a zone (tonnes/yr, scaled to full year).

    Uses EPA combustion factors in kg CO₂/MMBtu × generator heat rate (MMBtu/MWh).
    Only thermal generators with fuel combustion contribute.
    """
    T = res["T"]; scale = res["scale"]
    total_kg = 0.0
    for g in res["generators"]:
        if g["zone"] != zone_id: continue
        # Use per-generator override if present, else fall back to global table
        ef = float(g.get("co2_factor", EMISSIONS_KG_MMBTU.get(g["type"], 0.0)))
        if ef == 0: continue
        hr   = float(g.get("heat_rate", 1.0))   # MMBtu/MWh
        disp = np.asarray(res["gen_disp"].get(g["id"], np.zeros(T)))
        # kg CO₂ = MWh × MMBtu/MWh × kg CO₂/MMBtu
        total_kg += float(np.sum(np.clip(disp, 0, None))) * hr * ef * scale
    return total_kg / 1000.0   # kg → t


def _crf(wacc: float, lifetime: float) -> float:
    """Capital Recovery Factor: annualizes a lump-sum CapEx over `lifetime` years at `wacc`."""
    r = max(wacc, 0.0)
    n = max(lifetime, 1.0)
    if r < 1e-9:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _compute_zone_cost_breakdown(res: dict) -> dict:
    """Per-zone cost breakdown by component ($/yr).

    Returns {zone_id: {"Solar CapEx": x, "Wind CapEx": x, ..., "Total": x}}
    """
    T     = res["T"]
    scale = res["scale"]
    voll  = st.session_state.policy.get("voll", 10_000.0)

    _COMPONENTS = ["Solar CapEx", "Wind CapEx", "Battery CapEx",
                   "Thermal CapEx", "Fixed O&M", "Thermal OpEx", "Unserved"]
    breakdown: dict = {}
    for z in res["zones"]:
        row: dict = {k: 0.0 for k in _COMPONENTS}
        for g in res["generators"]:
            if g["zone"] != z:
                continue
            cap      = float(res["gen_cap"].get(g["id"], 0.0))
            existing = float(g.get("existing_mw", 0.0))
            new_cap  = max(0.0, cap - existing)
            wacc     = float(g.get("wacc", 0.08))
            lt       = float(g.get("lifetime", 25.0))
            inv      = new_cap * float(g.get("capex", 0.0)) * _crf(wacc, lt)
            row["Fixed O&M"] += cap * float(g.get("fom", 0.0))
            gtype = g["type"]
            if   gtype == "solar":   row["Solar CapEx"]   += inv
            elif gtype == "wind":    row["Wind CapEx"]    += inv
            elif gtype == "battery": row["Battery CapEx"] += inv
            else:                    row["Thermal CapEx"] += inv
            if gtype in ("gas", "nuclear", "coal"):
                disp  = np.asarray(res["gen_disp"].get(g["id"], np.zeros(T)))
                # vcost is stored as a constant array by assemble_inputs
                vcost = g.get("vcost")
                vc    = float(np.asarray(vcost)[0]) if vcost is not None else (
                        float(g.get("fuel_cost", 0)) * float(g.get("heat_rate", 1))
                        + float(g.get("vom", 0)))
                row["Thermal OpEx"] += float(np.sum(np.clip(disp, 0, None))) * vc * scale
        unserved = np.asarray(res["unserved"].get(z, np.zeros(T)))
        row["Unserved"] = float(np.sum(unserved)) * voll * scale
        row["Total"]    = sum(row[k] for k in _COMPONENTS)
        breakdown[z]    = row
    return breakdown


def _make_export_csv(res: Optional[dict] = None) -> bytes:
    """Build a multi-section CSV covering all inputs and (if available) results."""
    import io as _io
    buf = _io.StringIO()

    def _section(title: str):
        buf.write(f"\n# {title}\n")

    # ── Section 1: Generator inputs ───────────────────────────────────────────
    _section("GENERATOR INPUTS")
    gen_rows = []
    for z in st.session_state.zones:
        for g in z["generators"]:
            row = {
                "Zone":          z["name"],
                "Generator":     g.get("label", g["type"]),
                "Type":          g["type"],
                "Existing MW":   g.get("existing_mw", 0),
                "Max MW":        g.get("max_mw", 0),
                "Expandable":    g.get("expandable", True),
                "Retirable":     g.get("retirable", False),
                "CapEx ($/MW)":  g.get("capex", 0),
                "WACC (%)":      round(g.get("wacc", 0.08) * 100, 2),
                "Lifetime (yr)": g.get("lifetime", 25),
                "Fixed O&M ($/MW-yr)": g.get("fom", 0),
            }
            t = g["type"]
            if t in ("gas", "nuclear", "coal"):
                row["Heat Rate (MMBtu/MWh)"] = g.get("heat_rate", "")
                row["Fuel Cost ($/MMBtu)"]   = g.get("fuel_cost", "")
                row["VOM ($/MWh)"]           = g.get("vom", "")
                row["Ramp (MW/h)"]           = g.get("ramp", "")
                row["Min Load (0-1)"]         = g.get("min_load", "")
            if t in ("gas", "coal"):
                row["CO2 Factor (kg/MMBtu)"] = g.get("co2_factor", "")
            if t == "battery":
                row["Duration (h)"]        = g.get("duration", "")
                row["Charge Eff (%)"]      = round(g.get("eta_charge", 0.96) * 100, 1)
                row["Discharge Eff (%)"]   = round(g.get("eta_discharge", 0.96) * 100, 1)
                row["Self-Discharge (/h)"] = g.get("self_discharge", "")
            if t == "solar":
                row["Climate Profile"] = g.get("climate", "")
            if t == "wind":
                row["Climate Profile"] = g.get("climate", "")
            gen_rows.append(row)
    pd.DataFrame(gen_rows).to_csv(buf, index=False)

    # ── Section 2: Zone inputs ────────────────────────────────────────────────
    _section("ZONE INPUTS")
    zone_rows = [{"Zone": z["name"], "Peak Load (MW)": z["load_mw"]}
                 for z in st.session_state.zones]
    pd.DataFrame(zone_rows).to_csv(buf, index=False)

    # ── Section 3: Transmission ───────────────────────────────────────────────
    _section("TRANSMISSION LINKS")
    tx_rows = [{"From": t["from"], "To": t["to"], "Capacity (MW)": t["cap_mw"]}
               for t in st.session_state.transmission]
    if tx_rows:
        pd.DataFrame(tx_rows).to_csv(buf, index=False)
    else:
        buf.write("(none)\n")

    # ── Section 4: Policy ─────────────────────────────────────────────────────
    _section("POLICY SETTINGS")
    pol = st.session_state.policy
    pol_rows = [
        {"Parameter": "Min RE (fraction)", "Value": pol.get("min_re", 0)},
        {"Parameter": "VOLL ($/MWh)",      "Value": pol.get("voll", 5000)},
        {"Parameter": "Lifetime default (yr)", "Value": pol.get("lifetime", 25)},
    ]
    pd.DataFrame(pol_rows).to_csv(buf, index=False)

    if res:
        # ── Section 5: Capacity results ───────────────────────────────────────
        _section("CAPACITY RESULTS (MW)")
        cap_rows = []
        for g in res.get("generators", []):
            cap_rows.append({
                "Zone":       g["zone"],
                "Generator":  g.get("label", g["type"]),
                "Type":       g["type"],
                "Existing MW":   float(g.get("existing_mw", 0)),
                "Optimal MW":    round(res["gen_cap"].get(g["id"], 0), 2),
                "New MW":        round(max(0, res["gen_cap"].get(g["id"], 0)
                                          - float(g.get("existing_mw", 0))), 2),
            })
        pd.DataFrame(cap_rows).to_csv(buf, index=False)

        # ── Section 6: Dispatch / energy results ──────────────────────────────
        _section("DISPATCH RESULTS (MWh/yr, scaled)")
        disp_rows = []
        scale = res.get("scale", 1.0)
        T     = res.get("T", 8760)
        for g in res.get("generators", []):
            gid = g["id"]
            if g["type"] in ("solar", "wind"):
                disp = res["gen_disp"].get(gid, np.zeros(T))
            else:
                disp = res["gen_disp"].get(gid, np.zeros(T))
            mwh_yr = float(np.sum(np.clip(np.asarray(disp), 0, None))) * scale
            disp_rows.append({
                "Zone":         g["zone"],
                "Generator":    g.get("label", g["type"]),
                "Type":         g["type"],
                "Energy (MWh/yr)": round(mwh_yr, 0),
                "CF (actual)":     round(mwh_yr / max(res["gen_cap"].get(gid, 1), 1) / 8760, 3)
                                   if res["gen_cap"].get(gid, 0) > 0 else 0,
            })
        pd.DataFrame(disp_rows).to_csv(buf, index=False)

        # ── Section 7: Cost summary ───────────────────────────────────────────
        _section("COST SUMMARY ($/yr)")
        costs = res.get("costs", {})
        cost_rows = [
            {"Component": "Solar CapEx",     "$/yr": costs.get("capex_solar", 0)},
            {"Component": "Wind CapEx",      "$/yr": costs.get("capex_wind",  0)},
            {"Component": "Battery CapEx",   "$/yr": costs.get("capex_batt",  0)},
            {"Component": "Thermal CapEx",   "$/yr": costs.get("capex_therm", 0)},
            {"Component": "Fixed O&M",       "$/yr": costs.get("fom_total",   0)},
            {"Component": "Thermal OpEx",    "$/yr": costs.get("opex_therm",  0)},
            {"Component": "Unserved (VOLL)", "$/yr": costs.get("opex_voll",   0)},
            {"Component": "TOTAL",           "$/yr": costs.get("total",       0)},
        ]
        pd.DataFrame(cost_rows).to_csv(buf, index=False)

    return buf.getvalue().encode()


def _zone_annualized_cost(res: dict, zone_id: str, lifetime: float) -> float:
    """Approximate annualized cost for a zone ($/yr)."""
    T = res["T"]; scale = res["scale"]
    total = 0.0
    for g in res["generators"]:
        if g["zone"] != zone_id: continue
        cap      = res["gen_cap"].get(g["id"], 0.0)
        existing = float(g.get("existing_mw", 0.0))
        new_cap  = max(0.0, cap - existing)
        lt       = float(g.get("lifetime", lifetime))
        wacc     = float(g.get("wacc", 0.08))
        fom      = float(g.get("fom", 0.0))
        # Annualized investment cost via CRF — on new capacity only
        total += new_cap * float(g.get("capex", 0.0)) * _crf(wacc, lt)
        # Fixed O&M on total installed capacity
        total += cap * fom
        if g["type"] in ("gas", "nuclear", "coal"):
            disp  = np.asarray(res["gen_disp"].get(g["id"], np.zeros(T)))
            vcost = g.get("vcost")
            vc    = float(np.asarray(vcost)[0]) if vcost is not None else (
                    float(g.get("fuel_cost", 0)) * float(g.get("heat_rate", 1))
                    + float(g.get("vom", 0)))
            total += float(np.sum(np.clip(disp, 0, None))) * vc * scale
    return total


def _canvas_data(zones: list, transmission: list, res: dict = None):
    zones_data = []
    for z in zones:
        zid = z["id"]
        # Aggregate MW by technology type
        type_mw: dict = defaultdict(lambda: {"mw": 0.0, "existing": 0.0})
        for g in z.get("generators", []):
            t = g["type"]
            type_mw[t]["mw"]       += float(g.get("max_mw",      0))
            type_mw[t]["existing"] += float(g.get("existing_mw", 0))
        techs = []
        for gtype in ("solar", "wind", "gas", "nuclear", "coal", "battery"):
            if gtype in type_mw and type_mw[gtype]["mw"] > 0:
                techs.append({
                    "type":     gtype,
                    "label":    GEN_CONFIG[gtype]["label"],
                    "color":    GEN_CONFIG[gtype]["color"],
                    "mw":       round(type_mw[gtype]["mw"]),
                    "existing": round(type_mw[gtype]["existing"]),
                })

        node: dict = {
            "id": zid, "name": z["name"],
            "load": round(z["load_mw"]),
            "x": z.get("x", 0.5), "y": z.get("y", 0.5),
            "techs": techs,
        }

        # Enrich with optimization results when available
        if res is not None:
            T = res["T"]; scale = res["scale"]
            re_gen = (
                _zone_type_disp(res, zid, "solar").sum() +
                _zone_type_disp(res, zid, "wind").sum()
            ) * scale
            tot_gen = sum(
                _zone_type_disp(res, zid, gt).sum()
                for gt in GEN_CONFIG
            ) * scale
            node["re_pct"] = round(re_gen / tot_gen * 100, 1) if tot_gen > 0 else 0.0
            node["total_cap_mw"] = round(sum(
                res["gen_cap"].get(g["id"], 0)
                for g in res["generators"] if g["zone"] == zid
            ))
            node["has_unserved"] = float(
                np.sum(res["unserved"].get(zid, np.zeros(T)))
            ) > 0.1
            # Per-type installed cap for popup
            node["result_techs"] = [
                {"type": gt, "label": GEN_CONFIG[gt]["label"],
                 "color": GEN_CONFIG[gt]["color"],
                 "cap_mw": round(_zone_cap_by_type(res, zid, gt))}
                for gt in GEN_CONFIG
                if _zone_cap_by_type(res, zid, gt) > 0.1
            ]
            # Emissions and cost
            lifetime = st.session_state.policy.get("lifetime", 25.0)
            zone_load_mwh = float(np.sum(res["dc_load"][zid])) * scale
            zone_cost     = _zone_annualized_cost(res, zid, lifetime)
            node["emissions_tco2"] = round(_zone_emissions_tco2(res, zid))
            node["lcoe"] = round(zone_cost / zone_load_mwh, 1) if zone_load_mwh > 0 else 0.0

        zones_data.append(node)

    # Attach per-zone cost breakdown (requires results)
    if res is not None:
        breakdown = _compute_zone_cost_breakdown(res)
        _COST_ORDER = [
            ("Solar CapEx",   "#E8A020"),
            ("Wind CapEx",    "#2D7DD2"),
            ("Battery CapEx", "#27AE60"),
            ("Thermal CapEx", "#8E6EC6"),
            ("Fixed O&M",     "#5D7A8E"),
            ("Thermal OpEx",  "#aaa"),
            ("Unserved",      "#A31F34"),
        ]
        for node in zones_data:
            zid  = node["id"]
            bd   = breakdown.get(zid, {})
            node["zone_costs"] = [
                {"label": lbl, "color": clr,
                 "value_k": round(bd.get(lbl, 0.0) / 1_000)}
                for lbl, clr in _COST_ORDER
                if bd.get(lbl, 0.0) > 500   # skip negligible components
            ]
            node["zone_total_cost_m"] = round(bd.get("Total", 0.0) / 1_000_000, 2)

    edges_data = []
    for l in transmission:
        edge = {"from": l["from"], "to": l["to"], "cap": round(l["cap_mw"])}
        if res is not None:
            # Try both directions (transfer key might be z1->z2 or z2->z1)
            key_ab = f"{l['from']}->{l['to']}"
            key_ba = f"{l['to']}->{l['from']}"
            arr = res["transfer"].get(key_ab, res["transfer"].get(key_ba))
            if arr is not None:
                a = np.asarray(arr)
                avg_abs = float(np.mean(np.abs(a)))
                net_avg = float(np.mean(a))
                util    = avg_abs / l["cap_mw"] if l["cap_mw"] > 0 else 0.0
                edge["avg_flow"] = round(avg_abs, 1)
                edge["net_flow"] = round(net_avg, 1)
                edge["util"]     = round(min(util, 1.0), 3)
        edges_data.append(edge)

    return zones_data, edges_data

# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCH_COLORS = {
    "solar": "#E8A020", "wind": "#2D7DD2",
    "gas": "#8E6EC6", "nuclear": "#E74C3C",
    "coal": "#7F8C8D", "battery": "#27AE60",
    "unserved": "#A31F34",
}

def _zone_type_disp(res, zone_id, gen_type):
    T = res["T"]; total = np.zeros(T)
    for g in res["generators"]:
        if g["zone"] == zone_id and g["type"] == gen_type:
            total += np.clip(np.asarray(res["gen_disp"].get(g["id"], np.zeros(T))), 0, None)
    return total

def _zone_batt_charge(res, zone_id):
    T = res["T"]; total = np.zeros(T)
    for g in res["generators"]:
        if g["zone"] == zone_id and g["type"] == "battery":
            total += np.asarray(res["gen_charge"].get(g["id"], np.zeros(T)))
    return total

def _zone_cap_by_type(res, zone_id, gen_type):
    return sum(res["gen_cap"].get(g["id"], 0)
               for g in res["generators"]
               if g["zone"] == zone_id and g["type"] == gen_type)

def plot_dispatch(res, zone_id, h_start=0, h_end=167):
    T = res["T"]
    h_end   = min(h_end, T - 1)
    hrs     = np.arange(T)
    load    = np.asarray(res["dc_load"][zone_id])
    fig     = go.Figure()

    # Accumulate stacked heights so we can set y-axis correctly
    pos_stack = np.zeros(T)   # cumulative positive (generation + imports)
    neg_stack = np.zeros(T)   # cumulative negative (charge + exports), positive values

    def add_pos(name, arr, color):
        arr = np.clip(np.asarray(arr), 0, None)
        if arr.max() < 0.01:
            return
        pos_stack[:] += arr
        fig.add_trace(go.Scatter(
            x=hrs, y=arr, name=name, stackgroup="pos",
            mode="none", fillcolor=color, line=dict(width=0),
            hovertemplate=f"{name}: %{{y:.1f}} MW<extra></extra>"))

    def add_neg(name, arr, color):
        arr = np.clip(np.asarray(arr), 0, None)
        if arr.max() < 0.01:
            return
        neg_stack[:] += arr
        fig.add_trace(go.Scatter(
            x=hrs, y=-arr, name=name,
            mode="none", fill="tozeroy", stackgroup="neg",
            fillcolor=color, line=dict(width=0),
            hovertemplate=f"{name}: %{{y:.1f}} MW<extra></extra>"))

    for gtype in ("solar", "wind", "gas", "nuclear", "coal", "battery"):
        d = _zone_type_disp(res, zone_id, gtype)
        if d.max() > 0.01:
            add_pos(GEN_CONFIG[gtype]["label"], d, _DISPATCH_COLORS[gtype])
    add_pos("Unserved", res["unserved"].get(zone_id, np.zeros(T)),
            _DISPATCH_COLORS["unserved"])

    # Transmission net flows
    xfer = res.get("transfer", {})
    if xfer:
        net = np.zeros(T)
        for key, arr in xfer.items():
            parts = key.split("->")
            if len(parts) == 2:
                z1, z2 = parts
                a = np.asarray(arr)
                if   z2 == zone_id: net += a
                elif z1 == zone_id: net -= a
        imports = np.clip( net, 0, None)
        exports = np.clip(-net, 0, None)
        if imports.max() > 0.01:
            add_pos("Tx Import", imports, "rgba(100,180,255,0.55)")
        if exports.max() > 0.01:
            add_neg("Tx Export", exports, "rgba(100,180,255,0.35)")

    chg = _zone_batt_charge(res, zone_id)
    if chg.max() > 0.01:
        add_neg("Batt charge", chg, "rgba(39,174,96,.3)")

    fig.add_trace(go.Scatter(
        x=hrs, y=load, name="Demand", mode="lines",
        line=dict(color="#333", width=1.5, dash="dash")))

    # Compute y-axis range from the visible window
    win     = slice(h_start, h_end + 1)
    max_pos = max(float(pos_stack[win].max()), float(load[win].max()), 1.0)
    max_neg = float(neg_stack[win].max()) if neg_stack[win].max() > 0 else 0.0

    fig.update_xaxes(range=[h_start, h_end])
    fig.update_yaxes(range=[-(max_neg * 1.15), max_pos * 1.15])
    fig.update_layout(
        title=f"Dispatch — {zone_id}", xaxis_title="Hour", yaxis_title="MW",
        hovermode="x unified", template="simple_white", height=340,
        legend=dict(
            orientation="h", font=dict(size=10),
            yanchor="top", y=-0.18,
            xanchor="center", x=0.5,
        ),
        margin=dict(t=50, b=90, l=50, r=10),
        font=dict(family="Inter,system-ui", size=11))
    return fig

def plot_capacity_mix(res):
    zones = res["zones"]; fig = go.Figure()
    totals = [0.0] * len(zones)
    for gtype in ("solar", "wind", "gas", "nuclear", "coal", "battery"):
        vals = [_zone_cap_by_type(res, z, gtype) for z in zones]
        for j, v in enumerate(vals):
            totals[j] += v
        if max(vals, default=0) > 0.1:
            fig.add_trace(go.Bar(name=GEN_CONFIG[gtype]["label"], x=zones, y=vals,
                                  marker_color=_DISPATCH_COLORS[gtype]))
    max_y = max(totals, default=0)
    fig.update_layout(title="Installed Capacity by Zone", xaxis_title="Zone", yaxis_title="MW",
                       barmode="stack", template="simple_white", height=310,
                       yaxis=dict(range=[0, max_y * 1.18]),
                       legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                       margin=dict(t=50,b=40,l=50,r=10),
                       font=dict(family="Inter,system-ui", size=11))
    return fig


def plot_zone_cost_breakdown(res: dict) -> go.Figure:
    """Stacked bar chart: annualized cost by component per zone ($/yr)."""
    breakdown = _compute_zone_cost_breakdown(res)
    zones = res["zones"]
    _COST_PALETTE = [
        ("Solar CapEx",   "#E8A020"),
        ("Wind CapEx",    "#2D7DD2"),
        ("Battery CapEx", "#27AE60"),
        ("Thermal CapEx", "#8E6EC6"),
        ("Fixed O&M",     "#5D7A8E"),
        ("Thermal OpEx",  "#aaa"),
        ("Unserved",      "#A31F34"),
    ]
    fig = go.Figure()
    totals = [0.0] * len(zones)
    for comp, color in _COST_PALETTE:
        vals = [breakdown.get(z, {}).get(comp, 0.0) for z in zones]
        for j, v in enumerate(vals):
            totals[j] += v
        if max(vals, default=0) > 1:
            fig.add_trace(go.Bar(
                name=comp, x=zones, y=vals, marker_color=color,
                hovertemplate=comp + ": $%{y:,.0f}/yr<extra></extra>",
            ))
    max_y = max(totals, default=0)
    fig.update_layout(
        title="Annualized Cost by Zone ($/yr)",
        xaxis_title="Zone", yaxis_title="$/yr",
        barmode="stack", template="simple_white", height=310,
        yaxis=dict(range=[0, max_y * 1.18]),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        margin=dict(t=50,b=40,l=50,r=10),
        font=dict(family="Inter,system-ui", size=11),
    )
    return fig

def plot_cost_breakdown(res):
    costs = res.get("costs", {})
    items = [("capex_solar","Solar CapEx","#E8A020"),("capex_wind","Wind CapEx","#2D7DD2"),
             ("capex_batt","Batt CapEx","#27AE60"),("capex_therm","Therm CapEx","#8E6EC6"),
             ("opex_therm","Therm OpEx","#aaa"),("opex_voll","Unserved","#A31F34")]
    labels, vals, colors = [], [], []
    for k, lbl, clr in items:
        v = costs.get(k, 0)
        if v > 1: labels.append(lbl); vals.append(v); colors.append(clr)
    fig = go.Figure(go.Pie(labels=labels, values=vals, marker_colors=colors,
                            textinfo="percent+label",
                            hovertemplate="%{label}: $%{value:,.0f}/yr<extra></extra>"))
    fig.update_layout(title="Cost Breakdown", height=280, template="simple_white",
                       legend=dict(font=dict(size=10)),
                       margin=dict(t=50,b=10,l=10,r=10),
                       font=dict(family="Inter,system-ui", size=11))
    return fig

def plot_soc(res, zone_id) -> Optional[go.Figure]:
    T = res["T"]; hrs = np.arange(T); fig = go.Figure()
    for g in res["generators"]:
        if g["zone"] != zone_id or g["type"] != "battery": continue
        soc = np.asarray(res["gen_soc"].get(g["id"], np.zeros(T)))
        if soc.max() < 0.01: continue
        fig.add_trace(go.Scatter(x=hrs, y=soc, name=g.get("label","Battery"),
                                  mode="lines", line=dict(color="#27AE60", width=2),
                                  hovertemplate="SOC: %{y:.1f} MWh<extra></extra>"))
    if not fig.data: return None
    fig.update_layout(title=f"Battery SOC — {zone_id}", xaxis_title="Hour",
                       yaxis_title="MWh", template="simple_white", height=260,
                       margin=dict(t=50,b=40,l=50,r=10),
                       font=dict(family="Inter,system-ui", size=11))
    return fig

def plot_transfer(res) -> Optional[go.Figure]:
    xfer = res.get("transfer", {})
    if not xfer: return None
    T = res["T"]; hrs = np.arange(T); fig = go.Figure()
    for label, arr in xfer.items():
        fig.add_trace(go.Scatter(x=hrs, y=np.asarray(arr), name=label, mode="lines"))
    fig.add_hline(y=0, line_dash="dot", line_color="#ccc")
    fig.update_layout(title="Transmission Flows", xaxis_title="Hour", yaxis_title="MW",
                       template="simple_white", height=260,
                       margin=dict(t=50,b=40,l=50,r=10),
                       font=dict(family="Inter,system-ui", size=11))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Help dialogs
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("About", width="small")
def show_about():
    st.markdown("""
**Purpose**

This tool is built for **educational use** — to help students and researchers develop an intuition for how electricity systems are planned.

It implements a **linear programming capacity expansion model** that finds the least-cost mix of generation technologies to reliably meet electricity demand, given technology costs, resource availability, and policy constraints.

---
*Built with Python · linopy · Streamlit*

<div style="margin-top:16px;font-size:11px;color:#888;">
  Created by <strong>Juan Senga</strong> &nbsp;·&nbsp;
  <a href="mailto:jsenga@mit.edu" style="color:#A31F34;">jsenga@mit.edu</a>
</div>
""", unsafe_allow_html=True)


@st.dialog("Model Info", width="large")
def show_model_help():
    st.markdown("""
### Capacity Expansion Model

A **Capacity Expansion Model (CEM)** finds the least-cost mix of generation capacity to reliably meet electricity demand over a planning horizon, subject to engineering and policy constraints.

---

#### Objective Function

$$\\min \\sum_g \\underbrace{x_g^{\\text{new}} \\cdot \\text{CapEx}_g \\cdot \\text{CRF}_g}_{\\text{(1) Investment cost}} + \\underbrace{x_g^{\\text{total}} \\cdot \\text{FOM}_g}_{\\text{(2) Fixed O\\&M}} + \\underbrace{\\sum_t d_{g,t} \\cdot \\text{VOM}_{g,t}}_{\\text{(3) Variable O\\&M}} + \\underbrace{\\sum_{z,t} u_{z,t} \\cdot \\text{VOLL}}_{\\text{(4) Unserved demand}}$$

| Term | Description |
|---|---|
| $x_g^{\\text{new}}$ | New capacity built (MW) — CapEx charged **only here** |
| $x_g^{\\text{total}}$ | Total installed capacity (MW) — existing + new |
| $\\text{CRF}_g$ | Capital Recovery Factor: $\\frac{r(1+r)^n}{(1+r)^n - 1}$, where $r$ = WACC, $n$ = lifetime |
| $\\text{FOM}_g$ | Fixed O&M (\\$/MW-yr) on total installed capacity |
| $d_{g,t}$ | Dispatch (MWh) at hour $t$ |
| $\\text{VOM}_{g,t}$ | Variable cost (\\$/MWh) — fuel × heat rate + VOM for thermal |
| $u_{z,t}$ | Unserved demand (MWh) in zone $z$ at hour $t$ |
| $\\text{VOLL}$ | Value of Lost Load (\\$/MWh) — penalty for unserved demand |

---

#### Key Constraints

**Energy balance** (every zone $z$, every hour $t$):
$$\\sum_g d_{g,t}^z + u_{z,t} + \\sum_{z'} f_{z'z,t} = L_{z,t}$$

where $f_{z'z,t}$ is net transmission import from zone $z'$ and $L_{z,t}$ is load.

**Capacity limits**:
- $0 \\leq x_g^{\\text{total}} \\leq \\text{MaxMW}_g$
- $x_g^{\\text{total}} \\geq \\text{ExistingMW}_g$ if not retirable
- $d_{g,t} \\leq x_g^{\\text{total}}$ for thermal generators

**Solar / Wind**:
$$d_{g,t} = x_g^{\\text{total}} \\cdot \\text{CF}_{g,t} - \\text{curtailment}_{g,t}$$

**Thermal ramp rates** (if finite ramp specified):
$$|d_{g,t} - d_{g,t-1}| \\leq \\text{Ramp}_g$$

**Thermal minimum load**:
$$d_{g,t} \\geq \\text{MinLoad}_g \\cdot x_g^{\\text{total}}$$

**Battery state of charge** (SOC dynamics):
$$\\text{SOC}_{g,t} = (1-\\sigma_g)\\,\\text{SOC}_{g,t-1} + \\eta_g^c \\, b_{g,t}^+ - \\frac{b_{g,t}^-}{\\eta_g^d}$$
$$\\text{SOC}_{g,t} \\leq x_g^{\\text{total}} \\cdot \\text{duration}_g$$

**Transmission** (bidirectional, per link):
$$-\\text{LineLimit} \\leq f_{z_1 z_2,t} \\leq \\text{LineLimit}$$

**Renewable energy minimum** (optional policy):
$$\\sum_g \\sum_t d_{g,t}^{\\text{RE}} \\geq \\text{MinRE} \\times \\text{TotalLoad}$$

---

#### Generator Types

| Type | Investment | Variable cost | Notes |
|---|---|---|---|
| Solar | CapEx × CRF | VOM ($/MWh) | CF from climate profile |
| Wind | CapEx × CRF | VOM ($/MWh) | CF from climate profile |
| Gas / Coal / Nuclear | CapEx × CRF | Fuel × heat rate + VOM | Ramp & min-load constraints |
| Battery | CapEx × CRF | VOM on charge + discharge | SOC dynamics, round-trip efficiency |

**Existing capacity** — generators with `Existing MW > 0` start with that capacity at zero marginal CapEx. If `Expandable`, the optimizer may build beyond it. If `Retirable`, it may retire below it.
""")


@st.dialog("How to Use", width="large")
def show_instructions():
    st.markdown("""
### Getting Started

**1. Zone loads** — set demand for each zone in the **Zones** row below the canvas.

**2. Build your fleet** — in the **Generation Fleet** table, add generators per zone. Each row lets you set:
- **Existing MW** — pre-built capacity (no CapEx charged)
- **Expandable** ✓ — can the optimizer add more beyond existing?
- **Retirable** ✓ — can the optimizer retire below existing?
- **Max MW** — upper bound on total capacity
- **Capex $/MW** — cost of new capacity

**3. Advanced params** — click **⚙ Params** to expand type-specific fields (climate for solar/wind, heat rate for thermal, duration for batteries).

**4. Transmission** — add links between zones with capacity limits.

**5. Policy** — set the renewable energy minimum, VOLL, and project lifetime.

**6. Solve** — choose a time resolution and click **Solve**. Results appear below.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Load profile chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_load_profile(zone_name: str, profile: np.ndarray, T: int) -> go.Figure:
    hrs  = np.arange(T)
    avg  = profile.mean()
    peak = profile.max()

    # X-axis label depends on T
    if T == 24:
        xlabel = "Hour of day"
        # Annotate hours of day
        tickvals = list(range(0, 25, 4))
        ticktext = [f"{h:02d}:00" for h in tickvals]
    elif T == 168:
        xlabel = "Hour of week"
        tickvals = list(range(0, 169, 24))
        ticktext = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]
    else:
        xlabel = "Hour of year"
        tickvals = []; ticktext = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hrs, y=profile,
        mode="lines",
        name="Load",
        fill="tozeroy",
        fillcolor="rgba(163,31,52,0.10)",
        line=dict(color="#A31F34", width=1.5),
        hovertemplate="Hour %{x}: %{y:.1f} MW<extra></extra>",
    ))
    fig.add_hline(
        y=avg, line_dash="dot", line_color="#999",
        annotation_text=f" Avg {avg:.0f} MW",
        annotation_font_size=10, annotation_font_color="#888",
        annotation_position="right",
    )

    xaxis_cfg: dict = dict(title=xlabel)
    if tickvals:
        xaxis_cfg.update(tickvals=tickvals, ticktext=ticktext)

    fig.update_layout(
        title=dict(
            text=f"{zone_name} — Load Profile  "
                 f"<span style='font-size:11px;color:#888;'>"
                 f"peak {peak:.0f} MW &nbsp;·&nbsp; avg {avg:.0f} MW &nbsp;·&nbsp; "
                 f"load factor {avg/peak*100:.0f}%</span>",
            font=dict(size=13),
        ),
        xaxis=xaxis_cfg,
        yaxis=dict(title="MW", rangemode="tozero"),
        template="simple_white", height=210,
        margin=dict(t=50, b=40, l=55, r=80),
        font=dict(family="Inter,system-ui", size=11),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Zone load profile configuration
# ─────────────────────────────────────────────────────────────────────────────

def _render_zone_load_config(z: dict, T: int):
    """Left column: controls.  Right column: chart."""
    zid = z["id"]
    col_ctrl, col_chart = st.columns([1, 2.2])

    with col_ctrl:
        # Profile type selector
        _seed(f"lprof_{zid}", z.get("load_profile", "flat").title().replace("_", " "))
        chosen_label = st.selectbox(
            "Load shape", LOAD_PROFILES, key=f"lprof_{zid}")
        z["load_profile"] = LOAD_PROFILE_KEYS[chosen_label]

        # Peak load MW — shown for all presets; for CSV it acts as a fallback
        if z["load_profile"] != "upload_csv":
            _seed(f"lpeak_{zid}", float(z["load_mw"]))
            z["load_mw"] = st.number_input(
                "Peak load (MW)", min_value=1.0, step=10.0, key=f"lpeak_{zid}")
        else:
            st.markdown(
                "<span style='font-size:10px;color:#888;'>Peak MW is set by the uploaded values.</span>",
                unsafe_allow_html=True)

        # CSV upload
        if z["load_profile"] == "upload_csv":
            st.markdown(
                "<div style='font-size:10px;color:#555;margin-top:6px;margin-bottom:3px;'>"
                "Upload a single-column CSV with one MW value per row. "
                "Values are used as-is (no scaling).</div>",
                unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "CSV file", type=["csv"],
                key=f"lup_{zid}", label_visibility="collapsed")
            if uploaded is not None:
                try:
                    import io
                    df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8")), header=None)
                    raw = df.iloc[:, 0].values.astype(float)
                    z["load_data"] = raw.tolist()
                    z["load_mw"]   = float(np.max(raw))   # update peak MW
                    st.success(f"Loaded {len(raw):,} hourly values  (peak {z['load_mw']:.0f} MW)")
                except Exception as exc:
                    st.error(f"Could not parse CSV: {exc}")

            if z.get("load_data") is not None:
                n = len(z["load_data"])
                st.info(f"Uploaded: {n:,} values  (peak {z['load_mw']:.0f} MW)")
                if st.button("Clear upload", key=f"lclr_{zid}"):
                    z["load_data"] = None
                    z["load_mw"]   = 100.0
                    z["load_profile"] = "flat"
                    st.rerun()
            else:
                st.warning("No file uploaded yet — using Flat profile as fallback.")

        # Profile description blurb
        descriptions = {
            "flat":        "Constant demand every hour.",
            "residential": "Evening peak (~6–9 pm), overnight trough (~2–5 am). Weekend load ~20% lower.",
            "commercial":  "Business hours peak (9 am–6 pm), very low overnight. Weekend load ~22% lower.",
            "industrial":  "Near-flat with slight overnight dip. No major weekend effect.",
            "upload_csv":  "",
        }
        desc = descriptions.get(z["load_profile"], "")
        if desc:
            st.markdown(
                f"<div style='font-size:10px;color:#888;margin-top:6px;'>{desc}</div>",
                unsafe_allow_html=True)

    with col_chart:
        profile = generate_load_profile(
            profile_type=z["load_profile"],
            peak_mw=float(z["load_mw"]),
            T=T,
            custom_data=z.get("load_data"),
        )
        st.plotly_chart(plot_load_profile(z["name"], profile, T),
                        use_container_width=True)


def render_load_section():
    zones = st.session_state.zones
    T     = st.session_state.T

    st.markdown(
        '<div class="sec-hdr" style="border-left-color:#2980b9;color:#1a5a87;">'
        'ZONE LOAD PROFILES</div>',
        unsafe_allow_html=True)

    if len(zones) == 1:
        _render_zone_load_config(zones[0], T)
    else:
        tabs = st.tabs([z["name"] for z in zones])
        for tab, z in zip(tabs, zones):
            with tab:
                _render_zone_load_config(z, T)


# ─────────────────────────────────────────────────────────────────────────────
# Canvas section
# ─────────────────────────────────────────────────────────────────────────────

def render_canvas_section():
    zones = st.session_state.zones
    xmit  = st.session_state.transmission

    # Top controls
    n_zones = len(zones)
    col_widths = [1.2] * min(n_zones, 6) + [1.2]
    ctrl_cols  = st.columns(col_widths + [10 - sum(col_widths)])

    to_del = None
    for i, z in enumerate(zones[:6]):
        if ctrl_cols[i].button(
            f"× {z['name']}", key=f"zdel_{i}",
            help=f"Remove {z['name']}",
            disabled=(n_zones <= 1),
            use_container_width=True,
        ):
            to_del = i

    add_col = ctrl_cols[min(n_zones, 6)]
    if add_col.button("+ Zone", key="zadd", use_container_width=True):
        letter = chr(65 + n_zones)
        nz = _default_zone(f"Zone {letter}")
        nz["x"] = 0.15 + (n_zones * 0.22) % 0.70
        nz["y"] = 0.40 + (n_zones % 2) * 0.25
        st.session_state.zones.append(nz)
        st.session_state.last_results = None   # invalidate stale results
        st.rerun()

    if to_del is not None:
        did = st.session_state.zones[to_del]["id"]
        st.session_state.zones.pop(to_del)
        st.session_state.transmission = [
            l for l in st.session_state.transmission
            if l["from"] != did and l["to"] != did
        ]
        st.session_state.last_results = None   # invalidate stale results
        st.rerun()

    # Canvas
    st.markdown('<div class="card"><div class="card-hdr">ZONE NETWORK</div>'
                '<div style="padding:0;">', unsafe_allow_html=True)
    last_res = st.session_state.get("last_results")
    zones_data, edges_data = _canvas_data(zones, xmit, res=last_res)
    canvas_result = _zone_canvas(
        zones=zones_data, edges=edges_data,
        key=f"zone_canvas_{st.session_state.canvas_key}", default=None)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Handle drag-position updates (no widget keys involved — no rerun needed)
    if canvas_result is not None:
        cr = canvas_result
        if isinstance(cr, dict) and cr.get("type") == "move":
            for z in st.session_state.zones:
                if z["id"] == cr["id"]:
                    z["x"] = float(cr.get("x", z["x"]))
                    z["y"] = float(cr.get("y", z["y"]))
                    break
            # Position persisted in state; canvas already shows new position visually

# ─────────────────────────────────────────────────────────────────────────────
# Zone loads row
# ─────────────────────────────────────────────────────────────────────────────

def render_zones_row():
    zones = st.session_state.zones
    st.markdown('<div class="sec-hdr" style="border-left-color:#2980b9;color:#1a5a87;">ZONE LOADS</div>',
                unsafe_allow_html=True)
    n = len(zones)
    cols = st.columns([2] * n + [1])
    for i, z in enumerate(zones):
        with cols[i]:
            _seed(f"zload_{i}", float(z["load_mw"]))
            z["load_mw"] = st.number_input(
                f"{z['name']} — Load (MW)",
                min_value=1.0, step=10.0, key=f"zload_{i}")

# ─────────────────────────────────────────────────────────────────────────────
# Generation fleet table
# ─────────────────────────────────────────────────────────────────────────────

# Column ratios (must match between header and data rows)
_C = [1.0, 1.5, 0.85, 0.55, 0.55, 0.85, 0.7, 0.35]

def _fleet_header():
    cols = st.columns(_C)
    labels = ["TYPE", "LABEL", "EXIST. MW", "EXPAND.", "RETIRE.", "MAX MW", "PARAMS", ""]
    for col, lbl in zip(cols, labels):
        col.markdown(
            f"<div style='font-size:8px;font-weight:700;color:#999;"
            f"text-transform:uppercase;letter-spacing:.8px;padding:2px 0;'>{lbl}</div>",
            unsafe_allow_html=True)


def _render_generator_row(g: dict, z: dict) -> bool:
    """Render one generator as a table row. Returns True if user clicks Remove."""
    gid   = g["id"]
    gtype = g["type"]
    cfg   = GEN_CONFIG[gtype]
    color = cfg["color"]

    # Colored top-strip for each generator row — provides clear visual separation
    st.markdown(
        f'<div style="height:3px;background:linear-gradient(to right,{color}90,{color}18);'
        f'margin:2px 0 1px 0;border-radius:1px;"></div>',
        unsafe_allow_html=True)

    c_type, c_label, c_exist, c_expd, c_ret, c_max, c_adv, c_del = \
        st.columns(_C)

    # Type badge (display-only)
    c_type.markdown(
        f'<div class="gen-badge" style="background:{color}18;border:1px solid {color}40;">'
        f'<span class="gen-dot" style="background:{color};"></span>'
        f'<span style="color:{color};font-size:10px;font-weight:700;">{cfg["label"]}</span>'
        f'</div>',
        unsafe_allow_html=True)

    # Label
    _seed(f"gl_{gid}", g.get("label", cfg["label"]))
    g["label"] = c_label.text_input("Name", key=f"gl_{gid}", label_visibility="collapsed")

    # Existing MW
    _seed(f"gex_{gid}", float(g.get("existing_mw", 0)))
    g["existing_mw"] = c_exist.number_input("Existing MW", min_value=0.0, step=10.0,
                                             key=f"gex_{gid}", label_visibility="collapsed")

    # Expandable checkbox
    c_expd.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _seed(f"gexpd_{gid}", bool(g.get("expandable", True)))
    g["expandable"] = c_expd.checkbox("Expandable", key=f"gexpd_{gid}", label_visibility="collapsed")

    # Retirable checkbox
    c_ret.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    _seed(f"gret_{gid}", bool(g.get("retirable", False)))
    g["retirable"] = c_ret.checkbox("Retirable", key=f"gret_{gid}", label_visibility="collapsed")

    # Max MW — shows ∞ symbol when global uncap_all is active, else normal input
    if st.session_state.get("uncap_all", False):
        c_max.markdown(
            "<div style='font-size:18px;font-weight:700;color:#bbb;"
            "text-align:center;padding-top:2px;' title='No maximum capacity'>∞</div>",
            unsafe_allow_html=True)
        g["max_mw"] = g.get("max_mw", 500.0)
    else:
        _seed(f"gmx_{gid}", float(g.get("max_mw", 500.0)))
        g["max_mw"] = c_max.number_input("Max MW", min_value=0.0, step=50.0,
                                          key=f"gmx_{gid}", label_visibility="collapsed")

    # Params toggle
    _seed(f"gshow_{gid}", False)
    if c_adv.button("⚙ Params", key=f"gadv_{gid}", use_container_width=True):
        st.session_state[f"gshow_{gid}"] = not st.session_state.get(f"gshow_{gid}", False)
        st.rerun()

    # Remove
    if c_del.button("✕", key=f"grem_{gid}"):
        return True

    # Battery MWh info line (always visible for batteries)
    if gtype == "battery":
        dur = float(g.get("duration", 4.0))
        mwh = g["max_mw"] * dur
        st.markdown(
            f'<div style="padding:1px 6px 3px;font-size:10px;color:#27AE60;font-weight:600;">'
            f'{g["max_mw"]:.0f} MW &nbsp;·&nbsp; {mwh:,.0f} MWh storage capacity</div>',
            unsafe_allow_html=True)

    # Type-specific params (shown when toggled)
    if st.session_state.get(f"gshow_{gid}", False):
        # Top bar — labels this as an expanded params panel
        st.markdown(
            f'<div style="background:{color}18;border-left:4px solid {color};'
            f'border-top:1px solid {color}40;'
            f'padding:5px 12px 5px 12px;margin:0;">'
            f'<span style="font-size:9px;font-weight:700;letter-spacing:.9px;'
            f'text-transform:uppercase;color:{color};">'
            f'{cfg["label"]} parameters</span></div>',
            unsafe_allow_html=True)
        # Inner content wrapper — left rule keeps visual context
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:10px 14px 2px 18px;'
            f'background:{color}08;margin:0;">',
            unsafe_allow_html=True)
        _render_type_params(g, gtype, gid)
        st.markdown('</div>', unsafe_allow_html=True)
        # Bottom close bar — clearly terminates the panel
        st.markdown(
            f'<div style="height:4px;background:{color}30;'
            f'border-left:4px solid {color};margin:0 0 4px 0;"></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='height:3px;'></div>",
            unsafe_allow_html=True)
    return False


def _hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _plot_cf_preview(cf: np.ndarray, color: str, label: str) -> go.Figure:
    """Compact 168-h capacity factor preview chart."""
    hrs = np.arange(min(168, len(cf)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hrs, y=cf[:168], mode="lines",
        line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(color),
        hovertemplate="Hour %{x}: CF=%{y:.3f}<extra></extra>",
        name=label,
    ))
    fig.update_layout(
        height=160, template="simple_white",
        margin=dict(t=28, b=28, l=40, r=8),
        title=dict(text=f"{label} — capacity factor (first 168 h)", font=dict(size=11)),
        xaxis=dict(title="Hour", tickfont=dict(size=9)),
        yaxis=dict(title="CF", range=[0, 1], tickfont=dict(size=9)),
        showlegend=False,
        font=dict(family="Inter,system-ui", size=10),
    )
    return fig


def _render_lifetime_fom(g: dict, gid: str, default_lt: float, default_fom: float,
                         default_capex: float = 0.0, default_wacc: float = 0.08):
    """CapEx | WACC | Lifetime | Fixed O&M — four-column financial row."""
    st.markdown(
        "<div style='height:1px;background:rgba(0,0,0,0.07);margin:8px 0 6px;'></div>",
        unsafe_allow_html=True)
    lf1, lf2, lf3, lf4 = st.columns(4)
    _seed(f"gcp_{gid}",   float(g.get("capex",    default_capex)))
    _seed(f"gwacc_{gid}", float(g.get("wacc",     default_wacc) * 100))
    _seed(f"glt_{gid}",   float(g.get("lifetime", default_lt)))
    _seed(f"gfom_{gid}",  float(g.get("fom",      default_fom)))
    g["capex"]    = lf1.number_input("CapEx ($/MW)",
        min_value=0.0, step=50_000.0, format="%.0f", key=f"gcp_{gid}",
        help="Capital cost per MW of new capacity built")
    g["wacc"]     = lf2.number_input("WACC (%)",
        min_value=0.0, max_value=30.0, step=0.5, format="%.1f", key=f"gwacc_{gid}",
        help="Weighted Average Cost of Capital — used for CapEx annualization via CRF") / 100
    g["lifetime"] = lf3.number_input("Lifetime (yr)",
        min_value=1.0, max_value=60.0, step=1.0, key=f"glt_{gid}")
    g["fom"]      = lf4.number_input("Fixed O&M ($/MW-yr)",
        min_value=0.0, step=1_000.0, format="%.0f", key=f"gfom_{gid}")

    # Derived: annualized investment cost (CapEx × CRF)
    invest_cost = g["capex"] * _crf(g["wacc"], g["lifetime"])
    st.markdown(
        f'<div style="font-size:11px;color:#555;margin:4px 0 0 2px;">'
        f'Investment cost: <strong style="color:#14213d;">'
        f'${invest_cost:,.0f} /MW-yr</strong>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#888;">(CapEx × CRF at {g["wacc"]*100:.1f}% over {g["lifetime"]:.0f} yr)</span>'
        f'</div>',
        unsafe_allow_html=True)


def _render_type_params(g: dict, gtype: str, gid: str):
    """Render type-specific parameter inputs in a sub-row."""
    if gtype == "solar":
        sc1, sc2 = st.columns([2, 1])
        _seed(f"gclim_{gid}", g.get("climate", "Temperate / Midwest"))
        g["climate"] = sc1.selectbox(
            "Climate profile (solar)",
            SOLAR_CLIMATES, key=f"gclim_{gid}")
        _seed(f"gvom_{gid}", float(g.get("vom", 0.0)))
        g["vom"] = sc2.number_input("VOM ($/MWh)", min_value=0.0, step=0.5,
                                    key=f"gvom_{gid}")
        cf = generate_solar_cf(climate=g["climate"], T=168)
        st.plotly_chart(_plot_cf_preview(cf, "#E8A020", "Solar"), use_container_width=True,
                        config={"displayModeBar": False}, key=f"cf_solar_{gid}")
        _render_lifetime_fom(g, gid, 30.0, 17_000.0, default_capex=1_200_000.0)

    elif gtype == "wind":
        wc1, wc2 = st.columns([2, 1])
        _seed(f"gclim_{gid}", g.get("climate", "Temperate / Midwest"))
        g["climate"] = wc1.selectbox(
            "Climate profile (wind)",
            WIND_CLIMATES, key=f"gclim_{gid}")
        _seed(f"gvom_{gid}", float(g.get("vom", 0.0)))
        g["vom"] = wc2.number_input("VOM ($/MWh)", min_value=0.0, step=0.5,
                                    key=f"gvom_{gid}")
        cf = generate_wind_cf(climate=g["climate"], T=168)
        st.plotly_chart(_plot_cf_preview(cf, "#2D7DD2", "Wind"), use_container_width=True,
                        config={"displayModeBar": False}, key=f"cf_wind_{gid}")
        _render_lifetime_fom(g, gid, 25.0, 43_000.0, default_capex=1_500_000.0)

    elif gtype in ("gas", "nuclear", "coal"):
        default_co2 = EMISSIONS_KG_MMBTU.get(gtype, 0.0)
        has_combustion = gtype in ("gas", "coal")

        if has_combustion:
            tc1, tc2, tc3, tc4, tc5, tc6 = st.columns([1, 1, 1, 1, 1, 1])
        else:
            tc1, tc2, tc3, tc4, tc5 = st.columns([1, 1, 1, 1, 1])

        _seed(f"ghr_{gid}",  float(g.get("heat_rate",  6.5)))
        _seed(f"gfc_{gid}",  float(g.get("fuel_cost",  4.0)))
        _seed(f"gvom_{gid}", float(g.get("vom",        4.0)))
        _seed(f"grmp_{gid}", float(g.get("ramp",       1_000_000.0)))
        _seed(f"gml_{gid}",  float(g.get("min_load",   0.0)))
        g["heat_rate"] = tc1.number_input("Heat rate (MMBtu/MWh)",
            min_value=1.0, max_value=20.0, step=0.1, key=f"ghr_{gid}")
        g["fuel_cost"] = tc2.number_input("Fuel cost ($/MMBtu)",
            min_value=0.0, step=0.25, key=f"gfc_{gid}")
        g["vom"]       = tc3.number_input("VOM ($/MWh)",
            min_value=0.0, step=1.0, key=f"gvom_{gid}")
        g["ramp"]      = tc4.number_input("Ramp (MW/h)",
            min_value=0.0, step=10.0, key=f"grmp_{gid}",
            help="Very large = no ramp constraint")
        g["min_load"]  = tc5.number_input("Min load (0–1)",
            min_value=0.0, max_value=1.0, step=0.05, key=f"gml_{gid}")
        if has_combustion:
            _seed(f"gco2_{gid}", float(g.get("co2_factor", default_co2)))
            g["co2_factor"] = tc6.number_input(
                "CO₂ (kg/MMBtu)",
                min_value=0.0, max_value=200.0, step=0.5,
                key=f"gco2_{gid}",
                help="EPA combustion factor: Gas≈53, Coal≈95")

        vc = g["fuel_cost"] * g["heat_rate"] + g["vom"]
        st.caption(f"Variable cost: **${vc:.1f}/MWh**")
        default_lt   = {"gas": 30.0,       "nuclear": 40.0,       "coal": 30.0}[gtype]
        default_fom  = {"gas": 12_000.0,   "nuclear": 100_000.0,  "coal": 40_000.0}[gtype]
        default_capx = {"gas": 900_000.0,  "nuclear": 7_000_000.0,"coal": 4_000_000.0}[gtype]
        _render_lifetime_fom(g, gid, default_lt, default_fom, default_capex=default_capx)

    elif gtype == "battery":
        bc1, bc2, bc3, bc4 = st.columns([1, 1, 1, 1])
        _seed(f"gdur_{gid}",  float(g.get("duration",       4.0)))
        _seed(f"getac_{gid}", float(g.get("eta_charge",     0.96) * 100))
        _seed(f"getad_{gid}", float(g.get("eta_discharge",  0.96) * 100))
        _seed(f"gsd_{gid}",   float(g.get("self_discharge", 0.0002)))
        g["duration"]       = bc1.number_input("Duration (h)",
            min_value=0.5, max_value=24.0, step=0.5, key=f"gdur_{gid}")
        g["eta_charge"]     = bc2.number_input("Charge eff (%)",
            min_value=50.0, max_value=100.0, step=1.0, key=f"getac_{gid}") / 100
        g["eta_discharge"]  = bc3.number_input("Discharge eff (%)",
            min_value=50.0, max_value=100.0, step=1.0, key=f"getad_{gid}") / 100
        g["self_discharge"] = bc4.number_input("Self-disch (/h)",
            min_value=0.0, max_value=0.01, step=0.0001, format="%.4f", key=f"gsd_{gid}")
        rte = g["eta_charge"] * g["eta_discharge"] * 100
        mwh = g["max_mw"] * g["duration"]
        st.caption(f"Round-trip efficiency: **{rte:.1f}%** · Max storage: **{mwh:,.0f} MWh**")
        _render_lifetime_fom(g, gid, 15.0, 20_000.0, default_capex=300_000.0)


def render_fleet_section():
    zones = st.session_state.zones
    st.markdown('<div class="sec-hdr">GENERATION FLEET</div>', unsafe_allow_html=True)

    # ── Global no-cap toggle ──────────────────────────────────────────────────
    st.checkbox(
        "No maximum capacity",
        key="uncap_all",
        help="Remove the MAX MW limit for every generator across all zones.")

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    to_remove   = None
    gen_to_add  = None   # (zone_id, gtype)

    for z_idx, z in enumerate(zones):
        n_gens = len(z["generators"])
        label  = (f"{z['name']}  ·  "
                  f"{n_gens} generator{'s' if n_gens != 1 else ''}  ·  "
                  f"{round(z['load_mw'])} MW peak load")
        with st.expander(label, expanded=True):
            _fleet_header()
            for g in list(z["generators"]):
                if _render_generator_row(g, z):
                    to_remove = (z["id"], g["id"])

            # ── Per-zone add control ───────────────────────────────────────
            st.markdown(
                "<div style='height:4px;border-top:1px dashed #e0e0e0;margin:6px 0 4px;'></div>",
                unsafe_allow_html=True)
            ac1, ac2, ac3 = st.columns([2, 1, 5])
            _seed(f"add_type_{z['id']}", GEN_TYPES_DISPLAY[0])
            chosen_type = ac1.selectbox(
                "Type", GEN_TYPES_DISPLAY,
                key=f"add_type_{z['id']}", label_visibility="collapsed")
            if ac2.button("+ Add", key=f"add_gen_{z['id']}", use_container_width=True):
                gen_to_add = (z["id"], _DISPLAY_TO_KEY[chosen_type])

    # Apply removals and additions outside loop to avoid mutation mid-iteration
    if to_remove:
        z_id, g_id = to_remove
        for z in zones:
            if z["id"] == z_id:
                z["generators"] = [g for g in z["generators"] if g["id"] != g_id]
                break
        st.rerun()

    if gen_to_add:
        target_zid, gtype = gen_to_add
        for z in zones:
            if z["id"] == target_zid:
                existing_count = sum(
                    1 for zz in zones for g in zz["generators"] if g["type"] == gtype
                )
                zone_short = z["name"].replace(" ", "")
                auto_label = f"{TYPE_SHORT_LABEL[gtype]}_{existing_count + 1}_{zone_short}"
                z["generators"].append(
                    _new_generator(gtype,
                                   z.get("solar_climate", "Temperate / Midwest"),
                                   z.get("wind_climate",  "Temperate / Midwest"),
                                   label=auto_label))
                break
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Transmission section
# ─────────────────────────────────────────────────────────────────────────────

def render_transmission_section():
    """Transmission links — rendered inside an expander in main()."""
    xmit     = st.session_state.transmission
    zone_ids = [z["id"] for z in st.session_state.zones]

    if not xmit:
        if len(zone_ids) >= 2:
            st.caption("No transmission links yet — zones operate independently.")
        else:
            st.caption("Add at least 2 zones to enable transmission links.")

    to_del_x = None
    for xidx, link in enumerate(xmit):
        xc1, xc2, xc3 = st.columns([3, 2, 1])
        xc1.markdown(f"**{link['from']}** ↔ **{link['to']}**")
        _seed(f"xcap_{xidx}", float(link["cap_mw"]))
        link["cap_mw"] = xc2.number_input(
            "Cap (MW)", min_value=0.0, step=10.0,
            key=f"xcap_{xidx}", label_visibility="collapsed")
        if xc3.button("✕", key=f"xdel_{xidx}"):
            to_del_x = xidx
    if to_del_x is not None:
        xmit.pop(to_del_x)
        st.rerun()

    if len(zone_ids) >= 2:
        st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
        la1, la2, la3, la4 = st.columns([2, 2, 2, 1])
        from_z = la1.selectbox("From zone", zone_ids, key="xfrom",
                                label_visibility="collapsed")
        to_z   = la2.selectbox("To zone",   zone_ids, key="xto",
                                label_visibility="collapsed")
        _seed("xcap_new", 50.0)
        cap = la3.number_input("Capacity (MW)", min_value=0.0, step=10.0,
                               key="xcap_new", label_visibility="collapsed")
        if la4.button("+ Add", key="xadd", use_container_width=True):
            if from_z != to_z and not any(
                (l["from"] == from_z and l["to"] == to_z) or
                (l["from"] == to_z   and l["to"] == from_z)
                for l in xmit
            ):
                xmit.append({"from": from_z, "to": to_z, "cap_mw": cap})
                st.rerun()
    else:
        st.caption("Add at least 2 zones to define transmission links.")


# ─────────────────────────────────────────────────────────────────────────────
# Policy section
# ─────────────────────────────────────────────────────────────────────────────

def render_policy_section():
    """Policy inputs — rendered inside an expander in main()."""
    policy = st.session_state.policy

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    _seed("mre",  float(policy["min_re"]))
    _seed("voll", float(policy["voll"]))
    policy["min_re"] = pc1.number_input(
        "Min RE (0–1)",
        min_value=0.0, max_value=1.0, step=0.05, key="mre",
        help="Fraction of annual energy that must come from solar + wind")
    policy["voll"] = pc2.number_input(
        "VOLL ($/MWh)",
        min_value=0.0, step=500.0, key="voll",
        help="Value of Lost Load — penalty for each MWh of unserved demand")

    re_pct = policy["min_re"] * 100
    pc3.markdown(
        f'<div style="font-size:11px;color:#555;padding-top:22px;">'
        f'Minimum <strong>{re_pct:.0f}%</strong> of annual energy from solar + wind. '
        f'Unserved demand penalized at <strong>${policy["voll"]:,.0f}/MWh</strong>.'
        f'</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Solve bar
# ─────────────────────────────────────────────────────────────────────────────

def _solver_worker(inputs: dict, result_list: list, error_list: list,
                   stop_event: threading.Event):
    """Background thread: run solver and deposit result into shared lists."""
    try:
        result = build_and_solve(inputs)
        if not stop_event.is_set():
            result_list.append(result)
    except Exception as exc:
        if not stop_event.is_set():
            error_list.append(exc)


def render_solve_bar():
    """Always-visible solve bar — settings, threaded solve, stop button, results trigger."""
    st.markdown(
        '<div class="sec-hdr" style="border-left-color:var(--red);color:var(--red);">'
        'SOLVER &amp; SOLVE</div>',
        unsafe_allow_html=True)

    solving = st.session_state.get("_solving", False)

    # ── While solver is running ────────────────────────────────────────────────
    if solving:
        thread: threading.Thread = st.session_state.get("_solver_thread")
        if thread and thread.is_alive():
            # Show live status + stop button
            info_col, stop_col = st.columns([5, 1])
            info_col.markdown(
                f'<div style="padding:8px 0;font-size:13px;color:#555;">'
                f'⏳ <strong>Solving…</strong> &nbsp;'
                f'T = {st.session_state.T:,} h &nbsp;·&nbsp; '
                f'{st.session_state.solver.upper()}</div>',
                unsafe_allow_html=True)
            if stop_col.button("⏹ Stop", key="stop_btn", use_container_width=True):
                ev: threading.Event = st.session_state.get("_stop_event")
                if ev:
                    ev.set()
                st.session_state["_solving"]       = False
                st.session_state["_solver_thread"] = None
                st.rerun()
            time.sleep(0.4)   # brief poll delay before rechecking
            st.rerun()
            return

        # Thread finished — collect result
        st.session_state["_solving"] = False
        result_list = st.session_state.pop("_solve_result", [])
        error_list  = st.session_state.pop("_solve_error",  [])
        scenario_nm = st.session_state.pop("_pending_scenario", "")
        st.session_state["_solver_thread"] = None

        if error_list:
            st.error(f"**Solver error:** {error_list[0]}")
            st.exception(error_list[0])
        elif result_list:
            results = result_list[0]
            if results.get("status") != "optimal":
                st.error(f"Solver returned: **{results.get('status')}**. "
                         "Try relaxing constraints or increasing the time limit.")
            else:
                st.session_state.last_results = results
                if scenario_nm:
                    st.session_state.scenarios[scenario_nm] = copy.deepcopy(results)
                st.session_state.canvas_key += 1
        st.rerun()
        return

    # ── Normal (idle) state ────────────────────────────────────────────────────
    sb1, sb2, sb3, sb4, sb5, sb6 = st.columns([1.4, 3.0, 0.9, 1.5, 1.2, 1.2])

    solver_opts = {"HiGHS (free)": "highs", "Gurobi (license req.)": "gurobi"}
    chosen_solver = sb1.selectbox(
        "Solver engine", list(solver_opts.keys()),
        key="solver_sel", label_visibility="collapsed")
    st.session_state.solver = solver_opts[chosen_solver]

    T_opts = {"24 h — Instant": 24, "168 h — 1 week": 168, "8 760 h — Full year": 8_760}
    chosen_T = sb2.radio(
        "Time resolution", list(T_opts.keys()),
        index=0, key="T_sel", horizontal=True)
    st.session_state.T = T_opts[chosen_T]

    _seed("tlim", int(st.session_state.time_limit))
    st.session_state.time_limit = sb3.number_input(
        "Limit (s)", min_value=30, max_value=600, step=30, key="tlim")

    scenario_name = sb4.text_input(
        "Scenario name", value="Scenario 1", key="scen_name")

    solve = sb5.button(
        "▶  Solve", type="primary", use_container_width=True, key="solve_btn")

    res = st.session_state.last_results
    results_ready = res is not None
    if sb6.button(
        "📊 Results", use_container_width=True,
        key="view_results_btn", disabled=not results_ready,
        help="View optimization results" if results_ready else "Run Solve first",
    ):
        show_results_dialog()

    if st.session_state.T == 8_760:
        st.warning("Full-year runs may take several minutes.", icon="⏱")

    # ── Launch solver thread ───────────────────────────────────────────────────
    if solve:
        try:
            inputs = assemble_inputs()
        except Exception as exc:
            st.error(f"**Input error:** {exc}")
            return
        stop_event  = threading.Event()
        result_list: list = []
        error_list:  list = []
        thread = threading.Thread(
            target=_solver_worker,
            args=(inputs, result_list, error_list, stop_event),
            daemon=True,
        )
        thread.start()
        st.session_state["_solving"]          = True
        st.session_state["_solver_thread"]    = thread
        st.session_state["_stop_event"]       = stop_event
        st.session_state["_solve_result"]     = result_list
        st.session_state["_solve_error"]      = error_list
        st.session_state["_pending_scenario"] = scenario_name
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Results dialog
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("Results", width="large")
def show_results_dialog():
    """Pop-up dialog with all optimization results."""
    res = st.session_state.last_results
    if res is None:
        st.info("No results yet — configure your fleet and click **Solve**.")
        return

    # Warn if zone config has changed since the last solve
    current_zone_ids = set(z["id"] for z in st.session_state.zones)
    solved_zone_ids  = set(res["zones"])
    if current_zone_ids != solved_zone_ids:
        st.warning(
            "Zone configuration has changed since the last solve — "
            "results may be **outdated**. Run Solve again to update.",
            icon=None,
        )

    zones_r = res["zones"]
    T       = res["T"]
    scale   = res["scale"]
    costs   = res.get("costs", {})

    total_load_mwh = sum(float(np.sum(res["dc_load"][z])) * scale for z in zones_r)
    total_re_mwh   = sum(
        (_zone_type_disp(res, z, "solar").sum() + _zone_type_disp(res, z, "wind").sum()) * scale
        for z in zones_r)
    re_pct = total_re_mwh / total_load_mwh * 100 if total_load_mwh else 0.0

    st.markdown(f"""
<div class="results-hdr">
  <div class="results-eyebrow">OPTIMISATION RESULT</div>
  <div class="results-title">Optimal solution found</div>
  <div class="results-stats">
    Total cost: <strong>${costs.get('total',0)/1e6:.1f}M/yr</strong>
    &nbsp;·&nbsp; RE share: <strong>{re_pct:.1f}%</strong>
    &nbsp;·&nbsp; Solve time: <strong>{res.get('solve_time',0):.1f}s</strong>
    &nbsp;·&nbsp; T = {T:,} h
  </div>
</div>""", unsafe_allow_html=True)

    # KPI row
    k_cols = st.columns(len(GEN_CONFIG) + 1)
    for col, (gtype, cfg) in zip(k_cols, GEN_CONFIG.items()):
        total_cap = sum(_zone_cap_by_type(res, z, gtype) for z in zones_r)
        col.metric(cfg["label"], f"{total_cap:,.0f} MW")
    k_cols[-1].metric("System Cost", f"${costs.get('total',0)/1e6:.1f}M/yr")

    # Charts
    zone_sel = st.selectbox("Inspect zone:", zones_r, key="rzone")
    max_h    = T - 1
    h_range  = st.slider("Hour window", 0, max_h, (0, min(167, max_h)), key="hrng")

    rc1, rc2 = st.columns(2)
    rc1.plotly_chart(plot_dispatch(res, zone_sel, h_range[0], h_range[1]),
                     use_container_width=True)
    rc2.plotly_chart(plot_capacity_mix(res), use_container_width=True)

    # Zonal cost breakdown (full-width)
    st.plotly_chart(plot_zone_cost_breakdown(res), use_container_width=True)

    # Zonal cost table
    breakdown = _compute_zone_cost_breakdown(res)
    _COMPS = ["Solar CapEx", "Wind CapEx", "Battery CapEx",
              "Thermal CapEx", "Fixed O&M", "Thermal OpEx", "Unserved", "Total"]
    df_zc = pd.DataFrame(
        {z: {c: f"${breakdown.get(z, {}).get(c, 0)/1e6:.2f}M" for c in _COMPS}
         for z in zones_r}
    ).T.reset_index().rename(columns={"index": "Zone"})
    st.dataframe(df_zc, use_container_width=True, hide_index=True)

    rc3, rc4 = st.columns(2)
    rc3.plotly_chart(plot_cost_breakdown(res), use_container_width=True)
    soc_fig  = plot_soc(res, zone_sel)
    xfer_fig = plot_transfer(res)
    if soc_fig:  rc4.plotly_chart(soc_fig,  use_container_width=True)
    if xfer_fig: rc4.plotly_chart(xfer_fig, use_container_width=True)

    # Cost table
    st.markdown('<div class="card"><div class="card-hdr">COST SUMMARY</div>'
                '<div style="padding:12px 14px;">', unsafe_allow_html=True)
    cost_rows = [
        ("Solar CapEx",     costs.get("capex_solar", 0)),
        ("Wind CapEx",      costs.get("capex_wind",  0)),
        ("Battery CapEx",   costs.get("capex_batt",  0)),
        ("Thermal CapEx",   costs.get("capex_therm", 0)),
        ("Fixed O&M",       costs.get("fom_total",   0)),
        ("Thermal OpEx",    costs.get("opex_therm",  0)),
        ("Unserved (VOLL)", costs.get("opex_voll",   0)),
        ("TOTAL",           costs.get("total",       0)),
    ]
    df_c = pd.DataFrame(cost_rows, columns=["Component", "$/yr"])
    df_c["$/yr"] = df_c["$/yr"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(df_c, use_container_width=True, hide_index=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Download
    export_csv = _make_export_csv(res)
    st.download_button(
        "⬇ Download full inputs & results (CSV)", export_csv,
        file_name="cem_export.csv", mime="text/csv",
        use_container_width=True)

    # Solver log (collapsed by default)
    solver_log = res.get("solver_log", "")
    with st.expander("Solver log", expanded=False):
        if solver_log and solver_log.strip():
            st.code(solver_log, language=None)
        else:
            st.caption("No log captured.")

    # Scenario comparison (only if ≥2 scenarios saved)
    if len(st.session_state.scenarios) >= 2:
        st.markdown("---")
        st.markdown('<div class="card-hdr">SCENARIO COMPARISON</div>',
                    unsafe_allow_html=True)
        scens = st.session_state.scenarios
        sel   = st.multiselect("Compare scenarios:", list(scens.keys()),
                                default=list(scens.keys())[:4], key="scmp")
        if len(sel) >= 2:
            subset = {k: scens[k] for k in sel}
            sc1, sc2 = st.columns(2)
            fig_s = go.Figure()
            for gtype, cfg in GEN_CONFIG.items():
                vals = [sum(_zone_cap_by_type(subset[s], z, gtype)
                            for z in subset[s]["zones"]) for s in sel]
                if max(vals, default=0) > 0.1:
                    fig_s.add_trace(go.Bar(name=cfg["label"], x=sel, y=vals,
                                            marker_color=_DISPATCH_COLORS[gtype]))
            fig_s.update_layout(title="Capacity (MW)", barmode="stack",
                                 template="simple_white", height=260,
                                 margin=dict(t=50,b=40,l=50,r=10),
                                 font=dict(family="Inter,system-ui", size=11))
            sc1.plotly_chart(fig_s, use_container_width=True)
            fig_c = go.Figure()
            for k, lbl, clr in [
                ("capex_solar","Solar CapEx","#E8A020"),("capex_wind","Wind CapEx","#2D7DD2"),
                ("capex_batt","Batt CapEx","#27AE60"),("capex_therm","Therm CapEx","#8E6EC6"),
                ("opex_therm","Therm OpEx","#aaa"),("opex_voll","Unserved","#A31F34"),
            ]:
                vals = [subset[s].get("costs",{}).get(k,0)/1e6 for s in sel]
                if max(vals,default=0) > 0:
                    fig_c.add_trace(go.Bar(name=lbl, x=sel, y=vals, marker_color=clr))
            fig_c.update_layout(title="Cost (M$/yr)", barmode="stack",
                                 template="simple_white", height=260,
                                 margin=dict(t=50,b=40,l=50,r=10),
                                 font=dict(family="Inter,system-ui", size=11))
            sc2.plotly_chart(fig_c, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_state()

    # ── Sticky header — st.columns() row styled as the red bar via CSS ────────
    _hc_title, _hc_gap, _hc_minfo, _hc_about, _hc_instr = \
        st.columns([7, 3, 1.4, 1.1, 1.5])
    _hc_title.markdown(
        '<p class="cem-hdr-title">Capacity Expansion Modeling</p>',
        unsafe_allow_html=True)
    if _hc_minfo.button("Model Info", key="btn_minfo", use_container_width=True):
        show_model_help()
    if _hc_about.button("About", key="btn_about", use_container_width=True):
        show_about()
    if _hc_instr.button("Instructions", key="btn_instr", use_container_width=True):
        show_instructions()

    render_solve_bar()
    st.markdown("<div style='height:2px;'></div>", unsafe_allow_html=True)

    with st.expander("Zone Network", expanded=True):
        render_canvas_section()

    with st.expander("Transmission Links", expanded=False):
        render_transmission_section()

    with st.expander("Zone Load Profiles", expanded=False):
        render_load_section()

    with st.expander("Generation Fleet", expanded=True):
        render_fleet_section()

    with st.expander("Policy", expanded=False):
        render_policy_section()


if __name__ == "__main__":
    main()
