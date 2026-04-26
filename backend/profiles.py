"""
backend/profiles.py
===================
Generate synthetic hourly capacity-factor profiles for solar and wind,
and time-varying load profiles for different demand archetypes.

Solar model:  CF(h) = bell_curve(hour) × seasonal_amplitude × cloud_noise
Wind model:   AR(1) process with seasonal mean shift, clipped to [0, 1]
Load model:   24-h archetype shape × seasonal multiplier (repeated to fill T)
"""

import numpy as np

# ── Solar parameter table ─────────────────────────────────────────────────────

SOLAR_PARAMS = {
    "Desert / Southwest":   dict(peak_cf=1.00, noise_alpha=9,  noise_beta=1),
    "Sunny / Southeast":    dict(peak_cf=0.88, noise_alpha=6,  noise_beta=2),
    "Temperate / Midwest":  dict(peak_cf=0.80, noise_alpha=4,  noise_beta=2),
    "Cloudy / Northwest":   dict(peak_cf=0.62, noise_alpha=3,  noise_beta=3),
}

# ── Wind parameter table ──────────────────────────────────────────────────────

WIND_PARAMS = {
    "Coastal / Offshore":   dict(mean_cf=0.45, sigma=0.08, ar1=0.90, seasonal_amp=0.12),
    "Windy Plains":         dict(mean_cf=0.40, sigma=0.08, ar1=0.88, seasonal_amp=0.15),
    "Temperate / Midwest":  dict(mean_cf=0.30, sigma=0.07, ar1=0.86, seasonal_amp=0.12),
    "Low Wind / Southeast": dict(mean_cf=0.20, sigma=0.06, ar1=0.85, seasonal_amp=0.08),
}

SOLAR_CLIMATES = list(SOLAR_PARAMS.keys())
WIND_CLIMATES  = list(WIND_PARAMS.keys())

# ── Load profile archetype table ──────────────────────────────────────────────

# 24-hour demand shapes (fraction of peak).  Hour 0 = midnight.
_LOAD_SHAPES: dict[str, np.ndarray] = {
    "flat": np.ones(24),

    # Residential: overnight trough, morning ramp, midday dip, strong evening peak.
    "residential": np.array([
        0.52, 0.47, 0.43, 0.41, 0.43, 0.52,   # 0–5  h  overnight
        0.65, 0.80, 0.82, 0.76, 0.71, 0.69,   # 6–11 h  morning
        0.68, 0.67, 0.68, 0.72, 0.80, 0.95,   # 12–17 h midday → early evening
        1.00, 0.98, 0.91, 0.82, 0.70, 0.59,   # 18–23 h peak → decline
    ]),

    # Commercial: very low overnight, sharp 7am rise, flat business hours, quick drop.
    "commercial": np.array([
        0.30, 0.28, 0.27, 0.27, 0.28, 0.33,   # 0–5  h  minimal overnight
        0.48, 0.70, 0.88, 0.96, 1.00, 1.00,   # 6–11 h  ramp to full
        0.96, 0.95, 0.99, 0.98, 0.91, 0.77,   # 12–17 h business hours
        0.56, 0.45, 0.40, 0.37, 0.34, 0.31,   # 18–23 h rapid drop
    ]),

    # Industrial: relatively flat, modest overnight dip, slight day increase.
    "industrial": np.array([
        0.76, 0.74, 0.73, 0.73, 0.74, 0.76,   # 0–5  h
        0.84, 0.94, 1.00, 1.00, 1.00, 1.00,   # 6–11 h
        0.97, 0.98, 1.00, 1.00, 0.98, 0.95,   # 12–17 h
        0.92, 0.88, 0.85, 0.82, 0.79, 0.77,   # 18–23 h
    ]),
}

# Monthly seasonal multipliers (Jan–Dec) — same for all preset archetypes.
# Captures summer-AC and winter-heating dual peaks.
_SEASONAL_MULT = np.array([
    0.90, 0.87, 0.83, 0.79, 0.81, 0.93,   # Jan–Jun
    1.00, 0.98, 0.86, 0.81, 0.86, 0.93,   # Jul–Dec
])
_DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

LOAD_PROFILES      = ["Flat", "Residential", "Commercial", "Industrial", "Upload CSV"]
LOAD_PROFILE_KEYS  = {p: p.lower().replace(" ", "_") for p in LOAD_PROFILES}

T_FULL = 8_760


# ── Solar ─────────────────────────────────────────────────────────────────────

def generate_solar_cf(
    climate: str = "Temperate / Midwest",
    T: int = T_FULL,
    seed: int = 42,
) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    params = SOLAR_PARAMS.get(climate, SOLAR_PARAMS["Temperate / Midwest"])
    peak_cf     = params["peak_cf"]
    noise_alpha = params["noise_alpha"]
    noise_beta  = params["noise_beta"]

    hours       = np.arange(T_FULL)
    hour_of_day = hours % 24
    day_of_year = hours // 24

    day_frac  = np.zeros(T_FULL)
    daytime   = (hour_of_day >= 6) & (hour_of_day < 18)
    day_frac[daytime] = np.sin(np.pi * (hour_of_day[daytime] - 6) / 12)

    seasonal = 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
    noise    = rng.beta(noise_alpha, noise_beta, T_FULL)

    cf = np.clip(day_frac * seasonal * peak_cf * noise, 0.0, 1.0)
    return cf[:T]


# ── Wind ──────────────────────────────────────────────────────────────────────

def generate_wind_cf(
    climate: str = "Temperate / Midwest",
    T: int = T_FULL,
    seed: int = 99,
) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    params = WIND_PARAMS.get(climate, WIND_PARAMS["Temperate / Midwest"])
    mean_cf      = params["mean_cf"]
    sigma        = params["sigma"]
    ar1          = params["ar1"]
    seasonal_amp = params["seasonal_amp"]

    hours       = np.arange(T_FULL)
    day_of_year = hours // 24

    seasonal_mean = mean_cf + seasonal_amp * np.cos(2 * np.pi * day_of_year / 365)

    cf = np.zeros(T_FULL)
    cf[0] = seasonal_mean[0]
    innov = rng.normal(0, sigma, T_FULL)
    for t in range(1, T_FULL):
        cf[t] = ar1 * cf[t - 1] + (1 - ar1) * seasonal_mean[t] + innov[t]

    return np.clip(cf, 0.0, 1.0)[:T]


# ── Load profiles ─────────────────────────────────────────────────────────────

def generate_load_profile(
    profile_type: str = "flat",
    peak_mw: float = 100.0,
    T: int = 24,
    custom_data=None,
) -> np.ndarray:
    """Return a (T,) array of hourly load in MW.

    Parameters
    ----------
    profile_type : str
        One of "flat", "residential", "commercial", "industrial", "custom" /
        "upload_csv".  Case-insensitive, spaces replaced with underscores.
    peak_mw : float
        Scales the preset shape so its peak equals peak_mw.
        Ignored when profile_type is "custom" (uploaded values used as-is).
    T : int
        Number of hours to return.
    custom_data : list | np.ndarray | None
        Raw values from a user-uploaded CSV (MW).  Used only when profile_type
        is "custom" or "upload_csv".
    """
    key = profile_type.lower().replace(" ", "_")

    # ── Custom / uploaded ─────────────────────────────────────────────────────
    if key in ("custom", "upload_csv"):
        if custom_data is not None and len(custom_data) > 0:
            arr = np.asarray(custom_data, dtype=float)
            if len(arr) == T:
                return arr
            elif len(arr) > T:
                # Downsample: evenly spaced
                idx = np.round(np.linspace(0, len(arr) - 1, T)).astype(int)
                return arr[idx]
            else:
                # Upsample: tile + truncate
                repeats = (T // len(arr)) + 1
                return np.tile(arr, repeats)[:T]
        # Fallback: flat at peak_mw
        return np.full(T, peak_mw)

    # ── Preset archetypes ─────────────────────────────────────────────────────
    base = _LOAD_SHAPES.get(key, _LOAD_SHAPES["flat"]).copy()  # shape (24,)

    if T == 8_760:
        return _build_8760(base, peak_mw)

    # Tile daily pattern to fill T
    repeats = (T // 24) + 1
    arr     = np.tile(base, repeats)[:T].copy()

    # Add weekend reduction for residential and commercial
    if key in ("residential", "commercial") and T >= 48:
        # Assume simulation starts Monday (day 0)
        for h in range(T):
            day = h // 24
            if day % 7 >= 5:          # Saturday=5, Sunday=6
                arr[h] *= 0.78

    return arr * peak_mw


def _build_8760(base_24h: np.ndarray, peak_mw: float) -> np.ndarray:
    """Build a full 8760-hour profile with seasonal and weekend effects."""
    hours: list = []
    for m_idx, (n_days, s_mult) in enumerate(zip(_DAYS_PER_MONTH, _SEASONAL_MULT)):
        day_offset = sum(_DAYS_PER_MONTH[:m_idx])
        for d in range(n_days):
            abs_day   = day_offset + d
            is_weekend = abs_day % 7 >= 5
            day_shape  = base_24h * s_mult
            if is_weekend:
                day_shape = day_shape * 0.80
            hours.extend(day_shape)

    arr = np.array(hours[:T_FULL], dtype=float)
    if arr.max() > 0:
        arr = arr / arr.max() * peak_mw
    return arr


def load_profile_stats(profile: np.ndarray) -> dict:
    """Return basic statistics for display."""
    return {
        "peak_mw":  float(profile.max()),
        "avg_mw":   float(profile.mean()),
        "min_mw":   float(profile.min()),
        "load_factor": float(profile.mean() / profile.max()) if profile.max() > 0 else 0.0,
    }
