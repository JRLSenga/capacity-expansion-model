"""
backend/model.py  —  Capacity Expansion Model  (linopy LP)
==========================================================
Generator types: solar, wind, gas, nuclear, coal, battery.
Generators are passed as a flat list; each has a zone, type, and parameters.

inputs["generators"] list schema
---------------------------------
  Common fields (all types):
    id           str    unique generator ID
    zone         str    zone_id this generator belongs to
    type         str    "solar"|"wind"|"gas"|"nuclear"|"coal"|"battery"
    label        str    display name
    capex        float  $/MW capital cost  (applied only to NEW capacity)
    max_mw       float  maximum total capacity (MW)
    existing_mw  float  already-built capacity (default 0; no capex charged)
    expandable   bool   True → optimizer may build beyond existing_mw
    retirable    bool   True → optimizer may retire below existing_mw

  Solar / Wind only:
    cf        np.ndarray  capacity-factor time series (T,)

  Thermal (gas / nuclear / coal):
    vcost     np.ndarray  variable cost $/MWh (T,)
    ramp      float       max MW ramp per hour (use 1e9 to disable)
    min_load  float       min dispatch as fraction of capacity (0–1)

  Battery:
    duration  float  energy duration (h)
    eta_c     float  charging efficiency (0–1)
    eta_d     float  discharging efficiency (0–1)
    sigma     float  hourly self-discharge rate (fraction)
    vom_c     float  variable O&M while charging ($/MWh)
    vom_d     float  variable O&M while discharging ($/MWh)

inputs["transfer_pairs"]:  list of (z1, z2, cap_mw) — one entry per bidirectional link
"""

from __future__ import annotations
import io, sys, time
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import xarray as xr
import linopy


def build_and_solve(inputs: dict) -> dict:
    """Build LP, solve, return structured results."""
    _validate_inputs(inputs)

    zones    = inputs["zone_ids"]
    gens     = inputs["generators"]
    T        = inputs["T"]
    lifetime = float(inputs.get("lifetime", 25.0))
    scale    = 8_760 / T
    voll     = float(inputs.get("voll", 10_000.0))
    min_re   = float(inputs.get("min_re", 0.0))

    t0 = time.time()
    print(f"[model] Building: {len(zones)} zones, {len(gens)} generators, T={T}")

    m     = linopy.Model(chunk=None)
    t_idx = pd.RangeIndex(T, name="time")
    z_idx = pd.Index(zones, name="zone")

    # ── Classify generators by type ──────────────────────────────────────────
    solar_gens = [g for g in gens if g["type"] == "solar"]
    wind_gens  = [g for g in gens if g["type"] == "wind"]
    therm_gens = [g for g in gens if g["type"] in ("gas", "nuclear", "coal")]
    batt_gens  = [g for g in gens if g["type"] == "battery"]

    # ── Capacity variables + CapEx-new auxiliary variables ───────────────────
    # Pattern for each type:
    #   cap      = total installed capacity  (lower/upper set by existing/expandable/retirable)
    #   cap_new  = new capacity beyond existing (>= 0; used for CapEx only)
    #   Constraint: cap_new >= cap - existing_mw  (optimizer minimises, so tight)
    #   CapEx in objective: cap_new * capex / lifetime

    def _cap_bounds(g):
        existing   = float(g.get("existing_mw", 0))
        expandable = bool(g.get("expandable", True))
        retirable  = bool(g.get("retirable", False))
        lo = 0.0 if retirable else existing
        hi = max(float(g["max_mw"]), existing) if expandable else existing
        return existing, lo, hi

    sol_cap  = {}; sol_cnew = {}
    wnd_cap  = {}; wnd_cnew = {}
    thm_cap  = {}; thm_cnew = {}
    bat_cap  = {}; bat_cnew = {}

    for g in solar_gens:
        gid = g["id"]; ex, lo, hi = _cap_bounds(g)
        sol_cap[gid]  = m.add_variables(lower=lo, name=f"sc_{gid}")
        sol_cnew[gid] = m.add_variables(lower=0,  name=f"scn_{gid}")
        m.add_constraints(sol_cap[gid]  <= hi,              name=f"sc_max_{gid}")
        m.add_constraints(sol_cnew[gid] >= sol_cap[gid] - ex, name=f"scn_lb_{gid}")

    for g in wind_gens:
        gid = g["id"]; ex, lo, hi = _cap_bounds(g)
        wnd_cap[gid]  = m.add_variables(lower=lo, name=f"wc_{gid}")
        wnd_cnew[gid] = m.add_variables(lower=0,  name=f"wcn_{gid}")
        m.add_constraints(wnd_cap[gid]  <= hi,              name=f"wc_max_{gid}")
        m.add_constraints(wnd_cnew[gid] >= wnd_cap[gid] - ex, name=f"wcn_lb_{gid}")

    for g in therm_gens:
        gid = g["id"]; ex, lo, hi = _cap_bounds(g)
        thm_cap[gid]  = m.add_variables(lower=lo, name=f"tc_{gid}")
        thm_cnew[gid] = m.add_variables(lower=0,  name=f"tcn_{gid}")
        m.add_constraints(thm_cap[gid]  <= hi,              name=f"tc_max_{gid}")
        m.add_constraints(thm_cnew[gid] >= thm_cap[gid] - ex, name=f"tcn_lb_{gid}")

    for g in batt_gens:
        gid = g["id"]; ex, lo, hi = _cap_bounds(g)
        bat_cap[gid]  = m.add_variables(lower=lo, name=f"bc_{gid}")
        bat_cnew[gid] = m.add_variables(lower=0,  name=f"bcn_{gid}")
        m.add_constraints(bat_cap[gid]  <= hi,              name=f"bc_max_{gid}")
        m.add_constraints(bat_cnew[gid] >= bat_cap[gid] - ex, name=f"bcn_lb_{gid}")

    # ── Dispatch / storage variables ─────────────────────────────────────────
    sol_curt = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"scurt_{g['id']}") for g in solar_gens}
    wnd_curt = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"wcurt_{g['id']}") for g in wind_gens}
    thm_disp = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"tdisp_{g['id']}") for g in therm_gens}
    bat_chg  = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"bchg_{g['id']}")  for g in batt_gens}
    bat_dch  = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"bdch_{g['id']}")  for g in batt_gens}
    bat_soc  = {g["id"]: m.add_variables(lower=0, coords=[t_idx], name=f"bsoc_{g['id']}")  for g in batt_gens}

    unserved = m.add_variables(lower=0, coords=[z_idx, t_idx], name="unserved")

    # Transfer variables — one bidirectional variable per link
    transfer_pairs = inputs.get("transfer_pairs", [])
    transfer: dict = {}
    for entry in transfer_pairs:
        z1, z2, cap = entry
        transfer[(z1, z2)] = m.add_variables(
            lower=-float(cap), upper=float(cap), coords=[t_idx],
            name=f"xfer_{z1}_{z2}")

    dc_load = {z: np.asarray(inputs["dc_load"][z], dtype=float) for z in zones}

    # ── Constraints ───────────────────────────────────────────────────────────

    # Solar: curtailment bound (vectorized over time)
    for g in solar_gens:
        gid   = g["id"]
        cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
        m.add_constraints(sol_curt[gid] <= sol_cap[gid] * cf_da, name=f"sc_curt_{gid}")

    # Wind: curtailment bound (vectorized over time)
    for g in wind_gens:
        gid   = g["id"]
        cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
        m.add_constraints(wnd_curt[gid] <= wnd_cap[gid] * cf_da, name=f"wc_curt_{gid}")

    # Thermal: dispatch cap, min-load (vectorized), ramp (sequential loop)
    for g in therm_gens:
        gid  = g["id"]
        ramp = float(g.get("ramp", 1e9))
        ml   = float(g.get("min_load", 0.0))
        m.add_constraints(thm_disp[gid] <= thm_cap[gid], name=f"td_cap_{gid}")
        if ml > 0:
            m.add_constraints(thm_disp[gid] >= ml * thm_cap[gid], name=f"td_ml_{gid}")
        if ramp < 1e8:
            for t in range(1, T):
                dt  = thm_disp[gid].isel(time=t)
                dtm = thm_disp[gid].isel(time=t - 1)
                m.add_constraints(dt  - dtm <= ramp, name=f"td_ru_{gid}_{t}")
                m.add_constraints(dtm - dt  <= ramp, name=f"td_rd_{gid}_{t}")
            d0  = thm_disp[gid].isel(time=0)
            dT1 = thm_disp[gid].isel(time=T - 1)
            m.add_constraints(d0  - dT1 <= ramp, name=f"td_cyc_u_{gid}")
            m.add_constraints(dT1 - d0  <= ramp, name=f"td_cyc_d_{gid}")

    # Battery: SOC dynamics (sequential loop)
    for g in batt_gens:
        gid = g["id"]
        dur = float(g.get("duration",  4.0))
        ec  = float(g.get("eta_c",     0.96))
        ed  = float(g.get("eta_d",     0.96))
        sig = float(g.get("sigma",     0.0002))
        m.add_constraints(bat_soc[gid] <= bat_cap[gid] * dur, name=f"bs_cap_{gid}")
        m.add_constraints(bat_chg[gid] <= bat_cap[gid],       name=f"bchg_lim_{gid}")
        m.add_constraints(bat_dch[gid] <= bat_cap[gid],       name=f"bdch_lim_{gid}")
        m.add_constraints(bat_dch[gid] / ed <= bat_soc[gid],  name=f"bdch_soc_{gid}")
        for t in range(1, T):
            m.add_constraints(
                bat_soc[gid].isel(time=t) ==
                bat_soc[gid].isel(time=t - 1) * (1 - sig)
                + bat_chg[gid].isel(time=t) * ec
                - bat_dch[gid].isel(time=t) / ed,
                name=f"bs_dyn_{gid}_{t}")
        m.add_constraints(
            bat_soc[gid].isel(time=0) ==
            bat_soc[gid].isel(time=T - 1) * (1 - sig)
            + bat_chg[gid].isel(time=0) * ec
            - bat_dch[gid].isel(time=0) / ed,
            name=f"bs_cyc_{gid}")

    # Power balance (per zone, vectorized over time)
    for z in zones:
        zs = [g for g in solar_gens if g["zone"] == z]
        zw = [g for g in wind_gens  if g["zone"] == z]
        zt = [g for g in therm_gens if g["zone"] == z]
        zb = [g for g in batt_gens  if g["zone"] == z]
        supply = None

        for g in zs:
            cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
            term  = sol_cap[g["id"]] * cf_da - sol_curt[g["id"]]
            supply = term if supply is None else supply + term

        for g in zw:
            cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
            term  = wnd_cap[g["id"]] * cf_da - wnd_curt[g["id"]]
            supply = term if supply is None else supply + term

        for g in zt:
            supply = thm_disp[g["id"]] if supply is None else supply + thm_disp[g["id"]]

        for g in zb:
            net    = bat_dch[g["id"]] - bat_chg[g["id"]]
            supply = net if supply is None else supply + net

        for (z1, z2), xf in transfer.items():
            if z2 == z:
                supply = xf if supply is None else supply + xf
            elif z1 == z:
                supply = -xf if supply is None else supply - xf

        load_da = xr.DataArray(dc_load[z], coords={"time": t_idx})
        unsrv_z = unserved.sel(zone=z)
        if supply is None:
            m.add_constraints(unsrv_z == load_da, name=f"bal_{z}")
        else:
            m.add_constraints(supply + unsrv_z == load_da, name=f"bal_{z}")

    # RE coverage (scalar constraint)
    if min_re > 0 and (solar_gens or wind_gens):
        total_load_mwh = sum(float(np.sum(dc_load[z])) for z in zones) * scale
        re_expr = None
        for g in solar_gens:
            term = (sol_cap[g["id"]] * float(np.sum(g["cf"]))
                    - sol_curt[g["id"]].sum("time"))
            re_expr = term if re_expr is None else re_expr + term
        for g in wind_gens:
            term = (wnd_cap[g["id"]] * float(np.sum(g["cf"]))
                    - wnd_curt[g["id"]].sum("time"))
            re_expr = term if re_expr is None else re_expr + term
        if re_expr is not None:
            m.add_constraints(re_expr >= min_re * total_load_mwh, name="re_cov")

    # ── Objective ─────────────────────────────────────────────────────────────
    # Minimize: Investment cost + Fixed O&M + Variable O&M + VOLL
    #
    #  (1) Investment cost  = total_cap_MW × (CapEx × CRF)   [$/yr per generator]
    #  (2) Fixed O&M        = total_cap_MW × fom              [$/yr per generator]
    #  (3) Variable O&M     = dispatch_MWh × vom (fuel+VOM for thermal) [$/yr]
    #  (4) VOLL             = unserved_MWh × voll             [$/yr]
    #
    # CRF (Capital Recovery Factor) = r(1+r)^n / ((1+r)^n − 1)
    # where r = WACC, n = lifetime.  Reduces to 1/n when r → 0.
    # Investment cost applies to TOTAL installed MW so that retirements
    # (retirable=True generators) correctly reduce annual cost.
    # ─────────────────────────────────────────────────────────────────────────

    obj = linopy.LinearExpression(None, m)

    def _crf(g):
        r = max(float(g.get("wacc", 0.08)), 0.0)
        n = max(float(g.get("lifetime", lifetime)), 1.0)
        if r < 1e-9:
            return 1.0 / n
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def _inv(g):
        """Annualized investment cost ($/MW-yr) = CapEx × CRF."""
        return g["capex"] * _crf(g)

    # ── (1) Investment cost on NEW capacity only (sunk costs excluded) ────────
    for g in solar_gens: obj = obj + sol_cnew[g["id"]] * _inv(g)
    for g in wind_gens:  obj = obj + wnd_cnew[g["id"]] * _inv(g)
    for g in therm_gens: obj = obj + thm_cnew[g["id"]] * _inv(g)
    for g in batt_gens:  obj = obj + bat_cnew[g["id"]] * _inv(g)

    # ── (2) Fixed O&M on total installed capacity ─────────────────────────────
    for g in solar_gens:
        fom = float(g.get("fom", 0.0))
        if fom > 0: obj = obj + sol_cap[g["id"]] * fom
    for g in wind_gens:
        fom = float(g.get("fom", 0.0))
        if fom > 0: obj = obj + wnd_cap[g["id"]] * fom
    for g in therm_gens:
        fom = float(g.get("fom", 0.0))
        if fom > 0: obj = obj + thm_cap[g["id"]] * fom
    for g in batt_gens:
        fom = float(g.get("fom", 0.0))
        if fom > 0: obj = obj + bat_cap[g["id"]] * fom

    # ── (3) Variable O&M on dispatch ─────────────────────────────────────────
    # Solar & wind VOM ($/MWh × net generation, scaled)
    for g in solar_gens:
        vom = float(g.get("vom", 0.0))
        if vom > 0:
            cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
            net   = sol_cap[g["id"]] * cf_da - sol_curt[g["id"]]
            obj   = obj + net.sum("time") * (vom * scale)
    for g in wind_gens:
        vom = float(g.get("vom", 0.0))
        if vom > 0:
            cf_da = xr.DataArray(np.array(g["cf"]), coords={"time": t_idx})
            net   = wnd_cap[g["id"]] * cf_da - wnd_curt[g["id"]]
            obj   = obj + net.sum("time") * (vom * scale)
    # Thermal: variable cost includes fuel (fuel_cost × heat_rate) + VOM
    for g in therm_gens:
        vc    = np.array(g["vcost"]) * scale
        vc_da = xr.DataArray(vc, coords={"time": t_idx})
        obj   = obj + (thm_disp[g["id"]] * vc_da).sum("time")
    # Battery VOM on charge/discharge
    for g in batt_gens:
        obj = (obj
               + bat_chg[g["id"]].sum("time") * (float(g.get("vom_c", 0.5)) * scale)
               + bat_dch[g["id"]].sum("time") * (float(g.get("vom_d", 0.5)) * scale))

    # ── (4) Value of Lost Load (unserved demand penalty) ─────────────────────
    obj = obj + unserved.sum() * (voll * scale)

    m.objective = obj

    print(f"[model] Built in {time.time()-t0:.1f}s — solving…")
    t1 = time.time()

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver     = inputs.get("solver", "highs").lower()
    time_limit = float(inputs.get("time_limit", 300))
    solver_log = ""
    if solver == "gurobi":
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                status, cond = m.solve(solver_name="gurobi", TimeLimit=time_limit, OutputFlag=1)
            solver_log = buf.getvalue()
        except Exception:
            print("[model] Gurobi unavailable — using HiGHS")
            buf = io.StringIO()
            with redirect_stdout(buf):
                status, cond = m.solve(solver_name="highs", time_limit=time_limit, output_flag=True)
            solver_log = buf.getvalue()
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            status, cond = m.solve(solver_name="highs", time_limit=time_limit, output_flag=True)
        solver_log = buf.getvalue()

    solve_time = time.time() - t1
    print(f"[model] Done in {solve_time:.1f}s — status: {status} / {cond}")

    if status not in ("ok", "optimal") and "optimal" not in str(cond).lower():
        return {"status": status, "termination": str(cond),
                "error": "Solver did not find an optimal solution."}

    # ── Extract results ───────────────────────────────────────────────────────
    def val(v): return float(v.solution.values)
    def arr(v): return np.array(v.solution.values).flatten()

    gen_cap  = {}; gen_disp = {}
    gen_charge = {}; gen_dch = {}; gen_soc = {}

    for g in solar_gens:
        gid = g["id"]; cap = val(sol_cap[gid])
        gen_cap[gid]  = cap
        gen_disp[gid] = cap * np.array(g["cf"]) - arr(sol_curt[gid])

    for g in wind_gens:
        gid = g["id"]; cap = val(wnd_cap[gid])
        gen_cap[gid]  = cap
        gen_disp[gid] = cap * np.array(g["cf"]) - arr(wnd_curt[gid])

    for g in therm_gens:
        gid = g["id"]
        gen_cap[gid]  = val(thm_cap[gid])
        gen_disp[gid] = arr(thm_disp[gid])

    for g in batt_gens:
        gid = g["id"]
        gen_cap[gid]    = val(bat_cap[gid])
        gen_charge[gid] = arr(bat_chg[gid])
        gen_dch[gid]    = arr(bat_dch[gid])
        gen_soc[gid]    = arr(bat_soc[gid])
        gen_disp[gid]   = arr(bat_dch[gid]) - arr(bat_chg[gid])

    results = {
        "status":     "optimal",
        "objective":  float(m.objective.value),
        "solver_log": solver_log,
        "solve_time": solve_time,
        "zones":      zones,
        "T":          T,
        "scale":      scale,
        "dc_load":    dc_load,
        "generators": gens,
        "gen_cap":    gen_cap,
        "gen_disp":   gen_disp,
        "gen_charge": gen_charge,
        "gen_dch":    gen_dch,
        "gen_soc":    gen_soc,
        "unserved":   {z: arr(unserved.sel(zone=z)) for z in zones},
        "transfer":   {f"{z1}->{z2}": arr(xf) for (z1, z2), xf in transfer.items()},
    }
    results["costs"] = _compute_costs(
        results, solar_gens, wind_gens, therm_gens, batt_gens,
        lifetime, scale, voll)
    return results


def _compute_costs(res, solar_gens, wind_gens, therm_gens, batt_gens,
                   lifetime, scale, voll):
    gc = res["gen_cap"]

    def _crf(g):
        r = max(float(g.get("wacc", 0.08)), 0.0)
        n = max(float(g.get("lifetime", lifetime)), 1.0)
        if r < 1e-9:
            return 1.0 / n
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    def _new_cap(g):
        return max(0.0, gc[g["id"]] - float(g.get("existing_mw", 0)))

    def _inv(g):
        """Annualized investment cost on NEW capacity (new_cap × CapEx × CRF)."""
        return _new_cap(g) * g["capex"] * _crf(g)

    # Investment cost on new capacity only (mirrors the objective)
    capex_solar = sum(_inv(g) for g in solar_gens)
    capex_wind  = sum(_inv(g) for g in wind_gens)
    capex_therm = sum(_inv(g) for g in therm_gens)
    capex_batt  = sum(_inv(g) for g in batt_gens)

    fom_solar = sum(gc[g["id"]] * float(g.get("fom", 0)) for g in solar_gens)
    fom_wind  = sum(gc[g["id"]] * float(g.get("fom", 0)) for g in wind_gens)
    fom_therm = sum(gc[g["id"]] * float(g.get("fom", 0)) for g in therm_gens)
    fom_batt  = sum(gc[g["id"]] * float(g.get("fom", 0)) for g in batt_gens)
    fom_total = fom_solar + fom_wind + fom_therm + fom_batt

    opex_therm = sum(
        float(np.sum(np.clip(res["gen_disp"][g["id"]], 0, None) * np.array(g["vcost"]))) * scale
        for g in therm_gens)
    opex_voll  = sum(
        float(np.sum(res["unserved"][z])) * voll * scale
        for z in res["zones"])

    total = (capex_solar + capex_wind + capex_therm + capex_batt
             + fom_total + opex_therm + opex_voll)
    return dict(
        capex_solar=capex_solar, capex_wind=capex_wind,
        capex_therm=capex_therm, capex_batt=capex_batt,
        fom_solar=fom_solar, fom_wind=fom_wind,
        fom_therm=fom_therm, fom_batt=fom_batt, fom_total=fom_total,
        opex_therm=opex_therm, opex_voll=opex_voll,
        total=total)


def _validate_inputs(inputs: dict):
    for k in ("zone_ids", "T", "dc_load", "generators", "lifetime", "voll"):
        if k not in inputs:
            raise ValueError(f"Missing required input key: '{k}'")
    if not (1 <= inputs["T"] <= 8_760):
        raise ValueError(f"T must be 1–8760, got {inputs['T']}")
