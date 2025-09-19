
import base64
from io import BytesIO
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

from statsbombpy import sb

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc


COMPETITION_ID = 53
SEASON_ID = 315
TEAM = "England"

OPEN = {"creds": {}}
PITCH_X, PITCH_Y = 120.0, 80.0

NODE_SIZE = 170
ARROW_LW = 1
DROP_SELF_LOOPS = True
ARROW_CURVE_RAD = 0.22


COUNT_CEILING = 10.0
COLORBAR_TICKS = [2, 4, 6, 8, 10]
SHOW_10PLUS_ARROW = False
COLOR_MAP_BASE = "YlOrRd"
COLOR_LOW_CUTOFF = 0.1
COLOR_HIGH_CUTOFF = 1.00
COLOR_NORM = "power"
COLOR_GAMMA = 0.7

VERBOSE = False


ALIAS_BY_MATCH_ID: Dict[int, Dict[int, int]] = {}
ALIAS_BY_MATCH_NAME: Dict[int, Dict[str, str]] = {

    4018354: {
        "Lucia Di Guglielmo": "Lisa Boattin",
        "Elisabetta Oliviero": "Lucia Di Guglielmo",
    },
    3998843:{"Jessica Carter":"Alex Greenwood",
             "Alex Greenwood": "Jessica Carter",
             "Bethany Mead":"Ella Toone"},
    4018355:{"Jessica Carter":"Leah Williamson",
             "Leah Williamson":"Jessica Carter"},
    4020846:{"Lauren James": "Lauren Hemp",
             "Lauren Hemp":"Lauren James"}
}
SWAP_BY_MATCH_ID: Dict[int, List[Tuple[int, int]]] = {}
SWAP_BY_MATCH_NAME: Dict[int, List[Tuple[str, str]]] = {}


def dprint(*a, **k):
    if VERBOSE:
        print(*a, **k)

def _extract_team_name(cell, side_hint=None) -> str:
    if isinstance(cell, dict):
        for k in (f"{side_hint}_team_name" if side_hint else None, "team_name", "name", "home_team_name", "away_team_name"):
            if k and k in cell:
                return str(cell[k])
        if "name" in cell:
            return str(cell["name"])
    return str(cell)

def _as_xy(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return np.nan, np.nan

def _flatten_event_locations(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    if "location" in ev.columns:
        xy = ev["location"].apply(_as_xy)
        ev["x"] = xy.apply(lambda t: t[0]); ev["y"] = xy.apply(lambda t: t[1])
    else:
        ev["x"] = np.nan; ev["y"] = np.nan
    if "pass_end_location" in ev.columns:
        pxy = ev["pass_end_location"].apply(_as_xy)
        ev["end_x"] = pxy.apply(lambda t: t[0]); ev["end_y"] = pxy.apply(lambda t: t[1])
    else:
        ev["end_x"] = np.nan; ev["end_y"] = np.nan
    if "carry_end_location" in ev.columns:
        cxy = ev["carry_end_location"].apply(_as_xy)
        ev["carry_end_x"] = cxy.apply(lambda t: t[0]); ev["carry_end_y"] = cxy.apply(lambda t: t[1])
    else:
        ev["carry_end_x"] = np.nan; ev["carry_end_y"] = np.nan
    return ev

def canonical_team_name_in_events(ev: pd.DataFrame, team_hint: str) -> str:
    teams = pd.Series(sorted(ev["team"].dropna().unique()), dtype=str)
    hits = teams[teams.str.contains(team_hint, case=False, na=False)]
    if not hits.empty:
        return hits.iloc[hits.str.len().argmax()]
    return team_hint

def _rotate_180(x: float, y: float) -> Tuple[float, float]:
    xr = (PITCH_X - x) if pd.notna(x) else x
    yr = (PITCH_Y - y) if pd.notna(y) else y
    return xr, yr

def infer_attack_direction_by_period(ev: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    out = {}
    periods = sorted([int(p) for p in pd.to_numeric(ev["period"], errors="coerce").dropna().unique()])
    for team in ev["team"].dropna().unique():
        for period in periods:
            sub = ev[(ev["team"] == team) & (pd.to_numeric(ev["period"], errors="coerce") == period)]
            shots = sub[sub["type"] == "Shot"]
            if len(shots) >= 1:
                dir_lr = float(shots["x"].mean()) > (PITCH_X / 2)
            else:
                pc = sub[sub["type"].isin(["Pass", "Carry"])]
                if pc.empty:
                    dir_lr = True
                else:
                    dx = []
                    passes = pc[pc["type"] == "Pass"]; carries = pc[pc["type"] == "Carry"]
                    if not passes.empty:  dx += list((passes["end_x"] - passes["x"]).dropna())
                    if not carries.empty: dx += list((carries["carry_end_x"] - carries["x"]).dropna())
                    dir_lr = (np.median(dx) if len(dx) else 0.0) >= 0.0
            out[(team, period)] = "LR" if dir_lr else "RL"
    return out

def _needs_flip(directions, team: str, period) -> bool:
    p = pd.to_numeric(period, errors="coerce")
    if pd.isna(p): return False
    return directions.get((team, int(p)), "LR") == "RL"

def perspective_transform(ev: pd.DataFrame, ref_team: str) -> pd.DataFrame:
    ev = _flatten_event_locations(ev)
    dirs = infer_attack_direction_by_period(ev)
    ev = ev.copy()
    x_ref, y_ref, xe_ref, ye_ref = [], [], [], []
    for _, r in ev.iterrows():
        flip = _needs_flip(dirs, ref_team, r.get("period", np.nan))
        x0, y0 = r.get("x", np.nan), r.get("y", np.nan)
        xr, yr = _rotate_180(x0, y0) if flip else (x0, y0)
        ex, ey = (r.get("end_x", np.nan), r.get("end_y", np.nan)) if r.get("type") == "Pass" else (np.nan, np.nan)
        if flip and (pd.notna(ex) or pd.notna(ey)):
            ex, ey = _rotate_180(ex, ey)
        x_ref.append(xr); y_ref.append(yr); xe_ref.append(ex); ye_ref.append(ey)
    ev["x_ref"] = x_ref; ev["y_ref"] = y_ref; ev["xe_ref"] = xe_ref; ev["ye_ref"] = ye_ref
    return ev

def load_matches() -> pd.DataFrame:
    m = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID, fmt="dataframe", **OPEN)
    m["home_team_name"] = m["home_team"].apply(lambda c: _extract_team_name(c, "home"))
    m["away_team_name"] = m["away_team"].apply(lambda c: _extract_team_name(c, "away"))
    if "competition_stage" in m.columns and m["competition_stage"].map(type).eq(dict).any():
        m["stage"] = m["competition_stage"].apply(lambda d: d.get("name") if isinstance(d, dict) else d)
    elif "competition_stage_name" in m.columns:
        m["stage"] = m["competition_stage_name"]
    else:
        m["stage"] = m.get("stage", "")
    m["sort_date"] = pd.to_datetime(m.get("match_date", m.get("kick_off")), errors="coerce")
    return m

def select_team_matches(matches: pd.DataFrame, team: str) -> List[int]:
    def name_match(s: pd.Series, needle: str) -> pd.Series:
        return s.str.contains(rf"{re.escape(needle)}", case=False, na=False)
    mm = matches[name_match(matches["home_team_name"], team) | name_match(matches["away_team_name"], team)].copy()
    mm = mm.sort_values("sort_date", kind="stable")
    return mm["match_id"].astype(int).tolist()

@lru_cache(maxsize=256)
def cached_events(match_id: int) -> pd.DataFrame:
    ev = sb.events(match_id=int(match_id), fmt="dataframe", **OPEN)
    for col in ["team", "possession_team", "type", "pass_type"]:
        if col in ev.columns:
            ev[col] = ev[col].apply(lambda v: v.get("name") if isinstance(v, dict) else (str(v) if pd.notna(v) else ""))
    for col in ["period", "minute", "second", "timestamp", "possession"]:
        if col not in ev.columns: ev[col] = np.nan
    return _flatten_event_locations(ev)


def _clean_label(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)): return ""
    s = str(s).strip()
    return "" if s.lower() in {"none", "nan"} else s

def shorten_name_to_two(name: str, known_map: Dict[str,str]) -> str:
    s = _clean_label(name)
    return _clean_label(known_map.get(s, s))

def get_lineup_known_as_maps(match_id: int, team_hint: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    lineups = sb.lineups(match_id, fmt="dataframe", **OPEN)
    team_key = next((k for k in lineups.keys() if team_hint.lower() in k.lower()), None) or list(lineups.keys())[0]
    lu = lineups[team_key][["player_id", "player_name", "player_nickname"]].copy()
    lu["player_name"] = lu["player_name"].fillna("").astype(str).str.strip()
    lu["player_nickname"] = lu["player_nickname"].fillna("").astype(str).str.strip()
    lu["known_as"] = lu["player_nickname"].where(lu["player_nickname"] != "", lu["player_name"])
    id_to_known = dict(zip(lu["player_id"], lu["known_as"]))
    known_to_id = {v: k for k, v in id_to_known.items() if v}
    return id_to_known, known_to_id

def get_team_starting_slots(ev: pd.DataFrame, team: str, known_map: Dict[str,str], use_positions: bool = False):
    xi = ev[(ev["type"] == "Starting XI") & (ev["team"] == team)]
    id_to_lab, order = {}, []
    if xi.empty:
        return id_to_lab, order
    row = xi.iloc[0]
    lineup = (row.get("tactics", {}) or {}).get("lineup", []) or []

    def _short_pos(name: str) -> str:
        if not isinstance(name, str): return ""
        name = name.lower()
        table = {
            "goalkeeper":"GK","right back":"RB","right wing back":"RWB",
            "right center back":"RCB","right centre-back":"RCB",
            "center back":"CB","centre-back":"CB",
            "left center back":"LCB","left centre-back":"LCB",
            "left back":"LB","left wing back":"LWB","defensive midfield":"DM",
            "central midfield":"CM","attacking midfield":"AM","right midfield":"RM","left midfield":"LM",
            "right wing":"RW","left wing":"LW","second striker":"SS","striker":"ST","centre forward":"ST","center forward":"ST",
        }
        for k, v in table.items():
            if k in name: return v
        parts = [w[0].upper() for w in name.split() if w]
        return "".join(parts)[:3] if parts else name

    tmp = []
    for it in lineup:
        pid = it.get("player", {}).get("id")
        pname = it.get("player", {}).get("name", "")
        posname = it.get("position", {}).get("name", "")
        if pid is None: continue
        base = _clean_label(shorten_name_to_two(pname, known_map))
        parts = [p for p in str(pname).split() if p.strip()]
        surname = parts[-1] if parts else ""
        s_init = (surname[:1].upper() + ".") if surname else ""
        lab = _short_pos(posname) if use_positions else base
        tmp.append((pid, base, surname, s_init, lab))

    from collections import Counter
    base_counts = Counter([t[1] for t in tmp if t[1] != ""])
    used_labels = set()
    for pid, base, surname, s_init, lab in tmp:
        final = lab
        if not use_positions:
            if base and base_counts[base] > 1:
                candidate = f"{base} {s_init}".strip()
                if candidate in used_labels:
                    candidate = f"{base} {surname}".strip()
                final = _clean_label(candidate) or base
        final = _clean_label(final)
        id_to_lab[pid] = final
        used_labels.add(final); order.append(final)
    seen, keep = set(), []
    for l in order:
        if l not in seen:
            seen.add(l); keep.append(l)
    return id_to_lab, keep

def build_maps(ev: pd.DataFrame, team: str, known_map: Dict[str,str], use_positions: bool = False):
    id_to_slot, order = get_team_starting_slots(ev, team, known_map, use_positions)
    name_to_slot = {}
    xi = ev[(ev["type"] == "Starting XI") & (ev["team"] == team)]
    if not xi.empty:
        row = xi.iloc[0]
        lineup = (row.get("tactics", {}) or {}).get("lineup", []) or []
        for it in lineup:
            pid = it.get("player", {}).get("id")
            pname = it.get("player", {}).get("name", "")
            base = _clean_label(shorten_name_to_two(pname, known_map))
            final = _clean_label(id_to_slot.get(pid, ""))
            if pid in id_to_slot and final:
                if base:
                    name_to_slot[base] = final
                name_to_slot[final] = final

    subs = ev[(ev["type"] == "Substitution") & (ev["team"] == team)].copy()
    if not subs.empty:
        subs = subs.sort_values(["period","minute","second"], kind="stable")
        for _, r in subs.iterrows():
            off = r.get("player", {})
            on  = r.get("substitution_replacement", {})
            off_id = off.get("id") if isinstance(off, dict) else None
            on_id  = on.get("id") if isinstance(on, dict) else None
            off_nm = _clean_label(shorten_name_to_two(off.get("name",""), known_map)) if isinstance(off, dict) else ""
            on_nm  = _clean_label(shorten_name_to_two(on.get("name",""), known_map))  if isinstance(on, dict)  else ""
            slot = id_to_slot.get(off_id, name_to_slot.get(off_nm))
            if slot:
                if on_id is not None: id_to_slot[on_id] = slot
                if on_nm: name_to_slot[on_nm] = slot
    return id_to_slot, name_to_slot, order


def extract_ids_and_names(df: pd.DataFrame):
    pid  = df.get("player_id")
    prid = df.get("pass_recipient_id")
    def _id_from_obj_or_none(v): return v.get("id") if isinstance(v, dict) else None
    if (pid is None or pid.isna().all()) and "player" in df.columns:
        pid  = df["player"].apply(_id_from_obj_or_none)
    if (prid is None or prid.isna().all()) and "pass_recipient" in df.columns:
        prid = df["pass_recipient"].apply(_id_from_obj_or_none)
    pid  = pd.to_numeric(pid, errors="coerce")
    prid = pd.to_numeric(prid, errors="coerce")

    def _name(v):
        if isinstance(v, dict): return v.get("name","")
        return str(v) if pd.notna(v) else ""
    pname = df["player"].apply(_name) if "player" in df.columns else df.get("player_name", pd.Series([], dtype=str)).fillna("")
    rname = df["pass_recipient"].apply(_name) if "pass_recipient" in df.columns else df.get("pass_recipient_name", pd.Series([], dtype=str)).fillna("")
    return pid, prid, pname, rname

def filter_completed_passes(p: pd.DataFrame, min_len_m: float) -> pd.DataFrame:
    if p.empty: return p
    completed = p["pass_outcome"].isna()
    needed = ["x_ref","y_ref","xe_ref","ye_ref"]
    for c in needed:
        if c not in p.columns:
            p[c] = np.nan
    dx = (p["xe_ref"] - p["x_ref"]).astype(float)
    dy = (p["ye_ref"] - p["y_ref"]).astype(float)
    L  = np.sqrt(dx**2 + dy**2)
    return p[completed & L.ge(float(min_len_m)) & p[needed].notna().all(axis=1)].copy()

def filter_progressive_passes(p: pd.DataFrame, min_len_m: float, min_forward_m: float) -> pd.DataFrame:
    if p.empty: return p
    completed = p["pass_outcome"].isna()
    needed = ["x_ref","y_ref","xe_ref","ye_ref"]
    for c in needed:
        if c not in p.columns:
            p[c] = np.nan
    dx = (p["xe_ref"] - p["x_ref"]).astype(float)
    dy = (p["ye_ref"] - p["y_ref"]).astype(float)
    L  = np.sqrt(dx**2 + dy**2)
    return p[completed & L.ge(float(min_len_m)) & dx.ge(float(min_forward_m)) & p[needed].notna().all(axis=1)].copy()

def select_team_passes(ev: pd.DataFrame, team_hint: str, progressive: bool, min_len_m: float, min_forward_m: float) -> pd.DataFrame:
    team = canonical_team_name_in_events(ev, team_hint)
    e = perspective_transform(ev, ref_team=team)
    p_all = e[(e["team"] == team) & (e["type"] == "Pass")].copy()
    return filter_progressive_passes(p_all, min_len_m, min_forward_m) if progressive else filter_completed_passes(p_all, min_len_m)


def build_effective_alias_map(match_id: int, team_hint: str) -> Dict[int, int]:
    id_to_known, known_to_id = get_lineup_known_as_maps(match_id, team_hint)
    amap: Dict[int, int] = {}
    for src, dst in ALIAS_BY_MATCH_ID.get(match_id, {}).items():
        if src is not None and dst is not None:
            amap[int(src)] = int(dst)
    for src_name, dst_name in ALIAS_BY_MATCH_NAME.get(match_id, {}).items():
        s = known_to_id.get(_clean_label(src_name)); d = known_to_id.get(_clean_label(dst_name))
        if s is not None and d is not None:
            amap[int(s)] = int(d)
    for a, b in SWAP_BY_MATCH_ID.get(match_id, []):
        amap[int(a)] = int(b); amap[int(b)] = int(a)
    for a_name, b_name in SWAP_BY_MATCH_NAME.get(match_id, []):
        a = known_to_id.get(_clean_label(a_name)); b = known_to_id.get(_clean_label(b_name))
        if a is not None and b is not None:
            amap[int(a)] = int(b); amap[int(b)] = int(a)
    return amap

def _merge_label_dicts(master: Dict[int, str], add: Dict[int, str]) -> None:
    def splitnames(s: str) -> List[str]:
        return [t.strip() for t in str(s).split("/") if t and t.strip()]
    for pid, s in add.items():
        current = splitnames(master.get(pid, ""))
        incoming = splitnames(s)
        combined = []
        for nm in current + incoming:
            if nm and nm not in combined:
                combined.append(nm)
        if combined:
            master[pid] = " / ".join(sorted(combined, key=str.lower))

def apply_alias_to_ids(
    p: pd.DataFrame,
    xi_ids: Iterable[int],
    id_to_name: Dict[int, str],
    alias_map: Dict[int, int]
) -> Tuple[pd.DataFrame, List[int], Dict[int, str]]:
    if not alias_map:
        return p, list(xi_ids), id_to_name
    p = p.copy()
    p["src_id"] = p["src_id"].map(lambda i: int(alias_map.get(int(i), int(i))))
    p["dst_id"] = p["dst_id"].map(lambda i: int(alias_map.get(int(i), int(i))))
    xi_ids2 = [int(alias_map.get(int(pid), int(pid))) for pid in xi_ids]
    canon_to_names: Dict[int, List[str]] = {}
    for pid, name in id_to_name.items():
        cid = int(alias_map.get(int(pid), int(pid)))
        nm = _clean_label(name)
        if not nm: continue
        canon_to_names.setdefault(cid, [])
        if nm not in canon_to_names[cid]:
            canon_to_names[cid].append(nm)
    id_to_name2 = {cid: " / ".join(sorted(names, key=str.lower)) for cid, names in canon_to_names.items()}
    return p, xi_ids2, id_to_name2


def list_england_matches(team: str = "England") -> pd.DataFrame:
    matches = load_matches()
    mids = select_team_matches(matches, team)
    df = matches[matches["match_id"].isin(mids)].copy()
    def opp(row):
        home, away = str(row["home_team_name"]), str(row["away_team_name"])
        return away if team.lower() in home.lower() else home
    df["opponent"] = df.apply(opp, axis=1)
    df["label"] = df["sort_date"].dt.strftime("%Y-%m-%d") + " vs " + df["opponent"].astype(str)
    return df[["match_id", "label", "opponent"]].sort_values("label")

def build_nodes_edges_for_match(match_id: int, team: str, progressive: bool, min_len_m: float, min_forward_m: float):
    # Load events and build XI label map (Hudl/known-as)
    ev = cached_events(int(match_id))
    id_to_known, _ = get_lineup_known_as_maps(int(match_id), team)
    team_name = canonical_team_name_in_events(ev, team)
    id_to_slot, _, _ = build_maps(ev, team_name, id_to_known, use_positions=False)  # XI only
    xi_ids = list(id_to_slot.keys())


    p = select_team_passes(ev, team, progressive, min_len_m, min_forward_m)
    if p.empty:
        return None, None

    pid, prid, _, _ = extract_ids_and_names(p)
    p = p.assign(
        src_id=pd.to_numeric(pid, errors="coerce").astype("Int64"),
        dst_id=pd.to_numeric(prid, errors="coerce").astype("Int64")
    ).dropna(subset=["src_id","dst_id"]).astype({"src_id":"int","dst_id":"int"})

    alias_map = build_effective_alias_map(int(match_id), team)
    id_to_label = {k: v for k, v in id_to_slot.items()}
    if alias_map:

        p, xi_ids, id_to_label = apply_alias_to_ids(p, xi_ids, id_to_label, alias_map)


    xi_set = set(xi_ids)
    p = p[p["src_id"].isin(xi_set) & p["dst_id"].isin(xi_set)]
    if p.empty:
        return None, None


    edges = p.groupby(["src_id","dst_id"]).size().reset_index(name="count")

    starts = p.groupby("src_id")[["x_ref","y_ref"]].mean()
    ends   = p.groupby("dst_id")[["xe_ref","ye_ref"]].mean().rename(columns={"xe_ref":"x_ref","ye_ref":"y_ref"})
    nodes_xy = pd.concat([starts, ends]).groupby(level=0).mean().reset_index()
    nodes_xy.columns = ["player_id","x","y"]


    nodes_xy = nodes_xy[nodes_xy["player_id"].isin(xi_set)].copy()
    nodes_xy["node"] = nodes_xy["player_id"].map(id_to_label)
    nodes = nodes_xy[["node","x","y"]]

    edges = edges.assign(
        src = edges["src_id"].map(id_to_label),
        dst = edges["dst_id"].map(id_to_label),
    )[["src","dst","count"]]

    return nodes, edges

def build_aggregate(match_ids: List[int], team: str, progressive: bool, min_len_m: float, min_forward_m: float, per_game_average: bool = True):
    per_match_passes: List[pd.DataFrame] = []
    per_match_xis: List[List[int]] = []
    id_to_name_master: Dict[int, str] = {}
    for mid in match_ids:
        ev = cached_events(int(mid))
        id_to_known, _ = get_lineup_known_as_maps(int(mid), team)
        id_to_slot, _, _ = build_maps(ev, canonical_team_name_in_events(ev, team), id_to_known, use_positions=False)
        p_all = select_team_passes(ev, team, progressive, min_len_m, min_forward_m)
        if p_all.empty:
            per_match_xis.append(list(id_to_slot.keys()))
            continue
        pid, prid, _, _ = extract_ids_and_names(p_all)
        tmp = pd.DataFrame({
            "src_id": pd.to_numeric(pid, errors="coerce"),
            "dst_id": pd.to_numeric(prid, errors="coerce"),
            "x_ref": p_all["x_ref"], "y_ref": p_all["y_ref"],
            "xe_ref": p_all["xe_ref"], "ye_ref": p_all["ye_ref"],
        }).dropna(subset=["src_id","dst_id"])
        tmp["src_id"] = tmp["src_id"].astype(int); tmp["dst_id"] = tmp["dst_id"].astype(int)
        alias_map = build_effective_alias_map(int(mid), team)
        id2name = {k: v for k, v in id_to_slot.items()}
        xi_ids = list(id_to_slot.keys())
        if alias_map:
            tmp, xi_ids, id2name = apply_alias_to_ids(tmp, xi_ids, id2name, alias_map)
        per_match_passes.append(tmp[["src_id","dst_id","x_ref","y_ref","xe_ref","ye_ref"]])
        per_match_xis.append(xi_ids)
        _merge_label_dicts(id_to_name_master, id2name)

    if not per_match_passes:
        return None, None
    from collections import Counter
    counts = Counter()
    for xi in per_match_xis:
        counts.update(xi)
    if not counts:
        return None, None
    chosen_ids = [pid for pid, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:11]]
    chosen_set = set(chosen_ids)

    all_p = []
    for p in per_match_passes:
        if p is None or p.empty: continue
        q = p[(p["src_id"].isin(chosen_set)) & (p["dst_id"].isin(chosen_set))].copy()
        if not q.empty:
            all_p.append(q[["src_id","dst_id","x_ref","y_ref","xe_ref","ye_ref"]])
    if not all_p:
        return None, None
    P = pd.concat(all_p, ignore_index=True)
    n_games = len(per_match_xis)

    edges = P.groupby(["src_id","dst_id"]).size().reset_index(name="count")
    if per_game_average and n_games > 0:
        edges["count"] = edges["count"] / float(n_games)

    starts = P.groupby("src_id")[["x_ref","y_ref"]].mean()
    ends   = P.groupby("dst_id")[["xe_ref","ye_ref"]].mean().rename(columns={"xe_ref":"x_ref","ye_ref":"y_ref"})
    nodes_xy = pd.concat([starts, ends]).groupby(level=0).mean().reset_index()
    nodes_xy.columns = ["player_id","x","y"]
    nodes_xy = nodes_xy[nodes_xy["player_id"].isin(chosen_set)].copy()
    nodes_xy["node"] = nodes_xy["player_id"].map(lambda pid: id_to_name_master.get(pid, str(pid)))
    nodes = nodes_xy[["node","x","y"]]
    id2name = {pid: id_to_name_master.get(pid, str(pid)) for pid in chosen_ids}
    edges = edges.assign(src = edges["src_id"].map(id2name),
                         dst = edges["dst_id"].map(id2name))[["src","dst","count"]]
    return nodes, edges


def get_yellow_to_red_cmap() -> LinearSegmentedColormap:
    base = cm.get_cmap(COLOR_MAP_BASE)
    colors = base(np.linspace(COLOR_LOW_CUTOFF, COLOR_HIGH_CUTOFF, 256))
    return LinearSegmentedColormap.from_list("YlOrRd_clipped", colors)

def _draw_pitch(ax):
    ax.set_xlim(0, PITCH_X); ax.set_ylim(0, PITCH_Y)
    ax.add_patch(plt.Rectangle((0,0), PITCH_X, PITCH_Y, fill=False, lw=1))
    ax.axvline(PITCH_X/2, lw=0.8, alpha=0.6)
    for x0 in [0, PITCH_X-18]:
        ax.add_patch(plt.Rectangle((x0, (PITCH_Y-44)/2), 18, 44, fill=False, lw=0.8))
    for x0 in [0, PITCH_X-6]:
        ax.add_patch(plt.Rectangle((x0, (PITCH_Y-20)/2), 6, 20, fill=False, lw=0.8))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.invert_yaxis()

def render_network(nodes, edges, title, min_edge_count, count_label):
    import pandas as pd  # local
    if nodes is None or edges is None or len(nodes)==0 or len(edges)==0:
        fig, ax = plt.subplots(figsize=(8.6, 6.0))
        _draw_pitch(ax); ax.set_title(title or "")
        ax.text(PITCH_X/2, PITCH_Y/2, "No data for current filters", ha="center", va="center")
        return fig_to_data_uri(fig)

    edges = edges.copy()
    edges["src"] = edges["src"].astype(str).str.strip()
    edges["dst"] = edges["dst"].astype(str).str.strip()
    edges["count"] = pd.to_numeric(edges["count"], errors="coerce")
    edges = edges[edges["count"] >= float(min_edge_count)].copy()
    if DROP_SELF_LOOPS:
        edges = edges[edges["src"] != edges["dst"]]

    cmap = get_yellow_to_red_cmap()
    vmin, vmax = 3, float(COUNT_CEILING)
    if COLOR_NORM == "log":
        norm = mcolors.LogNorm(vmin=1e-3, vmax=vmax, clip=True)
    elif COLOR_NORM == "power":
        norm = mcolors.PowerNorm(gamma=COLOR_GAMMA, vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    nodes_idx = nodes.set_index("node")[["x","y"]]
    ep = (edges
          .merge(nodes_idx, left_on="src", right_index=True, how="left").rename(columns={"x":"x_src","y":"y_src"})
          .merge(nodes_idx, left_on="dst", right_index=True, how="left").rename(columns={"x":"x_dst","y":"y_dst"}))
    ep = ep.dropna(subset=["x_src","y_src","x_dst","y_dst"])

    dir_set = {(r["src"], r["dst"]) for _, r in ep.iterrows()}
    bidir_pairs = {tuple(sorted((u, v))) for (u, v) in dir_set if (v, u) in dir_set and u != v}

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    _draw_pitch(ax)

    ep = ep.sort_values("count")
    for _, r in ep.iterrows():
        x1, y1 = float(r["x_src"]), float(r["y_src"])
        x2, y2 = float(r["x_dst"]), float(r["y_dst"])
        col = cmap(norm(float(r["count"])))
        pair_key = tuple(sorted((r["src"], r["dst"])))
        if pair_key in bidir_pairs:
            sign = +1.0 if r["src"] < r["dst"] else -1.0
            conn = f"arc3,rad={sign * ARROW_CURVE_RAD}"
        else:
            conn = "arc3,rad=0.0"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=conn,
            arrowstyle='-|>', mutation_scale=12,
            shrinkA=8.0, shrinkB=8.0,
            linewidth=ARROW_LW, color=col, alpha=0.95, zorder=2
        )
        ax.add_patch(arrow)

    ax.scatter(nodes["x"], nodes["y"], s=NODE_SIZE, alpha=0.95,
               color="white", edgecolor="black", linewidth=0.8, zorder=3)
    for _, r in nodes.iterrows():
        ax.text(r["x"], r["y"]+1.6, str(r["node"]), ha="center", va="bottom", fontsize=8)

    ax.set_title(title or "")
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                        extend=("max" if SHOW_10PLUS_ARROW else "neither"))
    cbar.set_label(count_label)
    ticks = COLORBAR_TICKS or np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    tick_labels = []
    for t in ticks:
        if abs(t - vmax) < 1e-9: tick_labels.append(f"{int(vmax)}+")
        else: tick_labels.append(str(int(t)) if float(t).is_integer() else f"{t:g}")
    cbar.set_ticklabels(tick_labels)
    return fig_to_data_uri(fig)

def fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

matches_df = list_england_matches(TEAM)
AGG_VALUE = -1
MATCH_OPTIONS = [{"label": "Average of all matches", "value": AGG_VALUE}] + [
    {"label": lab, "value": int(mid)} for mid, lab in zip(matches_df["match_id"], matches_df["label"])
]

app.layout = dbc.Container([

    dbc.Card(
        dbc.CardBody([
            html.H2("England — Directed Passing Networks (EURO 2025)", className="mb-0")
        ]),
        className="mb-3 bg-primary text-white border-0"
    ),


    dbc.Row([
        dbc.Col([
            dbc.Label("Match"),
            dcc.Dropdown(MATCH_OPTIONS, value=AGG_VALUE, id="match-dd")
        ], md=6),
        dbc.Col([
            dbc.Label(""),
            dbc.Checklist(
                options=[{"label": " Progressive only", "value": "prog"}],
                value=["prog"], id="prog-toggle", switch=True
            )
        ], md=6),
    ], className="g-3 mb-2"),


    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Label("Min passes shown"),
                    dcc.Slider(
                        id="min-edge", min=1, max=10, step=1, value=0,
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ])
            ),
            md=6
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    dbc.Label("Min pass length (meters)"),
                    dcc.Slider(
                        id="min-len", min=0, max=40, step=5, value=0,
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ])
            ),
            md=6
        ),
    ], className="g-3 mb-3"),

    html.Div(id="title"),
    html.Img(
        id="plot",
        style={"width": "100%", "maxWidth": "860px", "border": "1px solid #ddd", "borderRadius": "8px"}
    ),
    html.Div(id="footnote", className="text-muted mt-2", style={"fontSize": "0.9rem"}),
], fluid=True)

@app.callback(
    Output("plot","src"),
    Output("title","children"),
    Output("footnote","children"),
    Input("match-dd","value"),
    Input("prog-toggle","value"),
    Input("min-edge","value"),
    Input("min-len","value"),
)
def update_plot(match_value, prog_values, min_edge, min_len):
    progressive = "prog" in (prog_values or [])
    min_forward = 0.0 if progressive else -1e9  # ignored when not progressive
    if match_value == AGG_VALUE:
        mids = matches_df["match_id"].astype(int).tolist()
        nodes, edges = build_aggregate(mids, TEAM, progressive, float(min_len), float(min_forward), per_game_average=True)
        title = "England — Average directed passing network"
        count_label = "Avg passes per game"
    else:
        nodes, edges = build_nodes_edges_for_match(int(match_value), TEAM, progressive, float(min_len), float(min_forward))
        lab = matches_df.set_index("match_id").loc[int(match_value), "label"]
        title = f"England — {lab}"
        count_label = "Pass count"
    img_src = render_network(nodes, edges, title, min_edge_count=int(min_edge), count_label=count_label)
    foot = "Filters: " + ("Progressive only" if progressive else "All completed") + f", min length ≥ {int(min_len)} m, edges ≥ {int(min_edge)}"
    return img_src, html.H4(title), foot


if __name__ == "__main__":
    app.run(debug=True)
