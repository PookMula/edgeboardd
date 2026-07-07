#!/usr/bin/env python3
"""
build_arsenal.py — Edge Board pitch-level arsenal builder.

Pulls this season's pitch-by-pitch Statcast data via pybaseball and aggregates it into
a compact arsenal.json with:
  • pitchers[id] -> arsenal (each pitch type: usage% + barrel/HR/hard-hit/EV allowed)
  • batters[id]  -> contact quality by pitch type + an "all pitches" baseline
Both are keyed by MLBAM player id (the same ids Edge Board already uses), so the site
matches them automatically. Drop arsenal.json next to edge-board.html.

Run locally:  python build_arsenal.py
(Needs: pip install pybaseball pandas numpy)
"""
import json, sys, datetime as dt
import numpy as np
import pandas as pd
from pybaseball import statcast

SEASON = dt.date.today().year
OPENER = f"{SEASON}-03-15"            # a little before opening day is fine
TODAY  = dt.date.today().isoformat()

MIN_PITCHER_PITCHES   = 150           # ignore pitchers with tiny samples
MIN_PITCHER_BIP_TYPE  = 15            # min balls-in-play of a pitch type for "allowed" rates
MIN_BATTER_BIP_ALL    = 40            # min BIP to include a batter at all
MIN_BATTER_BIP_TYPE   = 8             # min BIP of a pitch type to store that split

PITCH_NAME = {"FF":"four-seam","FA":"fastball","SI":"sinker","FT":"two-seam","FC":"cutter",
              "SL":"slider","ST":"sweeper","CU":"curveball","KC":"knuckle-curve","CS":"slow curve",
              "CH":"changeup","FS":"splitter","SV":"slurve","SC":"screwball","EP":"eephus",
              "KN":"knuckleball","FO":"forkball"}


def spray_pull(df):
    """True where the ball was hit to the batter's PULL side (uses hc_x/hc_y spray angle)."""
    angle = np.degrees(np.arctan2(df["hc_x"] - 125.42, 198.27 - df["hc_y"]))
    return (((df["stand"] == "R") & (angle < -10)) |
            ((df["stand"] == "L") & (angle > 10))).fillna(False)


def rnd(x, n):
    return None if x is None or (isinstance(x, float) and (np.isnan(x))) else round(float(x), n)


def agg_batter(d):
    n = len(d)
    if n == 0:
        return None
    return {
        "bip": int(n),
        "barrel":  rnd(d["is_barrel"].mean(), 4),
        "hardhit": rnd(d["is_hardhit"].mean(), 4),
        "ev":      rnd(d["launch_speed"].mean(), 1),
        "la":      rnd(d["launch_angle"].mean(), 1),
        "dist":    rnd(d["hit_distance_sc"].mean(), 0) if d["hit_distance_sc"].notna().any() else None,
        "pullbrl": rnd(d["is_pullbrl"].mean(), 4),
        "hr":      rnd(d["is_hr"].mean(), 4),
    }


def main():
    print(f"Pulling Statcast {OPENER} .. {TODAY} (this can take several minutes)...", flush=True)
    df = statcast(start_dt=OPENER, end_dt=TODAY)
    if df is None or df.empty:
        print("No Statcast data returned.", file=sys.stderr)
        sys.exit(1)

    df = df[df["pitch_type"].notna()].copy()
    df["is_bip"]     = df["description"].eq("hit_into_play")
    df["is_barrel"]  = df["launch_speed_angle"].eq(6)
    df["is_hardhit"] = df["launch_speed"].ge(95)
    df["is_hr"]      = df["events"].eq("home_run")
    try:
        df["is_pull"] = spray_pull(df)
    except Exception:
        df["is_pull"] = False
    df["is_pullbrl"] = df["is_barrel"] & df["is_pull"]

    bip = df[df["is_bip"]].copy()

    # ---------- PITCHERS: arsenal (usage) + what each pitch allows ----------
    pitchers = {}
    for pid, pdf in df.groupby("pitcher"):
        total = len(pdf)
        if total < MIN_PITCHER_PITCHES:
            continue
        thr = pdf["p_throws"].mode()
        throws = thr.iat[0] if not thr.empty else None
        pbip = bip[bip["pitcher"] == pid]
        pitches = []
        for pt, ptdf in pdf.groupby("pitch_type"):
            usage = len(ptdf) / total
            if usage < 0.02:
                continue
            b = pbip[pbip["pitch_type"] == pt]
            entry = {"type": pt, "name": PITCH_NAME.get(pt, pt), "usage": rnd(usage, 4)}
            if len(b) >= MIN_PITCHER_BIP_TYPE:
                entry["barrelAllowed"]  = rnd(b["is_barrel"].mean(), 4)
                entry["hrAllowed"]      = rnd(b["is_hr"].mean(), 4)
                entry["hardhitAllowed"] = rnd(b["is_hardhit"].mean(), 4)
                entry["evAllowed"]      = rnd(b["launch_speed"].mean(), 1)
            pitches.append(entry)
        pitches.sort(key=lambda x: -x["usage"])
        if pitches:
            pitchers[int(pid)] = {"throws": throws, "n": int(total), "pitches": pitches}

    # ---------- BATTERS: contact quality by pitch type + all-pitch baseline ----------
    batters = {}
    for bid, bdf in bip.groupby("batter"):
        if len(bdf) < MIN_BATTER_BIP_ALL:
            continue
        by_pitch = {}
        for pt, ptdf in bdf.groupby("pitch_type"):
            if len(ptdf) < MIN_BATTER_BIP_TYPE:
                continue
            by_pitch[pt] = agg_batter(ptdf)
        batters[int(bid)] = {
            "bip": int(len(bdf)),
            "hr": int(bdf["is_hr"].sum()),
            "all": agg_batter(bdf),
            "byPitch": by_pitch,
        }

    out = {
        "updated": TODAY,
        "season": SEASON,
        "pitchers": pitchers,
        "batters": batters,
        "counts": {"pitchers": len(pitchers), "batters": len(batters)},
    }
    with open("arsenal.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"Wrote arsenal.json — {len(pitchers)} pitchers, {len(batters)} batters.", flush=True)


if __name__ == "__main__":
    main()
