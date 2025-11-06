"""
Load NBA Season Stats and convert to pipeline format.
Saves to nba.json inside the same directory.
"""

import json
import pandas as pd
from pathlib import Path
import random

def load_nba():
    random.seed(42)

    # === Paths ===
    csv_path = Path("CMPE297-project\docs\Seasons_Stats.csv")
    out_path = Path(__file__).parent / "nba.json"

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)

    # --- Basic cleaning ---
    keep_cols = ["Year", "Player", "Tm", "PTS", "AST", "TRB", "G"]
    df = df[keep_cols].dropna(subset=["Player", "PTS", "AST", "TRB", "G"])

    # remove duplicates (some players have "TOT" + team rows)
    df = df.sort_values(["Player", "Year", "PTS"], ascending=[True, False, False])
    df = df.drop_duplicates(subset=["Player", "Year"], keep="first").reset_index(drop=True)

    # --- convert totals per game ---
    df["PTS_pg"] = (df["PTS"] / df["G"]).round(1)
    df["AST_pg"] = (df["AST"] / df["G"]).round(1)
    df["TRB_pg"] = (df["TRB"] / df["G"]).round(1)

    # --- Generate claims ---
    data = []
    for i, r in df.iterrows():
        player = str(r["Player"]).strip()
        year   = int(r["Year"])
        team   = str(r["Tm"]).strip()
        pts, ast, reb = r["PTS_pg"], r["AST_pg"], r["TRB_pg"]

        claim = f"{player} averaged {pts} points, {ast} assists, and {reb} rebounds per game in the {year} NBA season for {team}."

        # introduce small false claims to simulate misinformation
        if random.random() < 0.15:
            fake_pts = round(pts + random.uniform(2, 5), 1)
            claim = claim.replace(f"{pts}", f"{fake_pts}")
            label = "REFUTES"
        else:
            label = "SUPPORTS"

        data.append({
            "id": i,
            "claim": claim,
            "source": "NBA Season Stats",
            "label": label,
            "confidence": 1.0,
            "meta": {
                "player": player,
                "season": year,
                "team": team,
                "true_pts_pg": float(pts),
                "true_ast_pg": float(ast),
                "true_reb_pg": float(reb)
            }
        })

    # --- Save output ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} records to {out_path.resolve()}")
    return len(data)


if __name__ == "__main__":
    try:
        n = load_nba()
        print(f"NBA dataset processed successfully: {n} records")
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
