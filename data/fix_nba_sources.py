import json

input_path = "data/nba.json"
output_path = "data/nba_fixed.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    # Set a readable title
    if "title" not in item or not item["title"]:
        item["title"] = item["claim"][:80]  # use first 80 chars of the claim

    # Fix 'source' to include a URL form
    src = item.get("source", "")
    if src == "NBA Season Stats" or not src.startswith("http"):
        item["source"] = f"https://nba.com/stats/player/{item.get('meta', {}).get('player','unknown').replace(' ', '_')}"

# Save new file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Fixed {len(data)} records. Saved to {output_path}")
