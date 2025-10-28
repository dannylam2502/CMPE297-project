"""
Load FEVER dataset and convert to system format.
Saves to fever.json in the same directory as this script.
"""

import json
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets package...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets", "--break-system-packages"])
    from datasets import load_dataset


def load_fever():
    """
    Download FEVER dataset and convert to system format.
    Saves to fever.json in script directory.
    """
    print("Downloading FEVER dataset...")
    
    dataset = load_dataset("fever", "v1.0", split="train")
    
    print(f"Processing {len(dataset)} claims...")
    
    data = []
    for i, row in enumerate(dataset):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(dataset)}...")
        
        label = row.get("label", "NOT ENOUGH INFO")
        confidence = 1.0 if label in ["SUPPORTS", "REFUTES"] else 0.5
        
        data.append({
            "id": i,
            "claim": row["claim"],
            "source": "wikipedia",
            "confidence": confidence,
            "label": label
        })
    
    script_dir = Path(__file__).parent
    output_path = script_dir / "fever.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} claims to {output_path.absolute()}")
    return len(data)


if __name__ == "__main__":
    try:
        count = load_fever()
        print(f"FEVER dataset loaded: {count} claims")
    except Exception as e:
        print(f"Error loading FEVER: {e}")
        import traceback
        traceback.print_exc()
        exit(1)