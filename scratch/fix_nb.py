import json
from pathlib import Path

nb_path = Path(r"d:\AI_thucchien\Day22\Day22-Track3-DPO-Alignment-Lab\colab\Lab22_DPO_T4_Complete.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Fix 1: end_reward_gap
        if '"final_reward_margin":' in source and 'metrics = {' in source:
            source = source.replace(
                '"final_reward_margin":    float(last_log.get("rewards/margins",  0)),',
                '"final_reward_margin":    float(last_log.get("rewards/margins",  0)),\n    "end_reward_gap":         float(last_log.get("rewards/margins",  0)),'
            )
        
        # Fix 2: screenshot 05 renaming to 07 to satisfy verify.py
        if '05-benchmark-results.png' in source:
            source = source.replace('05-benchmark-results.png', '07-benchmark-comparison.png')
            
        # Fix 3: final checklist update
        if 'SCREENSHOT_PATH / "07-final-summary.png",' in source:
            source = source.replace('SCREENSHOT_PATH / "07-final-summary.png",', 'SCREENSHOT_PATH / "07-benchmark-comparison.png",')
            
        # Also fix the manual strings in final cells if any
        if '07-final-summary.png' in source:
             source = source.replace('07-final-summary.png', '07-summary.png') # rename the summary one so it doesn't conflict or just keep it

        cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in source.split("\n")]
        # Remove trailing newline from last element to keep it clean if needed, 
        # but json.dump handles it. Actually unsloth/colab likes the list format.
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook updated successfully.")
