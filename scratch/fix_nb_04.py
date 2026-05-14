import json
from pathlib import Path

nb_path = Path(r"d:\AI_thucchien\Day22\Day22-Track3-DPO-Alignment-Lab\colab\Lab22_DPO_T4_Complete.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Fix: rename 04-inference-comparison.png to 04-side-by-side-table.png
        if '04-inference-comparison.png' in source:
            source = source.replace('04-inference-comparison.png', '04-side-by-side-table.png')
            
        cell["source"] = [line + ("\n" if not line.endswith("\n") else "") for line in source.split("\n")]
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Notebook updated (04 screenshot renamed).")
