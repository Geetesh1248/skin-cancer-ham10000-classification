"""
train_moe_v2.py
───────────────
Wrapper linking to the enhanced train_moe_final.py pipeline.
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    final_script = Path(__file__).resolve().parent / "train_moe_final.py"
    runpy.run_path(str(final_script), run_name="__main__")
