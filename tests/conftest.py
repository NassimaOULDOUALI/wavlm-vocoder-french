"""pytest configuration — adds source directories to sys.path.
 
Insertion order matters: last insert(0,...) wins.
We insert in reverse so 01_baseline ends up at index 0 (highest priority).
"""
import sys
from pathlib import Path
 
root = Path(__file__).parent
# Reverse order: 03 first (lowest priority), 01 last (highest priority)
sys.path.insert(0, str(root / "03_ablation_study" / "src"))
sys.path.insert(0, str(root / "02_gan_vocoder" / "src"))
sys.path.insert(0, str(root / "01_baseline_no_gan"))
