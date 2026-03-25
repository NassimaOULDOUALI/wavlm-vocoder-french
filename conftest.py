"""pytest configuration — adds source directories to sys.path."""
import sys
from pathlib import Path

root = Path(__file__).parent
# Reverse order: last insert(0,...) has highest priority
# → 01_baseline_no_gan ends up at index 0
sys.path.insert(0, str(root / "03_ablation_study" / "src"))
sys.path.insert(0, str(root / "02_gan_vocoder" / "src"))
sys.path.insert(0, str(root / "01_baseline_no_gan"))
