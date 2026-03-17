"""pytest configuration — adds source directories to sys.path."""
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "01_baseline_no_gan"))
sys.path.insert(0, str(root / "02_gan_vocoder" / "src"))
sys.path.insert(0, str(root / "03_ablation_study" / "src"))
