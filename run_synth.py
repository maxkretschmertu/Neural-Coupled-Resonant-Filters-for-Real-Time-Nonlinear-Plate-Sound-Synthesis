"""Run the Phase-1 refactored plate synthesizer from the repository root."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plate_synth.app import main


if __name__ == "__main__":
    main()
