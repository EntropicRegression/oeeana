import sys

from emotion_analyzer.gui import run_gui
from emotion_analyzer.multi_gui import run_multi_gui


if __name__ == "__main__":
    if "--multi" in sys.argv[1:]:
        raise SystemExit(run_multi_gui())
    raise SystemExit(run_gui())
