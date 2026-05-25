from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.langgraph_demo.decision_graph import run_langgraph_demo


def main() -> None:
    result = run_langgraph_demo(
        question="Use a graph workflow to validate whether eGFR and A1c support CKD-related risk."
    )
    print(json.dumps(result["final_output"], indent=2))
    print(json.dumps({"used_langgraph_runtime": result.get("used_langgraph_runtime", False)}, indent=2))


if __name__ == "__main__":
    main()
