import json
from pathlib import Path
import pandas as pd

def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def basic_qc(patients: pd.DataFrame, labs: pd.DataFrame, notes: pd.DataFrame) -> dict:
    qc = {}
    qc["n_patients"] = len(patients)
    qc["n_labs"] = len(labs)
    qc["n_notes"] = len(notes)

    qc["missingness_patients"] = patients.isna().mean().to_dict()
    qc["missingness_labs"] = labs.isna().mean().to_dict()
    qc["missingness_notes"] = notes.isna().mean().to_dict()

    qc["lab_ranges"] = {
        "a1c_min": float(labs["a1c"].min()),
        "a1c_max": float(labs["a1c"].max()),
        "egfr_min": float(labs["egfr"].min()),
        "egfr_max": float(labs["egfr"].max()),
        "ldl_min": float(labs["ldl"].min()),
        "ldl_max": float(labs["ldl"].max()),
        "creatinine_min": float(labs["creatinine"].min()),
        "creatinine_max": float(labs["creatinine"].max()),
    }

    qc["dup_encounters_labs"] = int(labs.duplicated(["patient_id","encounter_id"]).sum())
    qc["dup_encounters_notes"] = int(notes.duplicated(["patient_id","encounter_id"]).sum())

    notes_len = notes["note_text"].astype(str).str.len()
    qc["note_len_mean"] = float(notes_len.mean())
    qc["note_len_p95"] = float(notes_len.quantile(0.95))

    return qc
