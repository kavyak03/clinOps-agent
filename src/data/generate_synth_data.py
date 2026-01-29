import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

RNG = np.random.default_rng(42)

@dataclass
class Patient:
    patient_id: str
    age: int
    sex: str
    bmi: float
    smoker: str  # "never"|"former"|"current"
    htn: bool
    t2d: bool
    ckd: bool
    hld: bool

def _rand_date(start: datetime, days: int) -> datetime:
    return start + timedelta(days=int(RNG.integers(0, days)))

def generate_patients(n: int):
    patients = []
    for i in range(n):
        pid = f"P{i:05d}"
        age = int(RNG.integers(25, 85))
        sex = random.choice(["F", "M"])
        bmi = float(np.clip(RNG.normal(29, 6), 18, 55))
        smoker = random.choice(["never", "former", "current"])
        # comorbidities correlated with age/BMI
        htn = RNG.random() < (0.15 + 0.005 * (age - 25))
        t2d = RNG.random() < (0.08 + 0.004 * (bmi - 25))
        ckd = RNG.random() < (0.05 + 0.003 * (age - 40))
        hld = RNG.random() < (0.10 + 0.004 * (age - 30))
        patients.append(Patient(pid, age, sex, bmi, smoker, bool(htn), bool(t2d), bool(ckd), bool(hld)))
    return patients

def generate_labs_for_patient(p: Patient, start: datetime, n_encounters: int):
    labs = []
    for e in range(n_encounters):
        dt = _rand_date(start, 365)
        # generate correlated labs
        a1c = float(np.clip(RNG.normal(5.6 + (1.6 if p.t2d else 0.2), 0.7), 4.2, 13.5))
        ldl = float(np.clip(RNG.normal(115 + (25 if p.hld else 0), 30), 40, 260))
        creat = float(np.clip(RNG.normal(0.9 + (0.5 if p.ckd else 0.0), 0.25), 0.4, 4.0))
        egfr = float(np.clip(120 / (creat * 1.2), 5, 140))  # synthetic inverse-ish relation
        labs.append({
            "patient_id": p.patient_id,
            "encounter_id": f"{p.patient_id}_E{e:03d}",
            "date": dt.strftime("%Y-%m-%d"),
            "a1c": a1c,
            "ldl": ldl,
            "creatinine": creat,
            "egfr": egfr,
        })
    return labs

def generate_notes_for_patient(p: Patient, labs_rows):
    notes = []
    for row in labs_rows:
        a1c = row["a1c"]; egfr = row["egfr"]; ldl = row["ldl"]
        meds = []
        if p.htn: meds.append(random.choice(["lisinopril", "losartan", "amlodipine"]))
        if p.hld: meds.append(random.choice(["atorvastatin", "rosuvastatin", "pravastatin"]))
        if p.t2d: meds.append(random.choice(["metformin", "semaglutide", "insulin glargine"]))
        meds_str = ", ".join(meds) if meds else "none"

        assessment = []
        if a1c >= 6.5:
            assessment.append("Diabetes likely based on elevated A1c.")
        elif 5.7 <= a1c < 6.5:
            assessment.append("Prediabetes range A1c; lifestyle counseling discussed.")
        if egfr < 45:
            assessment.append("Reduced kidney function noted; medication dosing reviewed.")
        if ldl > 190:
            assessment.append("Severe hypercholesterolemia; high intensity statin indicated.")

        note = f"""
HPI: Follow-up visit. Patient reports variable diet/exercise adherence.
Meds: {meds_str}.
Labs: A1c {a1c:.1f}%, LDL {ldl:.0f} mg/dL, eGFR {egfr:.0f} mL/min/1.73m2.
Assessment/Plan: {' '.join(assessment) if assessment else 'Stable; continue routine follow-up.'}
        """.strip()

        notes.append({
            "patient_id": row["patient_id"],
            "encounter_id": row["encounter_id"],
            "date": row["date"],
            "note_text": note
        })
    return notes

def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def main(out_dir: str = "data/raw", n_patients: int = 200, min_enc: int = 2, max_enc: int = 6):
    out = Path(out_dir)
    start = datetime(2024, 1, 1)

    patients = generate_patients(n_patients)
    patient_rows = [asdict(p) for p in patients]

    labs_all, notes_all = [], []
    for p in patients:
        n_enc = int(RNG.integers(min_enc, max_enc + 1))
        labs = generate_labs_for_patient(p, start, n_enc)
        notes = generate_notes_for_patient(p, labs)
        labs_all.extend(labs)
        notes_all.extend(notes)

    write_jsonl(out / "patients.jsonl", patient_rows)
    write_jsonl(out / "labs.jsonl", labs_all)
    write_jsonl(out / "notes.jsonl", notes_all)

    print(f"Wrote patients={len(patient_rows)}, labs={len(labs_all)}, notes={len(notes_all)} to {out}")

if __name__ == "__main__":
    main()
