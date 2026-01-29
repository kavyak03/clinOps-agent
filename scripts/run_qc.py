import json
from pathlib import Path
from src.data.qc import read_jsonl, basic_qc
from src.viz.plots import (
    plot_missingness, plot_lab_hist, plot_egfr_vs_creatinine, plot_note_length
)

def main():
    raw = Path("data/raw")
    rep = Path("reports")
    rep.mkdir(exist_ok=True)

    patients = read_jsonl(raw / "patients.jsonl")
    labs = read_jsonl(raw / "labs.jsonl")
    notes = read_jsonl(raw / "notes.jsonl")

    qc = basic_qc(patients, labs, notes)
    (rep / "qc_summary.json").write_text(json.dumps(qc, indent=2))

    plot_missingness(patients, "Missingness: patients", str(rep / "missing_patients.png"))
    plot_missingness(labs, "Missingness: labs", str(rep / "missing_labs.png"))
    plot_missingness(notes, "Missingness: notes", str(rep / "missing_notes.png"))

    for col in ["a1c", "ldl", "egfr", "creatinine"]:
        plot_lab_hist(labs, col, str(rep / f"hist_{col}.png"))

    plot_egfr_vs_creatinine(labs, str(rep / "egfr_vs_creatinine.png"))
    plot_note_length(notes, str(rep / "note_length.png"))

    print("QC complete. See reports/")

if __name__ == "__main__":
    main()
