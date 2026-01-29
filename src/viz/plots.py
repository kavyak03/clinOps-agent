import matplotlib.pyplot as plt
import pandas as pd

def plot_missingness(df: pd.DataFrame, title: str, outpath: str):
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure()
    plt.bar(miss.index, miss.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Fraction missing")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_lab_hist(df: pd.DataFrame, col: str, outpath: str):
    plt.figure()
    plt.hist(df[col].dropna().values, bins=30)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_egfr_vs_creatinine(labs: pd.DataFrame, outpath: str):
    plt.figure()
    plt.scatter(labs["creatinine"], labs["egfr"], s=10)
    plt.xlabel("Creatinine")
    plt.ylabel("eGFR")
    plt.title("Sanity check: eGFR vs Creatinine")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_note_length(notes: pd.DataFrame, outpath: str):
    lengths = notes["note_text"].astype(str).str.len()
    plt.figure()
    plt.hist(lengths.values, bins=30)
    plt.xlabel("Note length (characters)")
    plt.ylabel("Count")
    plt.title("Clinical note length distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
