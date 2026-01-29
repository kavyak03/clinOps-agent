import json
from pathlib import Path

def _safe_join_lines(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(s).strip() for s in x if str(s).strip())
    return str(x).strip()

def _clean(s: str) -> str:
    """
    Make strings safe for JSONL-by-line parsing:
    - remove/escape real newline characters
    - remove problematic unicode line separators
    """
    if s is None:
        return ""
    s = str(s)
    # normalize newlines and then escape them
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    # unicode line separators that can act like newlines in some tools
    s = s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return s.strip()

def main(config: str = "pqa_labeled", split: str = "train", max_examples: int = 2000):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"Install dependency: pip install datasets\nOriginal error: {e}")

    out_dir = Path("data/corpora/public")
    out_dir.mkdir(parents=True, exist_ok=True)

    # NEW FILE NAME to avoid reusing any corrupt file
    out_path = out_dir / "pubmedqa_corpus_singleline.jsonl"

    ds = load_dataset("qiaojin/PubMedQA", config, split=split)

    n = min(len(ds), max_examples)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            ex = ds[i]

            question = _clean(ex.get("question", ""))
            context = _clean(_safe_join_lines(ex.get("context", "")))
            long_answer = _clean(ex.get("long_answer", ""))
            final_decision = _clean(ex.get("final_decision", ""))

            text = _clean(
                f"Question: {question}\n"
                f"Context (abstract): {context}\n"
                f"Long answer (conclusion): {long_answer}\n"
                f"Short label: {final_decision}\n"
            )

            row = {
                "doc_id": f"PUBMEDQA_{i:06d}",
                "source": f"PubMedQA (public) [{config}/{split}]",
                "title": (question[:120] if question else f"PubMedQA_{i:06d}"),
                "text": text,
            }

            # json.dumps already escapes quotes; we've removed real newlines.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {n} docs to {out_path}")

if __name__ == "__main__":
    main()