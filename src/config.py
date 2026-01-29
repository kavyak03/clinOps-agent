from dataclasses import dataclass

@dataclass
class Config:
    # data sizes
    n_patients: int = 200
    min_encounters: int = 2
    max_encounters: int = 6

    # RAG
    embed_model_name: str = "all-MiniLM-L6-v2"
    chunk_max_chars: int = 500
    chunk_overlap: int = 80
    top_k: int = 5
