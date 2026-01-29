from src.data.generate_synth_data import main

if __name__ == "__main__":
    main(out_dir="data/raw", n_patients=200, min_enc=2, max_enc=6)
