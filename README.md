# Beat2Vibe — A regional music recommender system

This work presents a transition-aware music recommendation framework for controlled next-track prediction. Existing systems recommend similar songs but often fail to preserve listener preference, mood continuity, and smooth song-to-song transitions. Ranked playlists also include weak and low-probability candidates. To address these issues, our framework models next-track selection as a five-class transition task. It combines acoustic features, vibe-state learning, beat-transition modeling, and adaptive agentic ranking.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py --help
pytest -q
```

All commands are run from the repository root. For GPU execution use PyTorch compatible with your CUDA installation. Use Python 3.10–3.12. `config.yaml` pins the complete experiment settings and random seed.

## Data format

Make `data/tracks.csv` and `data/pairs.csv` following [data/README.md](data/README.md). The `transition` column must be **human-annotated**, not reconstructed from valence/tempo thresholds. Audio paths can be absolute to `tracks.csv`.

## Main pipeline

```bash
python run.py prepare --tracks data/tracks.csv --pairs data/pairs.csv --out data/processed
python run.py pretrain --data data/processed --out results/vibe_pretrained.pt
python run.py train --variant full --data data/processed --ssl results/vibe_pretrained.pt --out results/full.pt
python run.py train --variant fixed --data data/processed --ssl results/vibe_pretrained.pt --out results/fixed.pt
python run.py train --variant baseline --data data/processed --out results/pairwise.pt
python run.py evaluate --data data/processed --checkpoint results/full.pt --out results/full_test
python run.py evaluate --data data/processed --checkpoint results/fixed.pt --out results/fixed_test
python run.py evaluate --data data/processed --checkpoint results/pairwise.pt --out results/pairwise_test
python run.py compare --a results/full_test_ranking.csv --b results/fixed_test_ranking.csv --out results/fixed_vs_adaptive.json
python run.py plot-weights --ranking-csv results/full_test_ranking.csv --out results/controller_weights.png
```

For ablations rerun `train` with `--variant no_vibe`, `no_beat`, `audio_only`, or `audio_rhythm`, each with a unique `--out` checkpoint. `--variant fixed` holds the scoring weights fixed while preserving the learned pairwise transition classifier. The reported feature shapes/model capacity may differ across ablations: see `beat2vibe/models.py`.

## Baseline comparisons: genuine MERT and CLMR

**MERT** embeddings can be extracted using the original pretrained weights. This requires reviewing and explicitly opting in to remote model code because the published Hugging Face model uses `trust_remote_code`:

```bash
pip install -r requirements-optional.txt
python run.py extract-mert --tracks data/tracks.csv --out data/mert.npz --device cpu --allow-remote-code
python run.py external-baseline --data data/processed --features data/mert.npz --out results/mert
```

MERT official model: https://huggingface.co/m-a-p/MERT-v1-95M (check its **noncommercial license**).

**CLMR:** obtain the actual official CLMR checkpoint and export track embeddings using the official code at https://github.com/spijkervet/clmr. Make a `.npz` containing `track_id` (strings) and `embedding` (shape `[N,D]`, float32), then run:

```bash
python run.py external-baseline --data data/processed --features data/clmr.npz --out results/clmr
```

## Data and provenance

The manuscript combines DEAM, MTG-Jamendo and FMA-small, plus an in-house Indian music collection. Original dataset access is documented in [data/README.md](data/README.md). The in-house Indian dataset will made available after the paper publication.

### Important deviations/implementation details

- The present code uses a 3-convolution-layer acoustic CNN (kernels 5/3/3, 16/32/64 channels) and 128-dimensional acoustic features. SSL learns 64-dimensional unit-normalized projections, train-only K-means initializes 8 prototypes by default. Audio is loaded at 16 kHz, cropped to 12 seconds and represented using 128 Mel bands, first 128 frames.
- The SSL audio encoder and projection head are **frozen** during downstream training so that prototypes are stable. Downstream acoustic encoder initialization comes from that SSL model.
- Rhythm delta scale factors and class prototypes are estimated exclusively from training pairs. Missing valence/arousal values have explicit masks, train-only tag vocabulary and normalization prevent leakage.
