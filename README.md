# Beat2Vibe — reproducible research implementation

This repository is an **independent, executable reconstruction from the final October 2026 Beat2Vibe manuscript**, not the authors' original experimental code. It implements acoustic and rhythm processing, pairwise baseline, self-supervised contrastive vibe learning, training-only K-means prototypes, multimodal fusion, learned beat transitions, five-way transition classification, and adaptive transition-conditioned next-track ranking. It additionally contains MERT/CLMR feature-baseline adapters, ablation variants, evaluation metrics, tuning, paired testing, and the controller-weight visualization.

**Research-integrity notice:** This repository does **not** include the original audio, original human annotation files, official MERT/CLMR features, trained weights, or genuine experiment records. It does **not** reproduce or endorse the manuscript's reported percentages merely because they appear in the PDF. Newly produced numbers may differ. Some architecture dimensions/choices are clearly specified here as reproducible implementation choices because the manuscript leaves them underspecified; matching its parameter-count table is NOT claimed.

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

Make `data/tracks.csv` and `data/pairs.csv` following [data/README.md](data/README.md). The `transition` column must be **human-annotated**, not reconstructed from valence/tempo thresholds. Audio paths can be absolute or relative to `tracks.csv`. Store large recordings elsewhere; audio files and checkpoints are ignored by Git.

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

For ablations rerun `train` with `--variant no_vibe`, `no_beat`, `audio_only`, or `audio_rhythm`, each with a unique `--out` checkpoint. `--variant fixed` holds the scoring weights fixed while preserving the learned pairwise transition classifier. The reported feature shapes/model capacity may differ across ablations; see `beat2vibe/models.py`.

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

The repository intentionally does **not** label a newly initialized convolutional network as pretrained CLMR or pretrained MERT. The external feature baselines apply the same train/test track pairs and an explicit 128-hidden-unit pairwise classifier; the cosine-ranking benchmark is computed over the same deterministic candidate protocol. The official CLMR feature export and any upstream dependency installation must be documented in the final artifact release.

## Tuning and evaluation

```bash
python run.py tune --data data/processed --ssl results/vibe_pretrained.pt --out results/tuning.json
```

`train` checkpoints store configuration, training-only normalization and tag vocabulary, model state, validation macro-F1, and the final epoch's number of ranking triplets. Default splits allocate 85% of *tracks* to the training side, reserve 10% of that for internal validation, and 15% to final test; any cross-partition annotated pairs are **discarded** (their exact number is logged). Train/test split files are written during `prepare`.

The classification evaluator outputs `accuracy`, macro-P/R/F1 and the five-class confusion matrix plus per-pair predictions. The ranking evaluator writes per-query/track predicted scores and policy weights, with Recall@5/10, NDCG@5/10 and MRR. Paired Wilcoxon tests and bootstrap confidence intervals come from aligned actual query outputs, not fixed draft values.

**Ranking-ground-truth limitation:** the manuscript describes 100-candidate lists, but the provided paper does not include original sequential listening logs. By default, each ranked query has ONE annotated transition-positive track and 99 **unjudged** randomly sampled candidates. Those are treated as negative **for this diagnostic only**. This does not establish real next-listen relevance, and should never be presented as fully judged next-track recommendation evidence. This is explicitly marked in every ranking output with `relevance_source`. To validate real-world ranking, supply human preference/judged candidate annotations or historical next-track logs instead.

**Ranking-supervision limitation:** margin loss requires positive and contrasting-label candidates for the *same current track*. If such candidates are unavailable, the supervised ranking loss stays inactive (and the training log tells you). A complete experimental evaluation must confirm nonzero triplet coverage.

## Data and provenance

The manuscript combines DEAM, MTG-Jamendo and FMA-small, plus an in-house Indian music collection. Original dataset access is documented in [data/README.md](data/README.md); no audio is redistributed here. Confirm redistribution rights before releasing the Indian collection or listener-study data. If annotations or restricted audio cannot be shared, describe exact limitations and provide lawful derived features where possible.

## Manuscript-to-code correspondence

| Manuscript | Repository implementation |
|---|---|
| Eqs. 1–3 audio, rhythm, transition cues | `beat2vibe/data.py` |
| Eqs. 4–7 pairwise baseline | `PairwiseBaseline` in `beat2vibe/models.py` |
| Eq. 8 contrastive vibe loss and Eq. 9 prototypes | `VibeLearner`, `pretrain` in `beat2vibe/experiment.py` |
| Eqs. 10–11 beat delta / embedding | `Beat2Vibe.make_delta`, `Beat2Vibe.beat` |
| Eq. 12 track-level fusion | `Beat2Vibe.track` |
| Eqs. 13–18 controller, policy score, margin ranking | `Beat2Vibe.pair`, `train`, `evaluate_ranking` |
| Classification / recommendation results | `beat2vibe/metrics.py`, `evaluate` |
| Statistical testing | `beat2vibe/analysis.py` |

### Important deviations/implementation details

* The present code uses a 3-convolution-layer acoustic CNN (kernels 5/3/3; 16/32/64 channels) and 128-dimensional acoustic features. SSL learns 64-dimensional unit-normalized projections; train-only K-means initializes 8 prototypes by default. Audio is loaded at 16 kHz, cropped to 12 seconds and represented using 128 Mel bands, first 128 frames.
* The SSL audio encoder and projection head are **frozen** during downstream training so that prototypes are stable. Downstream acoustic encoder initialization comes from that SSL model.
* Rhythm delta scale factors and class prototypes are estimated exclusively from training pairs. Missing valence/arousal values have explicit masks; train-only tag vocabulary and normalization prevent leakage.
* The paper does not specify all details above, so they are **transparent new implementation selections**, not a claim of an exact historical rerun.
* The supplied PDF has a `track-disjoint 85:15` split, yet its actual partition files and original code were not supplied. This implementation reserves validation tracks within the training side, as noted above.

## Reproducibility release checklist

Before any publication/claim of replication, archive the exact audio IDs/versions, annotation guidelines and actual labels, train/val/test manifests, baseline exported embeddings and original model revisions, environment and random seeds, configuration/search records, fitted model weights, per-query outputs and test statistics, plus the real user-study design/raw ratings as permitted. None of these missing original artifacts are reconstructed or forged here.

## Publish this prepared local repository to GitHub

This archive includes a locally initialized Git history but **has not been uploaded to any GitHub account**. After creating a new empty repository in your own account (for example `Beat2Vibe`), run from the extracted folder:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/Beat2Vibe.git
git branch -M main
git push -u origin main
```

Replace `YOUR_ACCOUNT` with your actual account name. Do not upload licensed audio, private participant responses or unreviewed credentials. Alternatively, connect the GitHub integration in ChatGPT for assisted publishing.
