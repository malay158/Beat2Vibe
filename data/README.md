# Input data format and source references

Create `data/tracks.csv` with one row per song:

```csv
track_id,audio_path,valence,arousal,tag
song_0001,raw/song_0001.wav,0.31,0.56,mood_calm|instrument_piano
song_0002,raw/song_0002.wav,0.72,0.84,mood_happy|instrument_drums
```

Create `data/pairs.csv`, **only using your actual human-annotated transitions**:

```csv
current_id,candidate_id,transition
song_0001,song_0002,uplift
```

Allowed transitions (case insensitive): `sustain`, `uplift`, `soften`, `shift`, `drift`. Do not generate five-class labels automatically from VA/tempo thresholds and represent those as human annotations.

External official sources:

- DEAM: https://cvml.unige.ch/databases/DEAM/
- MTG-Jamendo: https://github.com/MTG/mtg-jamendo-dataset
- FMA-small: https://github.com/mdeff/fma
- Pretrained CLMR: https://github.com/spijkervet/clmr
- Pretrained MERT: https://huggingface.co/m-a-p/MERT-v1-95M

The Indian-domain audio/metadata source was described as in-house in the manuscript; its access location and license **were not supplied**.

`prepare` writes `audio_features.npz`, `tracks.csv`, `pairs.csv`, split manifests, and `config.json` inside `data/processed/`. Those outputs are excluded from Git by default until they are inspected for licensing/privacy.
