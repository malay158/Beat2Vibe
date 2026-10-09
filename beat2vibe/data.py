from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from . import CLASSES

RHYTHM_DIM = 6

def read_manifest(tracks_path, pairs_path):
    tracks = pd.read_csv(tracks_path).fillna({"tag": ""})
    pairs = pd.read_csv(pairs_path)
    required_tracks = {"track_id", "audio_path"}
    required_pairs = {"current_id", "candidate_id", "transition"}
    if not required_tracks.issubset(tracks.columns) or not required_pairs.issubset(pairs.columns):
        raise ValueError(f"Required columns: tracks={required_tracks}, pairs={required_pairs}")
    for key in ("track_id", "audio_path"):
        tracks[key] = tracks[key].astype(str)
    for key in ("current_id", "candidate_id"):
        pairs[key] = pairs[key].astype(str)
    pairs["transition"] = pairs.transition.astype(str).str.lower().str.strip()
    if tracks.track_id.duplicated().any():
        raise ValueError("Duplicate track_id values")
    if not pairs.transition.isin(CLASSES).all():
        raise ValueError("Unknown transition classes found")
    if not set(pairs.current_id).union(pairs.candidate_id).issubset(set(tracks.track_id)):
        raise ValueError("Pair refers to an unknown track_id")
    if (pairs.current_id == pairs.candidate_id).any():
        raise ValueError("Self-transitions are not supported")
    
    return tracks, pairs


def split_track_disjoint(tracks, pairs, seed, train_fraction=0.85, validation_within_train=0.1):
    
    ids = tracks.track_id.astype(str).tolist()
    if len(ids) < 7:
        raise ValueError("Need at least seven tracks for train/val/test splits")
    tr, te = train_test_split(ids, train_size=train_fraction, random_state=seed, shuffle=True)
    tr, va = train_test_split(tr, test_size=validation_within_train, random_state=seed, shuffle=True)
    mapping = {x: "train" for x in tr}
    mapping.update({x: "val" for x in va})
    mapping.update({x: "test" for x in te})
    a = pairs.current_id.map(mapping)
    b = pairs.candidate_id.map(mapping)
    assigned = pairs.copy()
    assigned["split"] = np.where(a == b, a, "cross_split_dropped")
    lost = int((assigned.split == "cross_split_dropped").sum())
    if lost:
        warnings.warn(f"Discarded {lost} cross-partition pairs to ensure track disjointness")
    return mapping, assigned


def extract_audio(audio_path, config):
    import librosa
    sr = int(config["sample_rate"])
    seconds = float(config["clip_seconds"])
    waveform, sr = librosa.load(str(audio_path), sr=sr, mono=True, duration=seconds)
    if waveform.size == 0:
        raise ValueError(f"Empty audio: {audio_path}")
    target = int(sr * seconds)
    waveform = np.pad(waveform[:target], (0, max(0, target - len(waveform))))
    hop = int(config["hop_length"])
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=2048, hop_length=hop,
        n_mels=int(config["n_mels"]), power=2.0)
    mel = librosa.power_to_db(mel, ref=np.max)
    # Each track's log-Mel uses a stable dB scale; no train/test data statistics leak.
    mel = np.clip((mel + 80.0) / 80.0, 0, 1)
    frames = int(config["n_frames"])
    mel = mel[:, :frames]
    mel = np.pad(mel, ((0, 0), (0, max(0, frames-mel.shape[1]))))
    onset = librosa.onset.onset_strength(y=waveform, sr=sr, hop_length=hop)
    try:
        tempo, beats = librosa.beat.beat_track(y=waveform, sr=sr, hop_length=hop)
        tempo = float(np.atleast_1d(tempo)[0])
        btime = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
        ibi = np.diff(btime)
    except Exception:
        tempo, beats, ibi = 0.0, [], np.array([])
    ibi_mean = float(np.mean(ibi)) if len(ibi) else 0.0
    ibi_std = float(np.std(ibi)) if len(ibi) else 0.0
    rhythm = np.array([tempo, len(beats), ibi_mean, ibi_std,
                       float(np.mean(onset)), float(np.std(onset))], dtype=np.float32)
    return mel.astype(np.float32), rhythm


def prepare(tracks_path, pairs_path, out_dir, config):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracks, pairs = read_manifest(tracks_path, pairs_path)
    mapping, pairs = split_track_disjoint(tracks, pairs, int(config['seed']),
        float(config['train_fraction']), float(config['validation_within_train']))
    # Relative audio paths are relative to the manifest, not the repository.
    root = Path(tracks_path).resolve().parent
    mels, rhythms, failures = [], [], []
    for record in tracks.to_dict('records'):
        p = Path(record['audio_path'])
        p = p if p.is_absolute() else root / p
        try:
            mel, rhythm = extract_audio(p, config)
        except Exception as e:
            failures.append((record['track_id'], str(e)))
            mel = np.zeros((int(config['n_mels']), int(config['n_frames'])), dtype=np.float32)
            rhythm = np.zeros(RHYTHM_DIM, dtype=np.float32)
        mels.append(mel)
        rhythms.append(rhythm)
    if failures:
        raise RuntimeError(f"Feature extraction failed for {len(failures)} tracks; examples: {failures[:3]}")
    tracks['split'] = tracks.track_id.map(mapping)
    # Data-only zip input supports no Python object deserialization.
    np.savez_compressed(out / 'audio_features.npz',
        mel=np.asarray(mels, np.float32), rhythm=np.asarray(rhythms, np.float32))
    tracks.to_csv(out / 'tracks.csv', index=False)
    pairs.to_csv(out / 'pairs.csv', index=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    for name in ('train','val','test'):
        tracks.loc[tracks.split == name, ['track_id']].to_csv(out / f'{name}_tracks.csv', index=False)
        pairs.loc[pairs.split == name].to_csv(out / f'{name}_pairs.csv', index=False)
    print(f"Prepared {len(tracks)} tracks, {len(pairs)} annotated pairs. Split pair counts: {pairs.split.value_counts().to_dict()}")
    return out


def rhythm_delta(r_i, r_j, eps=1e-6):
    
    a = np.asarray(r_i, dtype=np.float64)
    b = np.asarray(r_j, dtype=np.float64)
    return np.array([(b[0]-a[0])/(max(float(a[0]), 1.0)),
                     b[2]-a[2],b[3]-a[3],b[4]-a[4],b[5]-a[5]],np.float32)


def load_prepared(path, config):
    root = Path(path)
    tracks = pd.read_csv(root/'tracks.csv').fillna({'tag':''})
    pairs = pd.read_csv(root/'pairs.csv')
    ids = tracks.track_id.astype(str).tolist()
    for c in ['current_id','candidate_id']:
        pairs[c] = pairs[c].astype(str)
    tracks['track_id'] = tracks.track_id.astype(str)
    audio = np.load(root/'audio_features.npz',allow_pickle=False)
    if len(audio['mel']) != len(ids):
        raise ValueError("Feature count mismatch")
    return tracks,pairs, audio

class FeatureBank:
    """Training-only normalization and vocabulary, reusable on held-out tracks."""
    def __init__(self, tracks, audio, config, state=None):
        self.tracks=tracks.reset_index(drop=True)
        self.ids = self.tracks.track_id.astype(str).tolist()
        self.idx={x:i for i,x in enumerate(self.ids)}
        self.mel=audio['mel'].astype('float32')
        self.rhythm=audio['rhythm'].astype('float32')
        train = np.array(self.tracks.split.eq('train'))
        if not train.any():
            raise ValueError('No train tracks')
        tags=self.tracks.get('tag',pd.Series(['']*len(self.ids))).fillna('').astype(str)
        if state is None:
            counts={}
            for raw in tags[train]:
                for tag in raw.split('|'):
                    if tag.strip(): counts[tag.strip()]=counts.get(tag.strip(),0)+1
            vocab=[t for t,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:int(config['max_tags'])]]
        else: vocab=state['tag_vocab']
        self.tags=np.zeros((len(self.ids), max(1,len(vocab))),np.float32)
        for i,raw in enumerate(tags):
            for tag in raw.split('|'):
                if tag.strip() in vocab: self.tags[i,vocab.index(tag.strip())]=1.0
        def col(key):
            return pd.to_numeric(self.tracks[key],errors='coerce').to_numpy(dtype=np.float32) if key in self.tracks else np.full(len(self.ids),np.nan,np.float32)
        emo = np.stack([col('valence'), col('arousal')],axis=1)
        self.emotion=np.concatenate([np.nan_to_num(emo,nan=0),np.isfinite(emo).astype(np.float32)],axis=1)
        meta_cols=list(config.get('metadata_columns',[]))
        m=np.stack([col(k) for k in meta_cols],axis=1) if meta_cols else np.zeros((len(self.ids),0),np.float32)
        rawrh=self.rhythm.copy()
        if state is None:
            self.rh_mean=rawrh[train].mean(0)
            self.rh_std=rawrh[train].std(0)+1e-4
            self.meta_mean=np.nanmean(m[train],axis=0) if m.shape[1] else np.zeros(0)
            self.meta_mean=np.nan_to_num(self.meta_mean,nan=0)
            self.meta_std=np.nanstd(m[train],axis=0)+1e-4 if m.shape[1] else np.ones(0)
            self.meta_std=np.nan_to_num(self.meta_std,nan=1.0)
        else:
            self.rh_mean=np.asarray(state['rh_mean'],np.float32)
            self.rh_std=np.asarray(state['rh_std'],np.float32)
            self.meta_mean=np.asarray(state['meta_mean'],np.float32)
            self.meta_std=np.asarray(state['meta_std'],np.float32)
        self.normrh=(self.rhythm-self.rh_mean)/self.rh_std
        self.meta=np.concatenate([
            np.nan_to_num((m-self.meta_mean)/self.meta_std,nan=0), np.isfinite(m).astype(np.float32)],axis=1)
        self.state={'tag_vocab':vocab,'rh_mean':self.rh_mean.tolist(),'rh_std':self.rh_std.tolist(),
                    'meta_mean':self.meta_mean.tolist(),'meta_std':self.meta_std.tolist()}
    def tensors(self, ids, device):
        import torch
        ii=np.asarray([self.idx[str(x)] for x in ids],dtype=int)
        return tuple(torch.as_tensor(a[ii],dtype=torch.float32,device=device)
                     for a in (self.mel,self.normrh,self.tags,self.emotion,self.meta))
