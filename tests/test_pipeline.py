import json
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from beat2vibe.data import split_track_disjoint,read_manifest,rhythm_delta,FeatureBank
from beat2vibe.models import Beat2Vibe,PairwiseBaseline,contrastive_loss
from beat2vibe.metrics import ranking,paired_test


def test_disjoint_track_splits():
    tracks=pd.DataFrame({'track_id':[str(i) for i in range(30)]})
    pairs=pd.DataFrame({'current_id':['0','0','1','15','20'],
                        'candidate_id':['1','20','2','16','21'],
                        'transition':['sustain']*5})
    mapping,annot=split_track_disjoint(tracks,pairs,seed=7)
    assert len(mapping)==30
    for r in annot.itertuples():
        if r.split!='cross_split_dropped':
            assert mapping[r.current_id]==mapping[r.candidate_id]==r.split


def test_delta_and_ranking():
    r1=[100,10,0.6,0.1,1.,0.2];r2=[120,12,0.5,0.2,1.5,0.3]
    assert np.isclose(rhythm_delta(r1,r2)[0],0.2)
    got=ranking([0,1,0,0,0],ks=(1,3))
    assert got['recall@1']==0 and got['recall@3']==1
    assert np.isclose(got['mrr'],0.5)
    assert paired_test([1,0.8,0.9],[0.5,0.4,0.2])['p_value']<=1


def test_models_shapes_probabilities():
    torch.set_num_threads(2)
    rng=np.random.default_rng(5);n=3
    data=(torch.tensor(rng.random((n,128,128)),dtype=torch.float32),
          torch.tensor(rng.random((n,6)),dtype=torch.float32),
          torch.zeros((n,2)),torch.tensor([[0.4,0.5,1,1]]*n),torch.zeros((n,0)))
    model=Beat2Vibe(tag_dim=2,meta_dim=0,n_prototypes=4).eval()
    with torch.no_grad():
        x=model.track(data)
        rhythm_raw=torch.tensor([[100.,10.,.5,.1,1.,.2]]*n)
        result=model.pair(x,x,rhythm_raw,rhythm_raw,data[3],data[3])
        assert result['scores'].shape==(n,5)
        assert result['weights'].shape==(n,3)
        assert torch.allclose(result['weights'].sum(1),torch.ones(n),atol=1e-5)
        assert torch.isfinite(result['scores']).all()
        base=PairwiseBaseline().eval()
        xb=base.track(data)
        assert base.pair(xb,xb,rhythm_raw,rhythm_raw,data[3],data[3])['logits'].shape==(n,5)
    z=torch.nn.functional.normalize(torch.randn(4,16),dim=-1)
    assert torch.isfinite(contrastive_loss(z,z)).item()


def test_missing_data_labels_rejected(tmp_path):
    tracks=tmp_path/'t.csv';pairs=tmp_path/'p.csv'
    pd.DataFrame({'track_id':['a','b'],'audio_path':['a.wav','b.wav']}).to_csv(tracks,index=False)
    pd.DataFrame({'current_id':['a'],'candidate_id':['b'],'transition':['unknown']}).to_csv(pairs,index=False)
    with pytest.raises(ValueError,match='Unknown'):
        read_manifest(tracks,pairs)
