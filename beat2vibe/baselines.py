import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from . import CLASS_TO_ID
from .data import load_prepared
from .metrics import classification, ranking, aggregate_rank


def evaluate_external(data_dir,embedding_file,prefix,config):
    tracks,pairs,_=load_prepared(data_dir,config)
    obj=np.load(embedding_file,allow_pickle=False)
    ids=[str(x) for x in obj['track_id']]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate external embedding IDs')
    emb=obj['embedding'].astype('float32')
    if emb.ndim!=2 or len(ids)!=len(emb): raise ValueError('Embedding array must be [tracks, dimensions]')
    emap={x:emb[i] for i,x in enumerate(ids)}
    if not set(tracks.track_id.astype(str)).issubset(emap):
        raise ValueError('Some track IDs do not have external embeddings')
    # Reuse the same pair data and exactly the same track partition as the proposed model.
    def make_pair(subset):
        xi=np.asarray([emap[x] for x in subset.current_id]);xj=np.asarray([emap[x] for x in subset.candidate_id])
        x=np.concatenate([xi,xj,xj-xi,np.abs(xj-xi)],axis=1)
        return x,np.asarray([CLASS_TO_ID[x] for x in subset.transition])
    train=pairs[pairs.split=='train'];test=pairs[pairs.split=='test']
    if train.empty or test.empty: raise ValueError('Train and test pairs required')
    xtrain,ytrain=make_pair(train);xtest,ytest=make_pair(test)
    ss=StandardScaler().fit(xtrain)
    clf=MLPClassifier(hidden_layer_sizes=(128,),max_iter=200,early_stopping=False,
        random_state=int(config['seed']),learning_rate_init=float(config['learning_rate']))
    clf.fit(ss.transform(xtrain),ytrain)
    predicted=clf.predict(ss.transform(xtest))
    cls=classification(ytest,predicted)
    rows=[{'current_id':i,'candidate_id':j,'label':y,'predicted':int(p)} for i,j,y,p in
        zip(test.current_id,test.candidate_id,ytest,predicted)]
    # Cosmetic note: if no explicit listener/session labels, unjudged negatives are a proxy only.
    pool=tracks.loc[tracks.split=='test','track_id'].astype(str).tolist()
    known={}
    for r in test.itertuples(index=False):known.setdefault((r.current_id,r.transition),set()).add(r.candidate_id)
    rng=np.random.default_rng(int(config['seed'])+1327)
    all_metrics=[];detail=[];size=int(config['candidate_size'])
    for r in test.itertuples(index=False):
        choices=[t for t in pool if t!=r.current_id and t not in known[(r.current_id,r.transition)]]
        if len(choices)<size-1: continue
        cands=[r.candidate_id]+rng.choice(choices,size-1,replace=False).tolist()
        cs=cosine_similarity(emap[r.current_id][None,:],np.stack([emap[j] for j in cands]))[0]
        order=np.argsort(-cs,kind='stable'); relevance=np.zeros(size,dtype=int);relevance[0]=1
        all_metrics.append(ranking(relevance[order]))
        qid=f'{r.current_id}|{r.candidate_id}|{r.transition}'
        for rank,k in enumerate(order,start=1):
            detail.append({'query_id':qid,'rank':rank,'relevant':int(k==0),
                'candidate_id':cands[k],'score':float(cs[k]),
                'relevance_source':'annotated_positive' if k==0 else 'unjudged_assumed_negative'})
    result={'classification':cls,'ranking':({'candidate_size':size,'queries':len(all_metrics),
            'sampling_protocol':'one annotated positive + unjudged sampled alternatives',
            **aggregate_rank(all_metrics)} if all_metrics else {})}
    p=Path(prefix);p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(str(p)+'_classification.csv',index=False)
    pd.DataFrame(detail).to_csv(str(p)+'_ranking.csv',index=False)
    Path(str(p)+'_metrics.json').write_text(json.dumps(result,indent=2))
    return result


def extract_mert(tracks_csv,output,model_id='m-a-p/MERT-v1-95M',device='cpu'):
    """
    MERT-v1-95M expects 24 kHz input, model supplies custom code.
    """
    import librosa
    import torch
    from transformers import AutoModel, AutoFeatureExtractor
    df=pd.read_csv(tracks_csv)
    extractor=AutoFeatureExtractor.from_pretrained(model_id,trust_remote_code=True)
    model=AutoModel.from_pretrained(model_id,trust_remote_code=True).to(device).eval()
    ids=[];emb=[];root=Path(tracks_csv).resolve().parent
    for row in df.itertuples(index=False):
        p=Path(row.audio_path)
        if not p.is_absolute():p=root/p
        y,_=librosa.load(p,sr=24000,mono=True,duration=5.0)
        inputs=extractor(y,sampling_rate=24000,return_tensors='pt')
        with torch.inference_mode():
            states=model(**{k:v.to(device) for k,v in inputs.items()}).last_hidden_state
            z=states.mean(dim=1).cpu().numpy()[0]
        ids.append(str(row.track_id));emb.append(z)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,track_id=np.asarray(ids),embedding=np.asarray(emb,dtype=np.float32))
    return output
