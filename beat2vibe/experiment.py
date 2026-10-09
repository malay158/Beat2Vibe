
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.cluster import KMeans
from . import CLASS_TO_ID, CLASSES
from .data import load_prepared, FeatureBank
from .models import Beat2Vibe,PairwiseBaseline,VibeLearner,augment_spectrogram,contrastive_loss
from .metrics import classification, ranking, aggregate_rank


def device_for(choice='auto'):
    return torch.device('cuda' if choice=='auto' and torch.cuda.is_available() else
                        'cpu' if choice=='auto' else choice)

def set_seed(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def tensors(bank, ids, device):
    feat=bank.tensors(ids,device)
    raw=torch.as_tensor(bank.rhythm[[bank.idx[str(x)] for x in ids]],device=device)
    return feat,raw

def pretrain(data_dir, output, config, device='auto'):
    set_seed(config['seed']);device=device_for(device)
    tracks,_,audio=load_prepared(data_dir,config)
    ids=np.flatnonzero(tracks.split.eq('train'))
    if len(ids)<2: raise ValueError('Self-supervision needs >=2 training tracks')
    x=torch.from_numpy(audio['mel'][ids]).to(device)
    model=VibeLearner().to(device); optimizer=optim.AdamW(model.parameters(),lr=config['ssl_learning_rate'])
    history=[]
    for epoch in range(int(config['ssl_epochs'])):
        model.train();per=[]
        for batch in torch.randperm(len(x),device=device).split(int(config['ssl_batch_size'])):
            if len(batch)<2: continue
            z1=model(augment_spectrogram(x[batch]))
            z2=model(augment_spectrogram(x[batch]))
            loss=contrastive_loss(z1,z2,float(config['ssl_temperature']))
            optimizer.zero_grad();loss.backward();optimizer.step()
            per.append(float(loss.item()))
        if not per: raise ValueError('SSL batch size must be >=2')
        history.append(float(np.mean(per)))
        print(f'SSL epoch {epoch+1}: loss={history[-1]:.5f}')
    model.eval()
    with torch.no_grad():
        embeddings=np.concatenate([model(x[p]).cpu().numpy() for p in torch.arange(len(x),device=device).split(128)],axis=0)
    k=min(int(config['vibe_clusters']),len(embeddings))
    # If a tiny test fixture has too few tracks, use its available count; paper runs retain configured K.
    km=KMeans(n_clusters=k,n_init=10,random_state=config['seed']).fit(embeddings)
    centers=F.normalize(torch.as_tensor(km.cluster_centers_,dtype=torch.float32),dim=-1)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'prototypes':centers,'ssl_losses':history,
                'config':dict(config),'n_train_tracks':int(len(ids))},output)
    return output


def compute_rhythm_stats(bank,pairs):
    observed=pairs[pairs.split=='train']
    if observed.empty: raise ValueError('No training pairs after track-disjoint split')
    changes=[];classes=[]
    for row in observed.itertuples(index=False):
        a=bank.rhythm[bank.idx[row.current_id]];b=bank.rhythm[bank.idx[row.candidate_id]]
        changes.append([(b[0]-a[0])/max(float(a[0]),1.0),b[2]-a[2],b[3]-a[3],b[4]-a[4],b[5]-a[5]])
        classes.append(CLASS_TO_ID[row.transition])
    changes=np.asarray(changes,dtype=np.float32)
    scale=changes.std(0)+1e-4
    
    scale=np.maximum(scale,0.01)
    norm=changes/scale
    mu=np.stack([norm[np.asarray(classes)==k].mean(0) if k in classes else np.zeros(5)
                 for k in range(5)]).astype(np.float32)
    return torch.tensor(mu),torch.tensor(scale)


def train(data_dir,ssl_checkpoint,output,config,variant='full',device='auto',epochs=None):
    set_seed(config['seed']);device=device_for(device)
    tracks,pairs,audio=load_prepared(data_dir,config)
    bank=FeatureBank(tracks,audio,config)
    baseline=variant=='baseline'
    if baseline:
        model=PairwiseBaseline().to(device)
    else:
        if not ssl_checkpoint or not Path(ssl_checkpoint).exists():
            raise FileNotFoundError('Pretrained SSL checkpoint is required for Beat2Vibe')
        pre=torch.load(ssl_checkpoint,map_location='cpu',weights_only=True)
        model=Beat2Vibe(tag_dim=bank.tags.shape[1],meta_dim=bank.meta.shape[1],
            n_prototypes=len(pre['prototypes']),variant=variant).to(device)
        model.acoustic.load_state_dict({k.split('acoustic.',1)[1]:v for k,v in pre['model'].items() if k.startswith('acoustic.')})
        model.vibe_project.load_state_dict({k.split('project.',1)[1]:v for k,v in pre['model'].items() if k.startswith('project.')})
        model.prototypes.copy_(pre['prototypes'].to(device))
        model.freeze_ssl()
        mu,scale=compute_rhythm_stats(bank,pairs)
        model.rhythm_mu.copy_(mu.to(device));model.rhythm_scale.copy_(scale.to(device))
    training=pairs[pairs.split=='train'].reset_index(drop=True)
    validation=pairs[pairs.split=='val'].reset_index(drop=True)
    optimizer=optim.AdamW([x for x in model.parameters() if x.requires_grad],
        lr=float(config['learning_rate']),weight_decay=float(config['weight_decay']))
    by_anchor=defaultdict(list)
    for row in training.itertuples(index=False):
        by_anchor[row.current_id].append((row.candidate_id,CLASS_TO_ID[row.transition]))
    examples=[]
    for row in training.itertuples(index=False):
        y=CLASS_TO_ID[row.transition]
        possible=[j for j,label in by_anchor[row.current_id] if label!=y]
        examples.append((row.current_id,row.candidate_id,y,possible))
    best_metric=-float('inf');best_state=None
    n_epochs=int(epochs if epochs is not None else config['train_epochs'])
    for epoch in range(n_epochs):
        model.train();rand=np.random.default_rng(int(config['seed'])+epoch)
        order=rand.permutation(len(examples));losses=[];rank_count=0
        for chunk in np.array_split(order,max(1,int(np.ceil(len(order)/int(config['batch_size']))))):
            ex=[examples[int(i)] for i in chunk]
            ids_i=[x[0] for x in ex];ids_j=[x[1] for x in ex]
            f_i,r_i=tensors(bank,ids_i,device);f_j,r_j=tensors(bank,ids_j,device)
            xi=model.track(f_i);xj=model.track(f_j)
            out=model.pair(xi,xj,r_i,r_j,f_i[3],f_j[3])
            y=torch.tensor([x[2] for x in ex],device=device)
            cl=F.cross_entropy(out['logits'],y)
            rankloss=cl.new_zeros(())
            if not baseline:
                valid=[k for k,x in enumerate(ex) if x[3]]
                if valid:
                    negatives=[str(rand.choice(ex[k][3])) for k in valid]
                    f_n,r_n=tensors(bank,negatives,device)
                    xneg=model.track(f_n)
                    pairneg=model.pair(xi[valid],xneg,r_i[valid],r_n,f_i[3][valid],f_n[3])
                    ypos=y[valid]
                    positive=out['scores'][valid].gather(1,ypos[:,None]).squeeze(1)
                    negative=pairneg['scores'].gather(1,ypos[:,None]).squeeze(1)
                    rankloss=F.relu(float(config['ranking_margin'])-positive+negative).mean()
                    rank_count+=len(valid)
            loss=cl+float(config['ranking_beta'])*rankloss
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(float(loss.detach().cpu()))
        monitor=evaluate_classification(model,bank,validation,device)
        score=monitor['macro_f1'] if monitor else -float(np.mean(losses))
        if score>best_metric:
            best_metric=score
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'Train epoch {epoch+1}: loss={np.mean(losses):.5f} val_macro_f1={score:.4f} rank_triplets={rank_count}')
    if not baseline and rank_count==0:
        print('WARNING: No same-anchor contrasting labels available; ranking loss was inactive.')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':best_state,'variant':variant,'baseline':baseline,'tag_dim':bank.tags.shape[1],
        'meta_dim':bank.meta.shape[1],'n_prototypes':0 if baseline else len(model.prototypes),
        'feature_state':bank.state,'config':dict(config),'val_macro_f1':best_metric,
        'ranking_triplets_last_epoch':rank_count},output)
    return output


def restore(data_dir,checkpoint,device='auto'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    tracks,pairs,audio=load_prepared(data_dir,ck['config'])
    bank=FeatureBank(tracks,audio,ck['config'],state=ck['feature_state'])
    if ck['baseline']: model=PairwiseBaseline()
    else: model=Beat2Vibe(ck['tag_dim'],ck['meta_dim'],ck['n_prototypes'],ck['variant'])
    model.load_state_dict(ck['model']);model.to(device_for(device)).eval()
    return model,bank,pairs,ck

@torch.no_grad()
def evaluate_classification(model,bank,subset,device,return_rows=False):
    model.eval()
    predicted=[];actual=[];rows=[]
    subset=subset.reset_index(drop=True)
    for offset in range(0,len(subset),64):
        batch=subset.iloc[offset:offset+64]
        fi,ri=tensors(bank,batch.current_id.tolist(),device)
        fj,rj=tensors(bank,batch.candidate_id.tolist(),device)
        out=model.pair(model.track(fi),model.track(fj),ri,rj,fi[3],fj[3])
        pred=out['logits'].argmax(-1).cpu().numpy().tolist()
        y=[CLASS_TO_ID[z] for z in batch.transition]
        predicted.extend(pred);actual.extend(y)
        for i,(a,b,c,t,p) in enumerate(zip(batch.current_id,batch.candidate_id,y,batch.transition,pred)):
            rows.append({'current_id':a,'candidate_id':b,'label':t,'label_id':c,'predicted':CLASSES[p],
                'predicted_id':p,'confidence':float(out['prob'][i,p].cpu())})
    if not actual: return ({},[]) if return_rows else {}
    result=classification(actual,predicted)
    return (result,rows) if return_rows else result

@torch.no_grad()
def evaluate_ranking(model,bank,pairs,split,config,device,model_name='Beat2Vibe'):
    
    if isinstance(model,PairwiseBaseline):
        raise ValueError('Pairwise classifier has no trained ranking policy; use full/fixed Beat2Vibe')
    subset=pairs[pairs.split==split]
    trackpool=[x for x in bank.ids if bank.tracks.loc[bank.idx[x],'split']==split]
    rng=np.random.default_rng(int(config['seed'])+1327)
    n=int(config['candidate_size'])
    rows=[];query_metrics=[]
    positives=defaultdict(set)
    for row in subset.itertuples(index=False): positives[(row.current_id,row.transition)].add(row.candidate_id)
    for row in subset.itertuples(index=False):
        # Use only one known positive and sample unlabeled alternatives explicitly as such.
        pool=[t for t in trackpool if t!=row.current_id and t not in positives[(row.current_id,row.transition)]]
        if len(pool)<n-1: continue
        distractors=rng.choice(pool,n-1,replace=False).tolist()
        candidates=[row.candidate_id]+distractors
        ids_i=[row.current_id]*n
        fi,ri=tensors(bank,ids_i,device);fj,rj=tensors(bank,candidates,device)
        out=model.pair(model.track(fi),model.track(fj),ri,rj,fi[3],fj[3])
        target=CLASS_TO_ID[row.transition]
        scores=out['scores'][:,target]
        
        policy=F.softmax(scores/float(config['policy_temperature']),dim=0)
        order=torch.argsort(policy,descending=True).cpu().numpy().tolist()
        relevant=np.zeros(n,dtype=int); relevant[0]=1
        query=ranking(relevant[order]);query_metrics.append(query)
        qid=f'{row.current_id}|{row.candidate_id}|{row.transition}'
        for position,k in enumerate(order):
            rows.append({'query_id':qid,'model':model_name,'current_id':row.current_id,
                'candidate_id':candidates[k],'transition':row.transition,'rank':position+1,
                'score':float(scores[k].cpu()),'policy':float(policy[k].cpu()),
                'relevant':int(relevant[k]),'relevance_source':'annotated_positive' if k==0 else 'unjudged_assumed_negative',
                **{f'weight_{j}':float(out['weights'][k,j].cpu()) for j in range(3)}})
    if not query_metrics: raise ValueError('No ranking queries possible; candidate pool too small')
    return {'sampling_protocol':'one annotated positive + unjudged sampled alternatives',
            'candidate_size':n,'queries':len(query_metrics),**aggregate_rank(query_metrics)},rows


def evaluate(data_dir,checkpoint,output_prefix,split='test',device='auto'):
    import pandas as pd
    model,bank,pairs,ck=restore(data_dir,checkpoint,device)
    d=device_for(device)
    m,rows=evaluate_classification(model,bank,pairs[pairs.split==split],d,return_rows=True)
    if not m: raise ValueError(f'No annotated {split} pairs')
    path=Path(output_prefix);path.parent.mkdir(parents=True,exist_ok=True)
    (path.parent/(path.name+'_classification.json')).write_text(json.dumps(m,indent=2))
    pd.DataFrame(rows).to_csv(path.parent/(path.name+'_classification.csv'),index=False)
    result={'classification':m}
    if not ck['baseline']:
        rank,detail=evaluate_ranking(model,bank,pairs,split,ck['config'],d,ck['variant'])
        (path.parent/(path.name+'_ranking.json')).write_text(json.dumps(rank,indent=2))
        pd.DataFrame(detail).to_csv(path.parent/(path.name+'_ranking.csv'),index=False)
        result['ranking']=rank
    return result
