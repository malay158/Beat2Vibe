
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .metrics import paired_test

def compare(file_a,file_b,out):
    a=pd.read_csv(file_a);b=pd.read_csv(file_b)
    def by_query(frame):
        # Exactly one annotated positive per query is required.
        values={}
        for key,g in frame.groupby('query_id',sort=False):
            g=g.sort_values('rank')
            ranks=g.loc[g.relevant.eq(1),'rank'].to_numpy()
            if len(ranks)!=1: raise ValueError(f'Expected 1 positive for query {key}')
            rank=int(ranks[0]); values[key]={'rr':1.0/rank,'ndcg10':1.0/np.log2(rank+1) if rank<=10 else 0.0}
        return values
    a=by_query(a);b=by_query(b)
    keys=sorted(set(a)&set(b))
    if set(a)!=set(b):
        raise ValueError('Compared files must contain identical query_id sets and candidate protocols')
    results={}
    for key in ['rr','ndcg10']:
        va=[a[q][key] for q in keys];vb=[b[q][key] for q in keys]
        results[key]=paired_test(va,vb)
        rng=np.random.default_rng(42)
        differences=np.asarray(va)-np.asarray(vb)
        boot=np.array([differences[rng.integers(0,len(keys),len(keys))].mean() for _ in range(2000)])
        results[key]['bootstrap_95_ci']=np.quantile(boot,[0.025,0.975]).tolist()
    results['notes']='Paired Wilcoxon on identical queries; unjudged distractor relevance is assumed zero.'
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(results,indent=2))
    return results

def controller_plot(ranking_csv,out):
    
    import matplotlib.pyplot as plt
    df=pd.read_csv(ranking_csv)
    required=['transition','weight_0','weight_1','weight_2']
    if not set(required).issubset(df.columns):raise ValueError('Adaptive policy weight columns not found')
    means=df.groupby('transition')[required[1:]].mean()
    means=means.reindex([x for x in ['sustain','uplift','soften','shift','drift'] if x in means.index])
    ax=means.plot.bar(figsize=(8,4),rot=0)
    ax.legend(['Transition cue','Similarity','Rhythm'])
    ax.set_ylabel('Average learned controller weight')
    ax.set_xlabel('Annotated transition class')
    ax.figure.tight_layout();Path(out).parent.mkdir(parents=True,exist_ok=True)
    ax.figure.savefig(out,dpi=220);plt.close(ax.figure)
    return out
