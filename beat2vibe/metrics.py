import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from scipy.stats import wilcoxon
from . import CLASSES

def classification(y_true,y_pred):
    if not len(y_true): raise ValueError('No test predictions')
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    cm=confusion_matrix(y_true,y_pred,labels=range(len(CLASSES)))
    return {'accuracy':float(accuracy_score(y_true,y_pred)),
            'macro_precision':float(p),'macro_recall':float(r),'macro_f1':float(f),
            'confusion_matrix':cm.tolist(),'class_names':list(CLASSES)}

def ranking(relevance_sorted,ks=(5,10)):
    """Relevance is 0/1 for a fixed sampled candidate set."""
    y=np.asarray(relevance_sorted,dtype=int)
    if not len(y) or not y.any(): raise ValueError('Ranking query has no known positive')
    out={}
    for k in ks:
        out[f'recall@{k}']=float(y[:k].sum()/y.sum())
        ranks=np.arange(1,len(y[:k])+1)
        dcg=np.sum(y[:k]/np.log2(ranks+1))
        ideal=np.sum(np.sort(y)[::-1][:k]/np.log2(ranks+1))
        out[f'ndcg@{k}']=float(dcg/ideal if ideal>0 else 0)
    out['mrr']=float(1/(1+np.flatnonzero(y)[0]))
    return out

def aggregate_rank(rows):
    if not rows: raise ValueError('No evaluated ranking queries')
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def paired_test(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    if a.shape != b.shape or len(a)<2: raise ValueError('Need same aligned queries (>=2)')
    if np.allclose(a,b): return {'n':len(a),'mean_delta':0.0,'p_value':1.0}
    result=wilcoxon(a,b,zero_method='wilcox',alternative='two-sided',method='auto')
    return {'n':int(len(a)),'mean_delta':float((a-b).mean()),'p_value':float(result.pvalue)}
