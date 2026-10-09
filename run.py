"""Beat2Vibe command line entrypoint. Run `python run.py --help`."""
import argparse
import json
from pathlib import Path
import yaml


def cli():
    p=argparse.ArgumentParser(description='Reproducible Beat2Vibe research pipeline')
    p.add_argument('--config',default='config.yaml')
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--tracks',required=True);prep.add_argument('--pairs',required=True);prep.add_argument('--out',default='data/processed')
    pre=sub.add_parser('pretrain');pre.add_argument('--data',default='data/processed');pre.add_argument('--out',default='results/vibe_pretrained.pt');pre.add_argument('--device',default='auto')
    tr=sub.add_parser('train');tr.add_argument('--data',default='data/processed');tr.add_argument('--ssl',default='results/vibe_pretrained.pt');tr.add_argument('--out',default='results/beat2vibe.pt');tr.add_argument('--device',default='auto');tr.add_argument('--epochs',type=int)
    tr.add_argument('--variant',choices=['full','baseline','fixed','no_vibe','no_beat','audio_only','audio_rhythm'],default='full')
    ev=sub.add_parser('evaluate');ev.add_argument('--data',default='data/processed');ev.add_argument('--checkpoint',required=True);ev.add_argument('--out',default='results/test');ev.add_argument('--split',choices=['train','val','test'],default='test');ev.add_argument('--device',default='auto')
    comp=sub.add_parser('compare');comp.add_argument('--a',required=True);comp.add_argument('--b',required=True);comp.add_argument('--out',default='results/significance.json')
    fig=sub.add_parser('plot-weights');fig.add_argument('--ranking-csv',required=True);fig.add_argument('--out',default='results/controller_weights.png')
    ext=sub.add_parser('external-baseline');ext.add_argument('--data',default='data/processed');ext.add_argument('--features',required=True);ext.add_argument('--out',default='results/mert_baseline')
    mert=sub.add_parser('extract-mert');mert.add_argument('--tracks',required=True);mert.add_argument('--out',required=True);mert.add_argument('--device',default='cpu');mert.add_argument('--allow-remote-code',action='store_true')
    tune=sub.add_parser('tune');tune.add_argument('--data',default='data/processed');tune.add_argument('--ssl',default='results/vibe_pretrained.pt');tune.add_argument('--out',default='results/tuning.json');tune.add_argument('--device',default='auto')
    args=p.parse_args()
    conf=yaml.safe_load(Path(args.config).read_text())
    if args.command=='prepare':
        from beat2vibe.data import prepare
        prepare(args.tracks,args.pairs,args.out,conf)
    elif args.command=='pretrain':
        from beat2vibe.experiment import pretrain
        pretrain(args.data,args.out,conf,args.device)
    elif args.command=='train':
        from beat2vibe.experiment import train
        train(args.data,args.ssl,args.out,conf,args.variant,args.device,args.epochs)
    elif args.command=='evaluate':
        from beat2vibe.experiment import evaluate
        print(json.dumps(evaluate(args.data,args.checkpoint,args.out,args.split,args.device),indent=2))
    elif args.command=='compare':
        from beat2vibe.analysis import compare
        print(json.dumps(compare(args.a,args.b,args.out),indent=2))
    elif args.command=='plot-weights':
        from beat2vibe.analysis import controller_plot
        print(controller_plot(args.ranking_csv,args.out))
    elif args.command=='external-baseline':
        from beat2vibe.baselines import evaluate_external
        print(json.dumps(evaluate_external(args.data,args.features,args.out,conf),indent=2))
    elif args.command=='extract-mert':
        if not args.allow_remote_code:
            p.error('MERT uses custom model code; re-run with --allow-remote-code only after reviewing it')
        from beat2vibe.baselines import extract_mert
        print(extract_mert(args.tracks,args.out,device=args.device))
    elif args.command=='tune':
        from beat2vibe.experiment import train
        from pathlib import Path
        vals=[]
        for lr in (0.001,0.0003):
            for margin in (0.1,0.2):
                trial=dict(conf,learning_rate=lr,ranking_margin=margin)
                path=Path(args.out).parent/f'tune_lr{lr}_margin{margin}.pt'
                train(args.data,args.ssl,path,trial,'full',args.device)
                import torch
                ck=torch.load(path,map_location='cpu',weights_only=True)
                vals.append({'learning_rate':lr,'ranking_margin':margin,'val_macro_f1':ck['val_macro_f1']})
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        Path(args.out).write_text(json.dumps({'trials':vals,'best':max(vals,key=lambda x:x['val_macro_f1'])},indent=2))
        print(Path(args.out).read_text())

if __name__=='__main__':cli()
