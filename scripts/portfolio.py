"""Public command line; algorithms live in src/verified with provenance hashes."""
from pathlib import Path
import sys,argparse,json,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import ROOT,registry,model,configuration,initialize,atomic_json,read,sha
def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('init',help='Validate provisioned models and create portable runtime configuration')
    q=sub.add_parser('import-assets',help='Local provisioning; does not grant distribution rights');q.add_argument('--archive-root',type=Path,required=True)
    q=sub.add_parser('watch');q.add_argument('--policy',choices=['ikpd','bc','control','best','compare'],default='best');q.add_argument('--seed',type=int,default=7059);q.add_argument('--training-seed',type=int,default=42);q.add_argument('--max-steps',type=int,default=250);q.add_argument('--fps',type=float,default=50);q.add_argument('--repeat',action='store_true')
    q=sub.add_parser('evaluate');q.add_argument('--method',choices=['ikpd']+sorted(set(r['method'] for r in registry())),default='long_decay_strong');q.add_argument('--training-seed',type=int,default=42);q.add_argument('--first-seed',type=int,default=16000);q.add_argument('--episodes',type=int,default=100);q.add_argument('--output',type=Path,default=Path('results/repeat_verification'))
    q=sub.add_parser('train');q.add_argument('--output',type=Path,required=True);q.add_argument('--group',choices=['b0','weak','medium','strong'],default='b0');q.add_argument('--steps',type=int,default=4096);q.add_argument('--training-seed',type=int,default=42);q.add_argument('--render-training',action='store_true');q.add_argument('--record-training-rollouts',action='store_true');q.add_argument('--record-every-n-episodes',type=int,default=4);q.add_argument('--max-recorded-episodes',type=int,default=4);q.add_argument('--render-fps',type=float,default=50);q.add_argument('--disable-render-after',type=int);q.add_argument('--skip-evaluation-replays',action='store_true')
    q=sub.add_parser('replay');q.add_argument('--run',required=True);q.add_argument('--mode',choices=['actual','compare','progress'],default='actual');q.add_argument('--episode',type=int,default=1);q.add_argument('--update',type=int,default=1);q.add_argument('--fps',type=float,default=50);q.add_argument('--max-frames',type=int)
    q=sub.add_parser('pipeline');q.add_argument('--profile',choices=['smoke','full'],default='smoke');q.add_argument('--output',type=Path,required=True);q.add_argument('--ppo-method',choices=['scratch','init','anchor'],default='anchor');q.add_argument('--group',choices=['b0','medium','strong'],default='strong');q.add_argument('--schedule',choices=['fixed','linear'],default='fixed')
    sub.add_parser('results')
    q=sub.add_parser('record-video');q.add_argument('--policy',choices=['ikpd','bc','control','best'],default='best');q.add_argument('--seed',type=int,default=7059);q.add_argument('--output',type=Path,default=Path('results/new_diagnostic_video'))
    a=p.parse_args()
    if a.command=='results':print((ROOT/'reports/final_results.csv').read_text(encoding='utf-8-sig'));return
    if a.command=='import-assets':
        for r in registry():
            source=a.archive_root/r['source_relative'];assert sha(source)==r['sha256'];dest=ROOT/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        ref=a.archive_root/'reacher_anchor_calibration/configs/diagnostic_observations.npz';shutil.copy2(ref,ROOT/'configs'/ref.name)
        initialize();print('Provisioned verified local assets; publication permission remains pending.');return
    if a.command=='init':initialize();print('15 SHA256-verified model assets initialized');return
    if a.command=='replay':
        from watch_training_rollout import run
        a.speed=1.;run(a);return
    if a.command=='pipeline':
        from src.pipeline import run
        run(a);return
    # Expert-only diagnostics require the protocol, not undistributed learned weights.
    expert_only=(a.command=='evaluate' and a.method=='ikpd') or (a.command=='record-video' and a.policy=='ikpd')
    c=read(ROOT/'configs/anchor_template.json') if expert_only else configuration()
    if a.command=='train':
        from record_training_rollouts import parser,run
        args=['--config',str(ROOT/'runtime/experiment.json'),'--output',str(a.output.resolve()),'--group',a.group,'--steps',str(a.steps),'--training-seed',str(a.training_seed),'--record-every-n-episodes',str(a.record_every_n_episodes),'--max-recorded-episodes',str(a.max_recorded_episodes),'--render-fps',str(a.render_fps)]
        for key in ['render_training','record_training_rollouts','skip_evaluation_replays']:
            if getattr(a,key):args+=['--'+key.replace('_','-')]
        if a.disable_render_after:args+=['--disable-render-after',str(a.disable_render_after)]
        run(parser().parse_args(args));return
    if a.command=='watch':
        from watch_anchor_calibration import parser,run
        args=['--policy',{'best':'anchor'}.get(a.policy,a.policy),'--phase','long_decay','--group','strong','--step','100352','--seed',str(a.seed),'--training-seed',str(a.training_seed),'--max-steps',str(a.max_steps),'--fps',str(a.fps)]
        if a.repeat:args+=['--repeat']
        run(c,parser().parse_args(args));return
    if a.command=='evaluate':
        from evaluate_anchor import evaluate
        c=dict(c,output_root=str(a.output.resolve()));policy=None if a.method=='ikpd' else model(a.method,a.training_seed)
        result=evaluate(c,f'{a.method}_seed{a.training_seed}',policy,range(a.first_seed,a.first_seed+a.episodes),split='repeat_verification',expert=a.method=='ikpd')
        from src.figures import evaluation_plot
        evaluation_plot(a.output.resolve()/f'results/repeat_verification/{a.method}_seed{a.training_seed}')
        print(json.dumps(result,ensure_ascii=False));return
    if a.command=='record-video':
        from record_reacher_video import record
        import record_reacher_video as recorder
        from evaluate_anchor import load_policy
        recorder.load_bc=load_policy;recorder.setup=lambda c:None
        dest=a.output.resolve();(dest/'videos').mkdir(parents=True,exist_ok=True);c=dict(c,output_root=str(dest))
        mapping={'bc':'BC-DAgger','control':'long_control_b0','best':'long_decay_strong'}
        result=record(c,'ikpd' if a.policy=='ikpd' else 'bc_ppo',None if a.policy=='ikpd' else model(mapping[a.policy]),seed=a.seed,suffix='repeat_diagnostic')
        atomic_json(dest/'recording.json',dict(result,purpose='repeat diagnostic, not new independent test'))
if __name__=='__main__':main()
