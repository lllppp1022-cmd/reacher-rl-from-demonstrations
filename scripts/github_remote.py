"""Private-only GitHub repository operations using Git Credential Manager in memory.

Never prints credentials or writes them to a file, command argument or remote URL.
User must first complete Git Credential Manager's official browser/device login.
"""
from pathlib import Path
import os,sys,subprocess,json,urllib.request,urllib.error,base64,csv,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.runtime import read,atomic_json
OWNER='lllppp1022-cmd';NAME='reacher-rl-from-demonstrations';FULL=f'{OWNER}/{NAME}';URL=f'https://github.com/{FULL}'
STATE=ROOT/'reports/github_upload_runtime.json'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').strip()
def credential():
    env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
    result=subprocess.run(['git','credential','fill'],input=f'protocol=https\nhost=github.com\nusername={OWNER}\n\n',text=True,capture_output=True,env=env,cwd=ROOT,timeout=20)
    if result.returncode:raise RuntimeError('No usable local GitHub credential. Complete official Git Credential Manager login; never paste a token into chat.')
    values=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if not values.get('password'):raise RuntimeError('Git Credential Manager returned no usable credential.')
    return values['password']
def api(token,path,method='GET',data=None,allow404=False):
    request=urllib.request.Request('https://api.github.com'+path,data=None if data is None else json.dumps(data).encode(),method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'reacher-private-portfolio-uploader','X-GitHub-Api-Version':'2026-03-10'})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code==404 and allow404:return None
        raise RuntimeError(f'GitHub API {method} {path} returned HTTP {e.code}; credentials were not logged.') from None
    except urllib.error.URLError:raise RuntimeError('GitHub network request failed; no credential content was logged.') from None
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['check','create','verify']);a=p.parse_args()
    token=credential();user=api(token,'/user');assert user['login'].lower()==OWNER.lower(),'Authenticated account differs from authorized owner'
    repo=api(token,f'/repos/{FULL}',allow404=True)
    if a.mode=='check':print(json.dumps(dict(authenticated=True,login=user['login'],repository_exists=repo is not None,private=repo['private'] if repo else None)));return
    if a.mode=='create':
        created=False
        if repo:
            assert repo['private'],'Existing repository is not Private; do not change visibility or overwrite it'
            refs=api(token,f'/repos/{FULL}/git/matching-refs/heads/')
            assert not refs,'Same-name repository already has branch history; no overwrite attempted'
        else:
            repo=api(token,'/user/repos','POST',dict(name=NAME,description='Reproducible MuJoCo Reacher project featuring imitation learning, DAgger, PPO online fine-tuning, behavior anchoring, and real training rollout visualization.',private=True,auto_init=False,has_wiki=False,has_projects=False));created=True
        assert repo['private'] and repo['full_name'].lower()==FULL.lower()
        state=dict(account=user['login'],url=repo['html_url'],repository_id=repo['id'],private=True,created_this_run=created,pushed=False,remote_verified=False)
        atomic_json(STATE,state);print(json.dumps(state));return
    assert repo and repo['private'],'Remote absent or no longer Private'
    head=git('rev-parse','HEAD');ref=api(token,f'/repos/{FULL}/git/ref/heads/main');assert ref['object']['sha']==head
    ls=git('ls-remote','origin','refs/heads/main').split()[0];assert ls==head
    if repo['default_branch']!='main':
        repo=api(token,f'/repos/{FULL}','PATCH',dict(default_branch='main'))
    assert repo['default_branch']=='main'
    tree=api(token,f'/repos/{FULL}/git/trees/{head}?recursive=1');assert not tree['truncated']
    remote={r['path']:r['sha'] for r in tree['tree'] if r['type']=='blob'}
    raw=subprocess.check_output(['git','ls-tree','-r','-z','HEAD'],cwd=ROOT).decode('utf-8');local={}
    for row in raw.split('\0'):
        if row:
            info,name=row.split('\t',1);local[name]=info.split()[2]
    assert remote==local,'Remote Git blobs differ from local HEAD'
    for path in ['README.md','README_CN.md','scripts/portfolio.py','assets/figures/robot_control.png','assets/figures/pipeline.png','assets/figures/final_comparison.png']:assert path in remote
    for name in ['README.md','README_CN.md']:
        doc=api(token,f'/repos/{FULL}/contents/{name}?ref=main');text=base64.b64decode(doc['content']).decode('utf-8')
        for number in ['58.67%','2.3029','32.33%','2.7527','64.67%','1.9986']:assert number in text
    assert not any(Path(p).suffix.lower() in ['.zip','.pt','.npz','.gif','.mp4'] for p in remote)
    origin=git('remote','get-url','origin');assert origin==URL+'.git'
    state=dict(account=user['login'],url=repo['html_url'],repository_id=repo['id'],private=True,default_branch='main',head=head,remote_head=ls,uploaded_files=len(remote),remote_tree_matches_local=True,readmes_and_images_verified=True,excluded_binaries_absent=True,pushed=True,remote_verified=True,origin=origin)
    atomic_json(STATE,state);print(json.dumps(state))
if __name__=='__main__':
    try:main()
    except Exception as e:
        print('UPLOAD CHECK STOPPED:',str(e),file=sys.stderr);sys.exit(1)
