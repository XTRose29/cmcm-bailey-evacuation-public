"""Synchronize the built HTML to the existing personal-site URL using GitHub CLI."""
from pathlib import Path
import argparse,base64,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
REPO='XTRose29/xtrose29.github.io'
BRANCH='master'
TARGET='bailey-evacuation/index.html'
URL='https://xtrose29.github.io/bailey-evacuation/index.html'
def api(args,payload=None):
    result=subprocess.run(['gh','api',*args],input=None if payload is None else json.dumps(payload),text=True,capture_output=True,check=True)
    return json.loads(result.stdout)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publish',action='store_true');a=p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'web/scripts/build.py'),'--check'],check=True)
    data=(ROOT/'web/index.html').read_bytes();endpoint=f'repos/{REPO}/contents/{TARGET}'
    current=api([endpoint+'?ref='+BRANCH])
    local_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if current['sha']==local_sha:print('Already synchronized:',URL);return
    print('Target:',REPO,BRANCH,TARGET)
    if not a.publish:print('Dry run. Add --publish to update only the visualization HTML.');return
    r=api(['--method','PUT',endpoint,'--input','-'],{'message':'Publish final Bailey Hall model results explorer','branch':BRANCH,'sha':current['sha'],'content':base64.b64encode(data).decode()})
    print('Published commit:',r['commit']['sha']);print(URL)
if __name__=='__main__':main()
