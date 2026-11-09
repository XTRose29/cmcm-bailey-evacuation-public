"""Build the self-contained results explorer; no network is required to view it."""
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'model'))
from bailey.floorplan import SECTORS, RECTS, POLYS, FLOOR1, FLOOR2
def build():
    data=json.loads((ROOT/'web/data/results.json').read_text())
    data['geometry']={'sectors':SECTORS,'rects':RECTS,
                      'polys':{k:v.tolist() for k,v in POLYS.items()},
                      'floors':[FLOOR1,FLOOR2]}
    html=(ROOT/'web/src/page.template.html').read_text()
    for marker,value in [('/*RESULTS_DATA*/','const RESULTS = '+json.dumps(data,separators=(',',':'))+';'),
                         ('/*APP*/',(ROOT/'web/src/app.js').read_text())]:
        assert html.count(marker)==1
        html=html.replace(marker,value)
    return html
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    text=build();out=ROOT/'web/index.html'
    if a.check:
        assert out.read_text()==text,'Rebuild web/index.html'
        print('HTML matches source and experiment data.')
    else:
        out.write_text(text);print('Built',out)
