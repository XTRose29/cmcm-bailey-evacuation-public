"""Update costs without rerunning or modifying evacuation simulations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
import pandas as pd
from bailey.config import DEFAULT, COST_RANGES
from bailey.cost import design_table, cost_sensitivity
rows=pd.read_csv(ROOT/'results/formal/tables/cost_pareto.csv')[['managers','led','barriers','T_mc']].to_dict('records')
out=ROOT/'results/costs'
out.mkdir(exist_ok=True)
design_table(rows,DEFAULT,obj='T_mc').to_csv(out/'cost_recomputed_latest.csv',index=False)
for name,df in cost_sensitivity(rows,DEFAULT,ranges=COST_RANGES,n_draws=2000,seed=2026).items():
    df.to_csv(out/f'cost_recomputed_{name}.csv',index=False)
print('Updated cost tables using the published cost assumptions and unchanged simulation times.')
