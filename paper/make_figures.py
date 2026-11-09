from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source_data'
for folder in ('figures','previews'):
 (ROOT/folder).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#87929c','xtick.color':'#34434d','ytick.color':'#34434d','axes.labelcolor':'#24343e','text.color':'#24343e','pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
BLUE='#247496'; ORANGE='#d17a27'; GREEN='#3b8877'; PURPLE='#8261a3'; GRAY='#72818d'
colors={'normal':GRAY,'concert':BLUE,'lecture':GREEN,'blocked':ORANGE}
names={'normal':'Normal dismissal','concert':'Concert emergency','lecture':'Lecture emergency','blocked':'Main exit blocked'}
cases=['normal','concert','lecture','blocked']
def read(name):
 p=SOURCE/(name+'.csv'); return pd.read_csv(p)
def save(fig,name):
 fig.savefig(ROOT/'figures'/(name+'.pdf'),bbox_inches='tight',pad_inches=.12)
 fig.savefig(ROOT/'previews'/(name+'.png'),bbox_inches='tight',pad_inches=.12,dpi=240)
 plt.close(fig)
def grid(ax): ax.grid(axis='x',color='#e5e9ec',linewidth=.7); ax.set_axisbelow(True)
main=read('main_replicates')
# Distributions, keeping T90 separate from completed-run proportions.
fig,ax=plt.subplots(figsize=(7.1,3.5),layout='constrained')
for i,c in enumerate(cases):
 for metric,off,color in [('T90',-.16,BLUE),('T',.16,ORANGE)]:
  vals=main.loc[main.case==c,metric]
  bp=ax.boxplot(vals,positions=[i+off],vert=False,widths=.23,whis=(5,95),showfliers=False,patch_artist=True,manage_ticks=False,medianprops={'color':color,'linewidth':1.8},boxprops={'facecolor':color,'alpha':.2,'edgecolor':color},whiskerprops={'color':color},capprops={'color':color})
  ax.plot(vals.mean(),i+off,'o',color=color,ms=4)
ax.axvline(180,color=GRAY,ls='--',lw=1)
ax.set_yticks(range(4),[names[c] for c in cases]); ax.invert_yaxis(); ax.set_xlabel('Time since announcement or start of dismissal (s)'); ax.set_xlim(100,510); grid(ax)
ax.legend(handles=[Patch(facecolor=BLUE,alpha=.5,label=r'$T_{90}$: 90% have left'),Patch(facecolor=ORANGE,alpha=.5,label=r'$T$: everyone has left'),Line2D([0],[0],color=GRAY,ls='--',label='180 s target')],loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3,fontsize=8.5)
save(fig,'bailey_main_times')
# Blocked-route announcement.
d=read('broadcast_delays_summary').sort_values('case')
fig,ax=plt.subplots(figsize=(6.8,3.1),layout='constrained')
y=d.T_mean.to_numpy(); lo=d.restricted_mean_ci_low.to_numpy(); hi=d.restricted_mean_ci_high.to_numpy()
ax.errorbar(d.case,y,yerr=[y-lo,hi-y],fmt='o-',color=BLUE,capsize=3,lw=1.5,ms=5)
ax.set_xlabel('Time of blocked-route announcement (s)'); ax.set_ylabel('Mean building evacuation time (s)'); ax.set_xticks(d.case); ax.set_ylim(408,442); ax.grid(color='#e5e9ec'); ax.set_axisbelow(True)
save(fig,'bailey_broadcast_delay')
# Peak queue diagnostics: four most crowded links in each deterministic run.
fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),layout='constrained',sharex=True)
for ax,c in zip(axs.flat,cases):
 d=read('queue_'+c).sort_values('peak_queue',ascending=False).head(4).iloc[::-1]
 labels=d.label.str.replace('E side stair','East side stairs',regex=False).str.replace('W side stair','West side stairs',regex=False)
 ax.barh(range(4),d.peak_queue,color=colors[c],height=.6)
 ax.set_yticks(range(4),labels,fontsize=9); ax.set_title(names[c],loc='left',fontweight='bold'); ax.set_xlim(0,140); grid(ax)
 for i,x in enumerate(d.peak_queue): ax.text(x+2,i,f'{x:.0f}',va='center',fontsize=9)
for ax in axs[1]: ax.set_xlabel('Peak queue (people)')
save(fig,'bailey_peak_queues')
# Paired effect confidence intervals.
d=read('main_paired')
labels=['Concert: barriers','Concert: LEDs','Concert: LEDs + barriers','Lecture: LEDs','Lecture: barriers']
fig,ax=plt.subplots(figsize=(7.1,3.2),layout='constrained')
y=d.restricted_delta_mean.to_numpy(); lo=d.ci_low.to_numpy(); hi=d.ci_high.to_numpy()
ax.errorbar(y,np.arange(len(d)),xerr=[y-lo,hi-y],fmt='o',capsize=4,color=BLUE,lw=1.8,ms=5)
ax.axvline(0,color=GRAY,ls='--'); ax.set_yticks(range(len(d)),labels); ax.invert_yaxis(); grid(ax)
ax.set_xlabel(r'Paired change in evacuation time, $\Delta T$ (s)'); ax.set_xlim(-3.3,1.1)
save(fig,'bailey_guidance_effects')
# Wheelchair last-arrival outcomes, not T_ground.
fig,ax=plt.subplots(figsize=(7.1,3.3),layout='constrained')
for i,c in enumerate(cases):
 d=main[main.case==c]; a=d.T_refuge.mean(); b=d.T_street.mean()
 ax.plot([a,b],[i,i],color='#b8c2ca',lw=3)
 ax.plot(a,i,'o',color=BLUE,ms=7);ax.plot(b,i,'s',color=ORANGE,ms=6)
 ax.annotate(f'{a:.1f}',(a,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
 ax.annotate(f'{b:.1f}',(b,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
ax.set_yticks(range(4),[names[c] for c in cases]); ax.set_ylim(3.6,-.65);ax.set_xlim(160,605);grid(ax)
ax.set_xlabel('Mean time for all three wheelchair users to arrive (s)')
ax.legend(handles=[Line2D([0],[0],marker='o',color=BLUE,ls='',label='Outdoor waiting area'),Line2D([0],[0],marker='s',color=ORANGE,ls='',label='Street level')],loc='upper center',bbox_to_anchor=(.5,-.21),ncol=2,fontsize=9)
save(fig,'bailey_wheelchair_times')
# Largest six parameter effects in each emergency scenario.
d=read('parameter_tornado_no_wheelchairs')
labelmap={'prep':'Reaction-time scale','len':'Travel length','v':'Walking speed','q':'Passage flow','radius_m':'Manager radius','p_follow':'Following share','beta_queue':'Queue avoidance','beta_dist':'Distance preference','beta_habit':'Main-door preference','w_side_stair':'Side-stair width','w_stage_stair':'Stage-stair width','w_main_exterior':'Main-door width','w_side_door':'Side-door width','stair_factor':'Stair speed','beta_guide':'Guidance weight','w_foyer_door':'Foyer-door width','w_stage_door_mezz':'Stage-route door width'}
fig,axs=plt.subplots(1,2,figsize=(8.2,3.9),layout='constrained',sharex=True)
for ax,c in zip(axs,['concert','lecture']):
 sub=d[d.scenario==c].copy(); sub['delta']=sub['T']-sub.T_base
 keys=sub.groupby('key').delta.apply(lambda z:abs(z).max()).nlargest(6).index
 for i,k in enumerate(keys):
  z=sub[sub.key==k].sort_values('value')
  a,b=z.iloc[0],z.iloc[-1]
  ax.hlines(i,min(a.delta,b.delta),max(a.delta,b.delta),color='#b9c4cb',lw=3)
  ax.plot(a.delta,i,'o',color=BLUE,ms=5); ax.plot(b.delta,i,'s',color=ORANGE,ms=5)

 ax.axvline(0,color=GRAY,lw=1,ls='--');ax.set_yticks(range(6),[labelmap[k] for k in keys],fontsize=9);ax.invert_yaxis();ax.set_xlim(-150,265);ax.set_title(names[c],loc='left',fontweight='bold');grid(ax);ax.set_xlabel('Change from baseline time (s)')
fig.legend(handles=[Line2D([0],[0],marker='o',color=BLUE,ls='',label='Lower parameter value'),Line2D([0],[0],marker='s',color=ORANGE,ls='',label='Higher parameter value')],loc='outside lower center',ncol=2,fontsize=9)
save(fig,'bailey_parameter_sensitivity')
# Step comparisons: show mean changes directly; not confidence intervals on paired effect.
det=read('time_step_check'); mc=read('numerical_resolution_summary')
fig,axs=plt.subplots(1,2,figsize=(7.5,3.2),layout='constrained')
for i,c in enumerate(['concert','lecture','blocked']):
 v=det[det.case==c].set_index('dt')['T']; a=v.loc[.2]-v.loc[.1];b=v.loc[.05]-v.loc[.1]
 axs[0].plot(a,i-.12,'s',color=ORANGE,ms=6);axs[0].plot(b,i+.12,'o',color=BLUE,ms=6)
 m=mc[mc.case.str.startswith(c+'|')].set_index('case').T_mean
 delta=m.loc[c+'|dt0.05']-m.loc[c+'|dt0.1']
 axs[1].barh(i,delta,color=BLUE,height=.4)
 axs[1].text(delta+(.08 if delta>=0 else -.08),i,f'{delta:+.2f}',ha='left' if delta>=0 else 'right',va='center',fontsize=9)
for ax in axs:
 ax.axvline(0,color=GRAY,lw=1); ax.set_yticks(range(3),[names[c] for c in ['concert','lecture','blocked']],fontsize=9);ax.set_ylim(2.6,-.6);grid(ax);ax.set_xlabel('Change from the 0.1 s result (s)')
axs[0].set_xlim(-9,9);axs[1].set_xlim(-.7,1.5)
axs[0].set_title('(a) Deterministic results',loc='left',fontweight='bold');axs[1].set_title('(b) Monte Carlo means',loc='left',fontweight='bold')
axs[0].legend(handles=[Line2D([0],[0],color=ORANGE,marker='s',ls='',label='0.2 s step'),Line2D([0],[0],color=BLUE,marker='o',ls='',label='0.05 s step')],loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,fontsize=8)
axs[1].text(.03,.04,'0.05 s step; 30 runs per setting',transform=axs[1].transAxes,fontsize=8.5)
save(fig,'bailey_time_step')
print('Created 7 PDF figures and 7 PNG previews.')
