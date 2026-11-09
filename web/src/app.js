'use strict';
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const names={normal:'Normal concert dismissal',concert:'Concert emergency',blocked:'Main exit blocked',lecture:'Lecture emergency',barriers:'Concert + barriers',led:'Concert + LEDs',combo:'Concert + barriers + LEDs',lecture_led:'Lecture + LEDs',lecture_barriers:'Lecture + barriers'};
const fmt=(n,d=1)=>Number(n).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const svg=(tag,attrs={},text)=>{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;return e;};
const txt=(x,y,s,extra={})=>svg('text',{x,y,fill:'#526970','font-size':12,...extra},s);
let selected='M2',frame=0,playing=false,last=0,elapsed=0;
const current=()=>RESULTS.replays[$('scenario').value];
for(const [key,label] of Object.entries(names)){const e=document.createElement('option');e.value=key;e.textContent=label;$('scenario').append(e);}
$('scenario').value='concert';
function polygon(id){const g=RESULTS.geometry;
 if(g.sectors[id]){const [r0,r1,a,b]=g.sectors[id],points=[];
  for(let k=0;k<=24;k++){const z=(a+(b-a)*k/24)*Math.PI/180;points.push([-r1*Math.cos(z),r1*Math.sin(z)]);}
  for(let k=24;k>=0;k--){const z=(a+(b-a)*k/24)*Math.PI/180;points.push([-r0*Math.cos(z),r0*Math.sin(z)]);}return points;}
 if(g.rects[id]){const [x,y,w,h]=g.rects[id];return [[x,-y],[x+w,-y],[x+w,-y-h],[x,-y-h]];}
 return (g.polys[id]||[]).map(([x,y])=>[x,-y]);}
function color(n){const z=Math.min(1,n/160);return `rgb(${Math.round(226-220*z)},${Math.round(238-125*z)},${Math.round(233-125*z)})`;}
function drawMap(){const r=current(),map=$('map'),floor=Number($('floor').value),metric=$('metric').value,byId=Object.fromEntries(r.nodes.map((n,i)=>[n.id,i]));map.replaceChildren();map.setAttribute('viewBox',floor?' -43 -9 86 47':'-43 -9 86 61');
 map.append(svg('rect',{x:-17.5,y:-6,width:35,height:6,fill:'#e4e8e0',rx:1}),txt(0,-2.2,'STAGE',{'font-size':1.6,'text-anchor':'middle'}));
 for(const id of RESULTS.geometry.floors[floor]){const base=id.split('@')[0],i=byId[base],points=polygon(id);if(!points.length)continue;const n=i===undefined?0:r[metric][frame][i];
  const isBlocked=$('scenario').value==='blocked'&&base==='EXIT-Main';
  const p=svg('path',{d:'M'+points.map(a=>a.join(',')).join('L')+'Z',fill:i===undefined?'#e6e8e6':isBlocked?'#f4ccbf':color(n),stroke:base===selected?'#3156ac':'#87a69c','stroke-width':base===selected?.65:.18,tabindex:0,role:'button','aria-label':`${base}, ${n} people`});
  p.append(svg('title',{},`${r.nodes[i]?.label||base}: ${n} people`));p.onclick=()=>{selected=base;draw();};p.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selected=base;draw();}};map.append(p);
  const x=points.reduce((s,p)=>s+p[0],0)/points.length,y=points.reduce((s,p)=>s+p[1],0)/points.length;
  let label=base.replace('EXIT-','').replace('Stage-','Stg-');
  map.append(txt(x,y,label,{'font-size':base.startsWith('EXIT')?1.1:1.4,'text-anchor':'middle',fill:n>90?'white':'#254a4d','pointer-events':'none'}));
  if((base.match(/^[OMB][123]$/)||base==='FOY')&&i!==undefined)map.append(txt(x,y+2.2,String(n),{'font-size':1.7,'text-anchor':'middle',fill:n>90?'white':'#254a4d','pointer-events':'none'}));
  if(base==='M2')map.append(svg('path',{d:'M'+points.map(a=>a.join(',')).join('L')+'Z',fill:'none',stroke:'#587bc9','stroke-width':.55,'pointer-events':'none'}));
  if(!['normal','lecture','lecture_led','lecture_barriers'].includes($('scenario').value)&&RESULTS.manager_nodes.includes(base))map.append(svg('circle',{cx:x+2.2,cy:y-1.7,r:.7,fill:'#eda33d',stroke:'#6f501c','stroke-width':.12}));
 }
 const i=byId[selected],n=r.nodes[i];$('detail').textContent=n?`${n.id} · ${n.label}. ${r.occupancy[frame][i]} people ${n.kind==='exit'?'have reached this exit':'currently here'}; ${r.queue[frame][i]} waiting to enter. ${n.outside?'This area is outside the building.':''}`:'This area is closed or absent in the selected situation.';
}
function drawCurve(){const r=current(),c=$('curve');c.replaceChildren();const x=t=>40+350*t/r.time.at(-1),y=n=>175-145*n/1329;
 for(const n of [0,500,1000,1329]){c.append(svg('line',{x1:40,x2:390,y1:y(n),y2:y(n),stroke:'#dbe6df'}),txt(33,y(n)+4,n,{'text-anchor':'end','font-size':10}));}
 c.append(svg('path',{d:r.time.map((t,i)=>(i?'L':'M')+x(t)+','+y(r.inside[i])).join(' '),fill:'none',stroke:'#087d7b','stroke-width':2.5}));
 if(180<=r.time.at(-1))c.append(svg('line',{x1:x(180),x2:x(180),y1:20,y2:175,stroke:'#b67d40','stroke-dasharray':'4 4'}),txt(x(180),14,'180 s',{'text-anchor':'middle','font-size':10}));
 c.append(svg('circle',{cx:x(r.time[frame]),cy:y(r.inside[frame]),r:4,fill:'#efac45'}),txt(40,196,'0 s'),txt(390,196,fmt(r.time.at(-1))+' s',{'text-anchor':'end'}));
}
function draw(){const r=current();frame=Math.min(frame,r.time.length-1);const t=r.time[frame];$('time').textContent=fmt(t)+' s';$('inside').textContent=fmt(r.inside[frame],0);$('left').textContent=fmt(1329-r.inside[frame],0);$('finish').textContent=fmt(r.T)+' s';$('timeline').value=frame;$('slider-time').textContent=fmt(t)+' s';drawMap();drawCurve();}
function load(){playing=false;$('play').textContent='Play';frame=0;elapsed=0;const r=current();$('timeline').max=r.time.length-1;
 const key=$('scenario').value;const reaction=key==='normal'?'truncated Gaussian':key.startsWith('lecture')?'Gamma':'lognormal';
 $('scenario-note').textContent=`Reaction times: ${reaction}. Wheelchair users start in M2. ${key==='blocked'?'The five main doors are blocked; information is broadcast after 20 s. ':''}In this run, wheelchair users reach an outdoor refuge by ${fmt(r.wheelchair.T_refuge)} s and street level with assistance by ${fmt(r.wheelchair.T_street)} s.`;draw();}
$('scenario').onchange=load;$('floor').onchange=draw;$('metric').onchange=draw;
$('timeline').oninput=()=>{frame=+$('timeline').value;elapsed=current().time[frame];draw();};
$('play').onclick=()=>{if(frame===current().time.length-1){frame=0;elapsed=0;}playing=!playing;$('play').textContent=playing?'Pause':'Play';last=performance.now();};$('reset').onclick=load;
function animate(now){if(playing){const r=current();elapsed+=(now-last)/1000*Number($('speed').value);while(frame<r.time.length-1&&r.time[frame+1]<=elapsed)frame++;if(frame===r.time.length-1){playing=false;$('play').textContent='Play';}draw();}last=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);
function table(id,headers,rows){const t=$(id);const h=document.createElement('thead'),hr=document.createElement('tr');headers.forEach(x=>{const e=document.createElement('th');e.scope='col';e.textContent=x;hr.append(e);});h.append(hr);t.append(h);const body=document.createElement('tbody');rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(x=>{const td=document.createElement('td');td.textContent=x;tr.append(td);});body.append(tr);});t.append(body);}
table('results-table',['Situation','Runs','Mean (s)','95th percentile (s)','Under 180 s'],RESULTS.main.map(r=>[names[r.case],r.K,fmt(r.T_mean,2),fmt(r.T95,2),'0 / '+r.K]));
table('guidance-table',['Comparison','Difference (s)','95% interval (s)'],RESULTS.paired.map(r=>[names[r.design],fmt(r.restricted_delta_mean,2),`[${fmt(r.ci_low,2)}, ${fmt(r.ci_high,2)}]`]));
table('cost-table',['Design','Annual cost','Mean (s)'],RESULTS.cost.filter(r=>r.pareto).map(r=>[r.managers+' managers'+(r.led?' + LEDs':'')+(r.barriers?' + barriers':''),'$'+fmt(r.cost_total,2),fmt(r.T_mc,2)]));
{const s=$('bars'),left=270,scale=750/500;for(const n of [0,100,200,300,400,500]){s.append(svg('line',{x1:left+n*scale,x2:left+n*scale,y1:15,y2:327,stroke:'#e0e9e3'}),txt(left+n*scale,345,n+' s',{'text-anchor':'middle'}));}
 RESULTS.main.forEach((r,i)=>{const y=22+i*34;s.append(txt(left-12,y+14,names[r.case],{'text-anchor':'end','font-size':13}),svg('rect',{x:left,y,width:r.T_mean*scale,height:19,rx:3,fill:'#138b80'}),svg('line',{x1:left+r.T_mean*scale,x2:left+r.T95*scale,y1:y+9.5,y2:y+9.5,stroke:'#527b75','stroke-width':2}),svg('circle',{cx:left+r.T95*scale,cy:y+9.5,r:3,fill:'#527b75'}),txt(left+r.T_mean*scale-7,y+13,fmt(r.T_mean),{fill:'white','text-anchor':'end','font-size':11}));});
 s.append(svg('line',{x1:left+180*scale,x2:left+180*scale,y1:10,y2:327,stroke:'#b67d40','stroke-dasharray':'4 4'}),txt(270,368,'Bars: mean time   ·   Dots: 95th percentile   ·   Dashed line: 180 s target',{'font-size':12}));}
load();
