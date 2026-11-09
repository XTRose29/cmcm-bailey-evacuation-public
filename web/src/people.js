// Rendering adapter: timing and population transfers are supplied by the Python model.
const ReplayPeople = (() => {
 const polar=ClassicLayout.polar;
 const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
 function pathPoints(g,ids){const out=[];for(let i=0;i<ids.length;i++){const a=g.nodes[ids[i]];if(!i){out.push(a);continue;}const b=g.nodes[ids[i-1]],ra=distance(a,{x:520,y:110}),rb=distance(b,{x:520,y:110});
  if(a.floor===b.floor&&Math.abs(ra-rb)<.1){const aa=Math.atan2(a.y-110,520-a.x),ab=Math.atan2(b.y-110,520-b.x);for(let j=1;j<=8;j++)out.push({...polar(ra,ab+(aa-ab)*j/8),floor:a.floor});}else out.push(a);
 }return out;}
 function shortest(g,start,end,from,to){
  if(start===end)return [start];const d=new Float64Array(g.nodes.length).fill(Infinity),prev=new Int32Array(g.nodes.length).fill(-1),seen=new Set(),heap=[[0,start]];d[start]=0;
  const corridor=/^C\d+$/.test(from)&&/^C\d+$/.test(to),seating=/^[OMB]/.test(from)&&/^[AMD]/.test(to);
  while(true){if(!heap.length)throw Error(`No display path: ${from} → ${to}`);heap.sort((a,b)=>b[0]-a[0]);const [best,u]=heap.pop();if(seen.has(u))continue;if(u===end)break;seen.add(u);
   for(const eid of g.nodes[u].adj){const e=g.edges[eid],v=e.a===u?e.b:e.a,n=g.nodes[v];if(n.kind==='seat'&&g.nodes[u].kind!=='seat')continue;
    if(corridor&&e.kind!=='corridor')continue;
    if(seating&&['corridor','door','sideStair','mainDoor','mainStair','stair','stageAccess','portico'].includes(e.kind))continue;
    if(e.kind==='mainStair')continue;
    if(e.kind==='stageAccess'&&!/ST/.test(from+to))continue;
    if(e.kind==='stair'&&!/BS|ST/.test(from+to))continue;
    if(e.kind==='porticoDetour'&&!/PORS/.test(from+to))continue;
    const nd=best+Math.max(1,distance(g.nodes[u],n));if(nd<d[v]){d[v]=nd;prev[v]=u;heap.push([nd,v]);}
   }
  }
  const ids=[end];while(ids[0]!==start)ids.unshift(prev[ids[0]]);return ids;
 }
 class Replay {
  constructor(r,key){this.r=r;this.key=key;this.layout=new ClassicLayout.Layout();const g=this.layout,find=label=>g.nodes.find(n=>n.label===label).id;
   this.anchors={FOY:g.foyer,'EXIT-Main':find('Exterior landing'),POR1:find('West portico'),POR2:find('East portico'),'EXIT-Side-W':find('West exterior stairs'),'EXIT-Side-E':find('East exterior stairs'),'EXIT-Stage-W':find('West stage exit'),'EXIT-Stage-E':find('East stage exit')};
   for(let k=1;k<=6;k++)this.anchors['A'+k]=find('A'+k);
   for(let k=1;k<=9;k++)this.anchors['D'+k]=find('D'+k);
   [0,4,8,12,15,17,20,24,28,32].forEach((n,i)=>this.anchors['C'+(i+1)]=g.corr[n]);
   [5,16,27].forEach((n,i)=>this.anchors['M'+(i+1)]=g.cross[n]);
   for(let k=1;k<=3;k++)this.anchors['F'+k]=find('F'+k);
   ['L1','L2','R1','R2'].forEach((s,i)=>this.anchors['S'+(i+1)]=find(s));
   this.anchors.BS1=find('West balcony stairs');this.anchors.BS2=find('East balcony stairs');this.anchors.ST1=find('West stage stairs');this.anchors.ST2=find('East stage stairs');
   const stageDown=g.nodes.filter(n=>n.label==='Stage stair landing').sort((a,b)=>a.x-b.x);this.stageDown=stageDown;
   g.link(g.cross[0],stageDown[0].id,'stageAccess');g.link(g.cross[32],stageDown[1].id,'stageAccess');
   const south=g.node(520,680,0,'portico','South portico'),sw=g.node(92,680,0,'portico'),se=g.node(948,680,0,'portico');this.anchors.PORS=south;
   g.link(this.anchors.POR1,sw,'porticoDetour');g.link(sw,south,'porticoDetour');g.link(south,se,'porticoDetour');g.link(se,this.anchors.POR2,'porticoDetour');
   const groups={O:'orchestra',M:'mezzanine',B:'balcony'},assigned={};
   for(const [prefix,group] of Object.entries(groups)){const seats=g.seats.filter(s=>s.group===group).sort((a,b)=>a.angle-b.angle||a.row-b.row);let cursor=0;
    for(let k=1;k<=3;k++){const id=prefix+k,ni=r.nodes.findIndex(n=>n.id===id),count=r.person_tracks.filter(t=>t[1]===ni).length;assigned[id]=seats.slice(cursor,cursor+count);cursor+=count;}
   }
   const used={},cache=new Map();this.people=r.person_tracks.map((track,id)=>{
    const origin=r.nodes[track[1]].id,seat=assigned[origin][used[origin]||0];used[origin]=(used[origin]||0)+1;
    const anchors=[seat.node];for(let j=2;j<track.length;j+=2){const n=r.nodes[track[j+1]].id,prev=r.nodes[track[j-1]].id;
     if(/^B[123]$/.test(n)){const a=[5,16,27][+n[1]-1];const p=polar(321,Math.PI*a/32);anchors.push(g.nodes.filter(v=>v.floor===1&&v.kind==='corridor').sort((a,b)=>distance(a,p)-distance(b,p))[0].id);}
     else if(/^ST/.test(n)&&/^M/.test(prev))anchors.push(stageDown[+n[2]-1].id);
     else {if(this.anchors[n]===undefined)throw Error('Unmapped cell '+n);anchors.push(this.anchors[n]);}
    }
    const segments=[];for(let j=0;j<anchors.length;j++){
     const node=r.nodes[track[j*2+1]],next=j+1<anchors.length?r.nodes[track[j*2+3]]:node,start=track[j*2]*RESULTS.dt,end=j+1<anchors.length?track[j*2+2]*RESULTS.dt:Infinity;
     const cacheKey=[anchors[j],anchors[j+1],node.id,next.id,id%5].join(':');let points=cache.get(cacheKey);
     if(!points){let ids=j+1<anchors.length?shortest(g,anchors[j],anchors[j+1],node.id,next.id):[anchors[j]];
      if(node.id==='FOY'&&next.id==='EXIT-Main'){const door=find('E'+(id%5+1));ids=[g.foyer,door,this.anchors['EXIT-Main']];}
      points=pathPoints(g,ids);cache.set(cacheKey,points);
     }
     let length=0;const lengths=[0];for(let k=1;k<points.length;k++){length+=Math.max(1,distance(points[k-1],points[k]));lengths.push(length);}
     segments.push({node,start,end,points,length,lengths,initial:j===0});
    }
    return {id,origin,group:groups[origin[0]],seat,segments};
   });
   this.duration=r.time.at(-1);this.time=0;this.seek(0);
  }
  locate(a,t){let lo=0,hi=a.segments.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(a.segments[m].start<=t+1e-8)lo=m;else hi=m-1;}return lo;}
  position(a,t=this.time){const k=this.locate(a,t),s=a.segments[k],d=s.end-s.start;
   let q=Number.isFinite(d)?Math.min(1,Math.max(0,(t-s.start)/d)):0;
   if(s.initial){const travel=Math.min(d*.8,s.length/15);q=Math.min(1,Math.max(0,(t-(s.end-travel))/Math.max(.01,travel)));}
   const target=q*s.length;let i=1;while(i<s.lengths.length&&s.lengths[i]<target)i++;
   const u=s.points[Math.max(0,i-1)],v=s.points[Math.min(i,s.points.length-1)],f=i<s.lengths.length?(target-s.lengths[i-1])/(s.lengths[i]-s.lengths[i-1]||1):0;
   const p=u.floor===v.floor?{x:u.x+(v.x-u.x)*f,y:u.y+(v.y-u.y)*f,floor:u.floor}:{...(f<.5?u:v)};
   if(!s.initial){const spread=s.node.kind==='seat'?3:2;p.x+=((a.id*7)%5-2)*spread;p.y+=((a.id*11)%7-3)*spread;}
   return {...p,node:s.node,done:s.node.kind==='exit',opacity:s.node.kind==='exit'?Math.max(0,1-(t-s.start)/.7):1,segment:k};
  }
  seek(t){this.time=Math.min(this.duration,Math.max(0,t));this.counts=new Int32Array(this.r.nodes.length);this.positions=this.people.map(a=>{const p=this.position(a);this.counts[this.r.nodes.indexOf(p.node)]++;return p;});this.inside=this.r.nodes.reduce((s,n,i)=>s+(n.kind!=='exit'&&!n.outside?this.counts[i]:0),0);}
  remainingRoute(id){const a=this.people[id],k=this.locate(a,this.time);return a.segments.slice(k).flatMap(s=>s.points);}
 }
 return {Replay,pathPoints};
})();
