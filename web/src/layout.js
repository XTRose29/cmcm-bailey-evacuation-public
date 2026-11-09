// Schematic drawing geometry only. All movement counts and timing come from Python.
const ClassicLayout = (() => {
const PX=12, polar=(r,a)=>({x:520-r*Math.cos(a),y:110+r*Math.sin(a)});
class Layout {
 constructor(){this.nodes=[];this.edges=[];this.seats=[];this.exits=[];this.build();}
 node(x,y,floor,kind='aisle',label=''){const n={id:this.nodes.length,x,y,floor,kind,label,adj:[],waiting:[]};this.nodes.push(n);return n.id;}
 pnode(r,a,floor,kind='aisle',label=''){const p=polar(r,a);return this.node(p.x,p.y,floor,kind,label);}
 link(a,b,kind='aisle',width=1.2,length){const u=this.nodes[a],v=this.nodes[b];const e={id:this.edges.length,a,b,kind,width,length:length||Math.hypot(u.x-v.x,u.y-v.y)/PX,channels:Math.max(1,Math.floor(width/.6)),lanes:Array.from({length:2*Math.max(1,Math.floor(width/.6))},()=>[]),credit:[0,0],entered:[0,0]};this.edges.push(e);u.adj.push(e.id);v.adj.push(e.id);return e.id;}
 ring(r,angles,floor,kind='aisle',label=''){const nodes=[];for(let i=0;i<angles.length;i++){nodes.push(this.pnode(r,angles[i],floor,kind,label?label+(i+1):''));if(i)this.link(nodes[i-1],nodes[i],kind,kind==='corridor'?2.2:1.2,r*(angles[i]-angles[i-1])/PX);}return nodes;}
 
 row(r,lo,hi,n,group,left,right){const ids=[];for(let k=0;k<n;k++){const a=lo+(hi-lo)*(k+1)/(n+1),id=this.pnode(r,a,group==='balcony'?1:0,'seat');this.seats.push({node:id,group,angle:a,row:r,section:lo});ids.push(id);}const chain=[left,...ids,right];for(let k=1;k<chain.length;k++)this.link(chain[k-1],chain[k],'row',.55,r*(hi-lo)/(n+1)/PX);}
 build(){
  const A=Array.from({length:33},(_,i)=>Math.PI*i/32),mainIx=[0,4,12,20,28,32],doorIx=[0,4,8,12,16,20,24,28,32];
  const cross=this.ring(247,A,0),corr=this.ring(357,A,0,'corridor');this.cross=cross;this.corr=corr;
  
  const orchA=mainIx.map(i=>A[i]),orows=Array.from({length:19},(_,i)=>93+i*7.5),og=orchA.map(()=>[]);
  for(let j=0;j<6;j++){for(let r=0;r<orows.length;r++){og[j].push(this.pnode(orows[r],orchA[j],0));if(r)this.link(og[j][r-1],og[j][r],'aisle');}const access=this.pnode(240,orchA[j],0,'access','A'+(j+1));this.link(og[j].at(-1),access,'access',1);this.link(access,cross[mainIx[j]],'access',1);}
  
  let slots=[];for(let r=0;r<19;r++)for(let j=0;j<5;j++)slots.push({r,j,w:(orchA[j+1]-orchA[j])*orows[r]});this.allocate(slots,527,(s,n)=>this.row(orows[s.r],orchA[s.j],orchA[s.j+1],n,'orchestra',og[s.j][s.r],og[s.j+1][s.r]));
  const mr=[265,273,281,289,297,305,313],mg=doorIx.map(()=>[]);
  for(let j=0;j<9;j++){this.nodes[cross[doorIx[j]]].label ||= '';for(let r=0;r<7;r++){mg[j].push(this.pnode(mr[r],A[doorIx[j]],0));this.link(r?mg[j][r-1]:cross[doorIx[j]],mg[j][r],'aisle');}const d=this.pnode(334,A[doorIx[j]],0,'door','D'+(j+1));this.link(mg[j].at(-1),d,'aisle');this.link(d,corr[doorIx[j]],'door',1.2);}
  slots=[];for(let r=0;r<7;r++)for(let j=0;j<8;j++)slots.push({r,j,w:(A[doorIx[j+1]]-A[doorIx[j]])*mr[r]});this.allocate(slots,486,(s,n)=>this.row(mr[s.r],A[doorIx[s.j]],A[doorIx[s.j+1]],n,'mezzanine',mg[s.j][s.r],mg[s.j+1][s.r]));
  
  for(const side of ['L','R']){const isL=side==='L',ixs=isL?[3,8]:[24,29],x=isL?92:948,hub=this.node(x,400,0,'portico',isL?'West portico':'East portico');for(let j=0;j<2;j++){let p=polar(390,A[ixs[j]]),d=this.node(p.x,p.y,0,'sideDoor',side+(j+1));this.link(corr[ixs[j]],d,'door',1.1);const walk=this.node(x,p.y,0,'portico');this.link(d,walk,'portico',1.8);this.link(walk,hub,'portico',1.8);}let end=this.node(x,492,0,'outside',isL?'West exterior stairs':'East exterior stairs');this.link(hub,end,'sideStair',1.2,10);this.exits.push({node:end,key:side,label:isL?'West stairs':'East stairs',count:0});}
  
  const foyer=this.node(520,550,0,'foyer','FOYER');this.foyer=foyer;
  for(const ix of [14,16,18]){const p=polar(388,A[ix]),d=this.node(p.x,p.y,0,'foyerDoor','F'+(ix/2-6));this.link(corr[ix],d,'foyerDoor',1.3);this.link(d,foyer,'foyer',2.5);}
  const landing=this.node(520,645,0,'landing','Exterior landing');for(let i=0;i<5;i++){let d=this.node(440+i*40,602,0,'outerDoor','E'+(i+1));this.link(foyer,d,'foyer',2);this.link(d,landing,'mainDoor',1.4);}const mainEnd=this.node(520,716,0,'outside','Main exterior steps');this.link(landing,mainEnd,'mainStair',6,12);this.exits.push({node:mainEnd,key:'main',label:'Main exit',count:0});
  
  const BA=Array.from({length:33},(_,i)=>Math.PI*i/32),bRing=this.ring(321,BA,1,'corridor'),bix=[0,10,22,32],br=[263,277,291],bg=bix.map(()=>[]);
  for(let j=0;j<4;j++){for(let r=0;r<3;r++){bg[j].push(this.pnode(br[r],BA[bix[j]],1));if(r)this.link(bg[j][r-1],bg[j][r],'aisle');}this.link(bg[j].at(-1),bRing[bix[j]],'aisle');}
  slots=[];for(let r=0;r<3;r++)for(let j=0;j<3;j++)slots.push({r,j,w:(BA[bix[j+1]]-BA[bix[j]])*br[r]});this.allocate(slots,316,(s,n)=>this.row(br[s.r],BA[bix[s.j]],BA[bix[s.j+1]],n,'balcony',bg[s.j][s.r],bg[s.j+1][s.r]));
  for(let j=0;j<4;j++){const at=bRing[bix[j]],p=this.nodes[at],stage=j===0||j===3,left=j<2,key=stage?(left?'stageL':'stageR'):'main';
   this.nodes[at].label=stage?(left?'West stage stairs':'East stage stairs'):(left?'West balcony stairs':'East balcony stairs');this.nodes[at].kind='stair';
   
   const mid=this.node(p.x+(left?-18:18),p.y+25,1,'stairLanding');this.edges[this.link(at,mid,'stair',1.3,5)].stairGroup=key;
   const down=this.node(stage?(left?130:910):(left?410:630),stage?110:531,0,'stairLanding',stage?'Stage stair landing':'Balcony stair landing');this.edges[this.link(mid,down,'stair',1.3,5)].stairGroup=key;
   if(stage){const end=this.node(left?60:980,110,0,'outside',left?'West stage exit':'East stage exit');this.link(down,end,'stageDoor',1.2,4);this.exits.push({node:end,key,label:left?'West stage exit':'East stage exit',count:0});}else this.link(down,foyer,'foyer',1.8,6);
  }
 }
 allocate(slots,N,fn){const sum=slots.reduce((a,s)=>a+s.w,0);let counts=slots.map(s=>Math.floor(N*s.w/sum));let rest=N-counts.reduce((a,b)=>a+b,0);slots.map((s,i)=>({i,r:N*s.w/sum-counts[i]})).sort((a,b)=>b.r-a.r).slice(0,rest).forEach(s=>counts[s.i]++);slots.forEach((s,i)=>fn(s,counts[i]));}

}
return {Layout,polar};
})();
