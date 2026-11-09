"""Reconstruct display identities from integer cell transfers, without changing the model."""
from collections import deque
import numpy as np

def person_tracks(result):
    ids=result.b.ids
    edges=[(result.b.idx[a],result.b.idx[b]) for a,b in result.edges]
    initial=result.occ[0].astype(np.int64).copy()
    for e,(a,b) in enumerate(edges):
        n=int(result.flows[0,e]);initial[a]+=n;initial[b]-=n
    assert (initial>=0).all() and initial.sum()==1329
    people=[deque() for _ in ids]
    tracks=[]
    for node,count in enumerate(initial):
        for _ in range(int(count)):
            pid=len(tracks);tracks.append([0,node]);people[node].append(pid)
    for step,flows in enumerate(result.flows):
        arrivals=[]
        for e in np.flatnonzero(flows):
            a,b=edges[e];n=int(flows[e])
            assert n==flows[e] and n<=len(people[a])
            for _ in range(n):
                pid=people[a].popleft();tracks[pid].extend([step+1,b]);arrivals.append((b,pid))
        for node,pid in arrivals:people[node].append(pid)
        assert np.array_equal([len(q) for q in people],result.occ[step]),step
    return tracks
