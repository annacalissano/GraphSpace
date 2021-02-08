# Read files downloaded from the website in edge list format and save graphset
from core import Graph
from core import GraphSet
import os

dir=''

G=GraphSet(graph_type='directed')
files=os.listdir(dir)
for f in files:
    if not f.endswith('png') and not f.endswith('txt') and not f.endswith('npy'):
        fh = open(dir+"/"+f,"r")
        id={}
        x={}
        index=0
        for e in fh:
            l=[int(i) for i in e.split()]
            print(l)
            if(l[0] not in id.keys()): id[l[0]]=index; index+=1
            if(l[1] not in id.keys()): id[l[1]] = index; index += 1

            x[id[l[0]],id[l[1]]]=[l[2]]
        G.add(Graph(x=x,adj=None,s=[0]))
        del(x,l,fh,id,index)

G.write_to_text(dir+"Mammals_Grooming_kickouts.txt")
