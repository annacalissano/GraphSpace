import os
import sys


from core import Graph

from matcher import Matcher, GA #BK
from distance import euclidean

import numpy as np
from scipy import stats
from scipy.sparse import lil_matrix, vstack
import time

x1={}
x1[0,0]=[1]
x1[1,1]=[10]
x1[2,2]=[5]
x1[3,3]=[100]
x1[0,1]=[2]
x1[1,0]=[2]
x1[1,2]=[2]
x1[2,1]=[2]
x1[2,3]=[2]
x1[3,2]=[2]
x1[3,0]=[2]
x1[0,3]=[2]

x2={}
x2[0,0]=[100]
x2[1,1]=[1]
x2[2,2]=[10]
x2[3,3]=[5]
x2[0,1]=[2]
x2[1,0]=[2]
x2[1,2]=[2]
x2[2,1]=[2]
x2[2,3]=[2]
x2[3,2]=[2]
x2[3,0]=[2]
x2[0,3]=[2]

x3={}
x3[0,0]=[5]
x3[1,1]=[10]
x3[2,2]=[100]
x3[3,3]=[1]
x3[0,1]=[2]
x3[1,0]=[2]
x3[1,2]=[2]
x3[2,1]=[2]
x3[2,3]=[2]
x3[3,2]=[2]
x3[3,0]=[2]
x3[0,3]=[2]



G1=Graph(x=x1, adj=None, s=None)
G2=Graph(x=x2, adj=None, s=None)
G3=Graph(x=x3, adj=None, s=None)

print("GA Matcher with C++")
matcherGA=GA(c=True)
matcherGA.match(G1,G2)
print(matcherGA.f)

print("GA Matcher with python")
matcherGA=GA()
matcherGA.match(G1,G2)
print(matcherGA.f)

print("GA Matcher with C++")
matcherGA=GA(c=True)
matcherGA.match(G1,G3)
print(matcherGA.f)

print("GA Matcher with python")
matcherGA=GA()
matcherGA.match(G1,G3)
print(matcherGA.f)

print("GA Matcher with C++")
matcherGA=GA(c=True)
matcherGA.match(G3,G2)
print(matcherGA.f)

print("GA Matcher with python")
matcherGA=GA()
matcherGA.match(G3,G2)
print(matcherGA.f)
