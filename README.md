# GraphSpace
Graph Space Package

This python package allow study population of networks in Graph Space [1] as a special case of Structure Space introduced in [2].
The package is organized as follow:

core:
- Graph: class creating a graph object
- GraphSet: class creating a graph set
- Mean: class computing the Frechèt Mean and the variance of a set of graphs [2]
- MeanIterative: bootstrapped version of Frechèt Mean [2]

distance:
- euclidean: compute the euclidean distance between nodes and vector attributes

matcher:
- Matcher: parent class created to match two graphs
- child classes: whatever algorithm used to match two different networks, based on topology, node attributes and edge attributes. 
Child class gives back a reordered node sequence optimizing the matching criteria
    ID: identity match
    GA: graduate assignment match[2]
- alignment: class aligning two graphs with a specified matcher

AlignCompute:
- mean_aac: compute the Frechet Mean with the AAC algorithm and a given matcher [1]
- gpca_aac: compute the Geodesic Pricipal Components with AAC and a given matcher [1]

Aknowledgement: A great aknowledgment goes to Brijnesh Jain, whose code is used as a starting point for this package.

Please cite the package as:

Calissano, Feragen, Vantini (2020) "GraphSpace" https://github.com/annacalissano/GraphSpace/


[1] Calissano, Anna, Feragen, Aasa and Vantini Simone "Graph Space: Geodesic Principal Components for aPopulation of Network-valued Data" MOX Report (2020)

[2] Jain, Brijnesh J., and Klaus Obermayer. "Structure spaces." Journal of Machine Learning Research 10.Nov (2009): 2667-2714.

[3] Gold, Steven, and Anand Rangarajan. "A graduated assignment algorithm for graph matching." IEEE Transactions on pattern analysis            and machine intelligence 18.4 (1996): 377-388.



