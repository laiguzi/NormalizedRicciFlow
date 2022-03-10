# NormalizedRicciFlow

This is an implementation of the paper **Normalized Discrete Ricci Flow Used in Community Detection**

## Abstract
Complex network is a mainstream form of unstructured data in real world. 
Detecting communities in complex networks bears a wide range of applications. 
Different from the existing methods, which concentrate on applying statistics, graph theory or combinations, this work presents a new algorithm along a geometric avenue.
By utilizing normalized discrete Ricci flow with modified $\sigma$-weight-sum, and employing a limit-free Ricci curvature using $\ast$-coupling, this algorithm prevents the graph from collapsing to a point, and eliminates a hyper parameter $\alpha$ in discrete Ollivier Ricci curvature. Besides, experiments on real-world networks and artificial networks have shown that this normalized algorithm has a matching or better result, and is more robust with regard to unnormalized one.

```bash
cd NormalizedRicciFlow/StarRicciFlow/
./doit.sh
```