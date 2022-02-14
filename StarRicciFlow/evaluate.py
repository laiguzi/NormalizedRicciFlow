
import networkx as nx 
import importlib
import numpy as np
import os

def ARI(G, cc, clustering_label="club"):
    if importlib.util.find_spec("sklearn") is not None:
        from sklearn import preprocessing, metrics
    else:
        print("scikit-learn is not installed...")
        return -1

    complexlist = nx.get_node_attributes(G, clustering_label)

    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complexlist.values()))

    predict_dict = {}
    for idx, comp in enumerate(cc):
        for c in list(comp):
            predict_dict[c] = idx
    y_pred = []
    for v in complexlist.keys():
        y_pred.append(predict_dict[v])
    y_pred = np.array(y_pred)

    return metrics.adjusted_rand_score(y_true, y_pred)


def NMI(G, cc, clustering_label="club"):
    if importlib.util.find_spec("sklearn") is not None:
        from sklearn import preprocessing, metrics
    else:
        print("scikit-learn is not installed...")
        return -1
    
    complexlist = nx.get_node_attributes(G, clustering_label)
    
    le = preprocessing.LabelEncoder()
    y_true = le.fit_transform(list(complexlist.values()))
    
    predict_dict = {}
    for idx, comp in enumerate(cc):
        for c in list(comp):
            predict_dict[c] = idx
    
    y_pred = []
    for v in complexlist.keys():
        y_pred.append(predict_dict[v])
    y_pred = np.array(y_pred)
    
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def Modularity(G, cc, clustering_label="club"):
    return nx.algorithms.community.modularity(G, cc)


if __name__ == "__main__":
    #G = nx.karate_club_graph()
    history = []
    for i in range(100):
        g_file = os.path.join('/home/u2019000097/jupyterlab/Ricci/results/SBM/gexf/with_surgery', '{}.gexf'.format(i))
        G = nx.read_gexf(g_file)
        cc = list(nx.connected_components(G))
        ari = ARI(G, cc, 'block')
        history.append([i, len(cc), ari])
        print("number of connnected components: %d   ARI: %5f" % (len(cc), ari))
    sort = sorted(history, key = lambda x:x[2], reverse=True)
    print("Maximum: ", sort[0])

