
import networkx as nx
import itertools

def Gab(a,b):
    G = nx.Graph()
    a = a + 1
    b = b + 1
    for i in range(b):
        edges = itertools.combinations(range(i * a, (i + 1) * a ), 2)
        G.add_edges_from(edges)
    G.add_edges_from(itertools.combinations(range(0, a * b, a), 2))
    return G

def SBM(dic):
    sizes = dic['sizes']
    probs = dic['probs']
    G = nx.stochastic_block_model(sizes, probs)
    return G

def LFR(dic):
    n = dic['n']
    tau1 = dic['tau1']
    tau2 = dic['tau2']
    mu = dic['mu']
    average_degree = dic['average_degree']
    G = nx.generators.community.LFR_benchmark_graph(n, tau1, tau2, mu, average_degree, seed=0)
    communities = nx.get_node_attributes(G, 'community')
    nx.set_node_attributes(G, {n:None for n in G.nodes()}, 'community_idx')
    index = 0
    for node in G.nodes():
        if G.nodes[node]['community_idx'] == None:
            for c_node in G.nodes[node]['community']:
                G.nodes[c_node]['community_idx'] = index
                del G.nodes[c_node]['community']
            index += 1
        else:
            continue
    #print(nx.get_node_attributes(G, 'community_idx'))
    #print(nx.get_node_attributes(G, 'community'))
    #nx.write_gexf(G, "hhh.gexf")
    return G

#def football(dic):



if __name__ == "__main__":
    G = LFR({"n":100, "tau1": 3, "tau2": 1.5, "mu":0.5, "average_degree":10})
    #print(G.nodes[15]["community"])
