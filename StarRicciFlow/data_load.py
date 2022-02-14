
import graph_generate
import os
import networkx as nx
from utils_gamma import *

class Graph:

    def __init__(self, graph, args=None, data_dir=None):
        self.graph = graph
        if data_dir == None and args['my_graph'] == False:
            self.G = getattr(nx, graph, None)(*args['settings'])
        elif data_dir == None and args['my_graph'] == True:
                self.G = getattr(graph_generate, graph)(*args['settings'])
        else:
            if self.graph == 'football':
                self.G = nx.read_gml(data_dir)
            elif self.graph == 'facebook':
                edge_list = [x.strip() for x in open(data_dir, 'r').readlines()]
                self.G = nx.read_edgelist(edge_list, delimiter=' ')
                #print(self.G.nodes())
                circ_file = args['settings'][0]
                circs = [x.strip() for x in open(os.path.join(os.path.dirname(data_dir), circ_file['circles'])).readlines()]
                for circ in circs:
                    items = circ.split()
                    circ_id = items[0]
                    circ_nodes = items[1:]#[int(x) for x in items[1:]]
                    for node in circ_nodes:
                        if node in self.G.nodes():
                            self.G.nodes[node]['circle'] = circ_id
            else:
                edge_list = [x.strip() for x in open(data_dir, 'r').readlines()]
                self.G = nx.read_edgelist(edge_list, delimiter=',')
        
        n_edges = len(self.G.edges())
        weight = {e: 1.0 for e in self.G.edges()}
        nx.set_edge_attributes(self.G, weight, 'weight')
        self.node_colors()
        print("Data loaded. \nNumber of nodes： {}\nNumber of edges: {}".format(self.G.number_of_nodes(), self.G.number_of_edges()))

    def node_colors(self):
        if self.graph == 'karate_club_graph':
            for i in self.G.nodes():
                if self.G.nodes[i]['club'] == 'Officer':
                    self.G.nodes[i]['color'] = '#377eb8'
                else:
                    self.G.nodes[i]['color'] = '#ff7f00'
        else:
            pass

if __name__ == "__main__":
    graph = Graph('karate_club_graph')
    nx.write_gexf(graph.G, "karate.gexf")
