
import networkx as nx
import math
from multiprocessing import Pool, cpu_count
import numpy as np
import cvxpy as cvx
import time
import os
from utils import *
import surgery as Surgery


########################################################################
class StarNormalize:
    
    def __init__(self, G, weight="weight", proc=cpu_count()):
        self.G = G
        self.weight=weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2

    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths

    def _get_edge_density_distributions(self):
        densities = dict()

        def Gamma(i, j):
            return self.lengths[i][j]

        # Construct the density distributions on each node
        def get_single_node_neighbors_distributions(neighbors):
            # Get sum of distributions from x's all neighbors
            nbr_edge_weight_sum = sum([Gamma(x,nbr) for nbr in neighbors])

            if nbr_edge_weight_sum > self.EPSILON:
                result = [Gamma(x,nbr) / nbr_edge_weight_sum for nbr in neighbors]
            elif len(neighbors) == 0:
                return []
            else:
                result = [1.0 / len(neighbors)] * len(neighbors)
            result.append(0)
            return result

        for x in self.G.nodes():
            densities[x] = get_single_node_neighbors_distributions(list(self.G.neighbors(x)))

        return densities

    def _optimal_transportation_distance(self, x, y, d):
        star_coupling = cvx.Variable((len(y), len(x)))  # the transportation plan B
        # objective function sum(star_coupling(x,y) * d(x,y)) , need to do element-wise multiply here
        obj = cvx.Maximize(cvx.sum(cvx.multiply(star_coupling, d.T)))
        # constrains
        constrains = [cvx.sum(star_coupling)==0]

        constrains += [cvx.sum(star_coupling[:, :-1], axis=0, keepdims=True) == np.multiply(-1, x.T[:,:-1])]
        constrains += [cvx.sum(star_coupling[:-1, :], axis=1, keepdims=True) == np.multiply(-1, y[:-1])]

        constrains += [0 <= star_coupling[-1, -1], star_coupling[-1, -1] <= 2]
        constrains += [star_coupling[:-1,:-1] <= 0]
        constrains += [star_coupling[-1,:-1] <= 0]
        constrains += [star_coupling[:-1,-1] <= 0]

        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        # solve for optimal transportation cost
        return m

    def _distribute_densities(self, source, target):

        # Append source and target node into weight distribution matrix x,y
        source_nbr = list(self.G.neighbors(source))
        target_nbr = list(self.G.neighbors(target))

        # Distribute densities for source and source's neighbors as x
        if not source_nbr:
            source_nbr.append(source)
            x = [1]
        else:
            source_nbr.append(source)
            x = self.densities[source]

        # Distribute densities for target and target's neighbors as y
        if not target_nbr:
            target_nbr.append(target)
            y = [1]
        else:
            target_nbr.append(target)
            y = self.densities[target]

        # construct the cost dictionary from x to y
        d = np.zeros((len(x), len(y)))

        for i, src in enumerate(source_nbr):
            for j, dst in enumerate(target_nbr):
                assert dst in self.lengths[src], "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T  # the mass that source neighborhood initially owned
        y = np.array([y]).T  # the mass that target neighborhood needs to received

        return x, y, d

    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." % (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost

        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)

        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = m / self.lengths[source][target]  # Divided by the length of d(i, j)
        #print("Ricci curvature (%s,%s) = %f" % (source, target, result))

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        #if not self.lengths:
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()

        # Start compute edge Ricci curvature
        p = Pool(processes=self.proc)

        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        #result = self._wrap_compute_single_edge(args)
        result = p.map_async(self._wrap_compute_single_edge, args).get()
        p.close()
        p.join()

        return result

    def compute_ricci_curvature(self):
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())

        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / self.G.degree(n)


    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        
        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))
        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))

            sum_K_W = sum(self.G[v1][v2]["ricciCurvature"] * self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] *= (1.0 + step * (sum_K_W - self.G[v1][v2]["ricciCurvature"]))

            # Do normalization on all weights
            w = nx.get_edge_attributes(self.G, self.weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k]/sumw
                if w[k] < 0: w[k] = min([self.EPSILON, -w[k]])
                #assert(w[k]>0)
            nx.set_edge_attributes(self.G, w, self.weight)

            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)

            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))

        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))

        
        
        
        
        
#####################################################################################################        
class StarUnnormalize(StarNormalize):
    
    def __init__(self, G, weight="weight", proc=cpu_count()):
        self.G = G
        self.weight=weight
        self.proc = proc
        self.lengths = {}  
        self.densities = {}
        self.EPSILON = 1e-7
        self.base = math.e
        self.exp_power = 2
     
    
    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        
        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))
        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))

            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = (1.0-step * (self.G[v1][v2]["ricciCurvature"])) * self.lengths[v1][v2]
                
            ## Do normalization on all weights
            w = nx.get_edge_attributes(self.G, self.weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] / sumw * self.G.number_of_edges()
                if w[k] < 0: w[k] = min([self.EPSILON, -w[k]])
                assert(w[k]>0)
            nx.set_edge_attributes(self.G, w, self.weight)
            
            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)

            w = nx.get_edge_attributes(self.G, self.weight)
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))

        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))
        

#####################################################################################################
class aStarNormalize(StarNormalize):
    
    def __init__(self, G, weight="weight", proc=cpu_count()):
        self.G = G
        self.weight=weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2
        
    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        
        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))
        
        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))
            
            sum_K_W = sum(self.G[v1][v2]["ricciCurvature"] * self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            a = sum(self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = (1.0 + step * ((sum_K_W / a)- self.G[v1][v2]["ricciCurvature"])) * self.lengths[v1][v2]

            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)
            
            w = nx.get_edge_attributes(self.G, self.weight)
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))

        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))

        
#####################################################################################################        
class OllivierNormalize:
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2


    def _get_all_pairs_shortest_path(self):
        # Construct the all pair shortest path lookup
        lengths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight=self.weight))
        return lengths

    def _get_edge_density_distributions(self):
        densities = dict()

        # def inverse_sqr(dis):
        #     return dis
        #     #return 1.0/(dis)

        def inverse_sqr(dis):
            return self.base ** (-dis ** self.exp_power)

        # Construct the density distributions on each node
        def get_single_node_neighbors_distributions(neighbors, direction="successors"):

            # Get sum of distributions from x's all neighbors
            nbr_edge_weight_sum = sum([inverse_sqr(self.lengths[x][nbr]) for nbr in neighbors])

            if nbr_edge_weight_sum > self.EPSILON:
                result = [(1.0 - self.alpha) * inverse_sqr(self.lengths[x][nbr]) / nbr_edge_weight_sum for nbr in neighbors]
            elif len(neighbors) == 0:
                return []
            else:
                result = [(1.0 - self.alpha) / len(neighbors)] * len(neighbors)
            result.append(self.alpha)
            return result

        for x in self.G.nodes():
            densities[x] = get_single_node_neighbors_distributions(list(self.G.neighbors(x)))

        return densities

    def _distribute_densities(self, source, target):
        # Append source and target node into weight distribution matrix x,y
        source_nbr = list(self.G.predecessors(source)) if self.G.is_directed() else list(self.G.neighbors(source))
        target_nbr = list(self.G.successors(target)) if self.G.is_directed() else list(self.G.neighbors(target))

        # Distribute densities for source and source's neighbors as x
        if not source_nbr:
            source_nbr.append(source)
            x = [1]
        else:
            source_nbr.append(source)
            x = self.densities[source]["predecessors"] if self.G.is_directed() else self.densities[source]

        # Distribute densities for target and target's neighbors as y
        if not target_nbr:
            target_nbr.append(target)
            y = [1]
        else:
            target_nbr.append(target)
            y = self.densities[target]["successors"] if self.G.is_directed() else self.densities[target]

        # construct the cost dictionary from x to y
        d = np.zeros((len(x), len(y)))

        for i, src in enumerate(source_nbr):
            for j, dst in enumerate(target_nbr):
                assert dst in self.lengths[src], \
                    "Target node not in list, should not happened, pair (%d, %d)" % (src, dst)
                d[i][j] = self.lengths[src][dst]

        x = np.array([x]).T  
        y = np.array([y]).T  

        return x, y, d

    def _optimal_transportation_distance(self, x, y, d):
        rho = cvx.Variable((len(y), len(x)))  

        # objective function d(x,y) * rho * x, need to do element-wise multiply here
        obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))

        # \sigma_i rho_{ij}=[1,1,...,1]
        source_sum = cvx.sum(rho, axis=0, keepdims=True)
        constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
        prob = cvx.Problem(obj, constrains)

        m = prob.solve(solver="ECOS")  # change solver here if you want
        return m


    def _compute_ricci_curvature_single_edge(self, source, target):

        assert source != target, "Self loop is not allowed."  # to prevent self loop

        # If the weight of edge is too small, return 0 instead.
        if self.lengths[source][target] < self.EPSILON:
            print("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %
                           (source, target))
            return {(source, target): 0}

        # compute transportation distance
        m = 1  # assign an initial cost
        x, y, d = self._distribute_densities(source, target)
        m = self._optimal_transportation_distance(x, y, d)
        # compute Ricci curvature: k=1-(m_{x,y})/d(x,y)
        result = 1 - (m / self.lengths[source][target])  # Divided by the length of d(i, j)

        return {(source, target): result}

    def _wrap_compute_single_edge(self, stuff):
        return self._compute_ricci_curvature_single_edge(*stuff)

    def compute_ricci_curvature_edges(self, edge_list=None):
        if not edge_list:
            edge_list = []

        # Construct the all pair shortest path dictionary
        self.lengths = self._get_all_pairs_shortest_path()

        # Construct the density distribution
        if not self.densities:
            self.densities = self._get_edge_density_distributions()

        # Start compute edge Ricci curvature
        p = Pool(processes=self.proc)

        # Compute Ricci curvature for edges
        args = [(source, target) for source, target in edge_list]

        result = p.map_async(self._wrap_compute_single_edge, args).get()
        p.close()
        p.join()

        return result

    def compute_ricci_curvature(self):
        if not nx.get_edge_attributes(self.G, self.weight):
            print('Edge weight not detected in graph, use "weight" as edge weight.')
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = 1.0
        edge_ricci = self.compute_ricci_curvature_edges(self.G.edges())

        # Assign edge Ricci curvature from result to graph G
        for rc in edge_ricci:
            for k in list(rc.keys()):
                source, target = k
                self.G[source][target]['ricciCurvature'] = rc[k]

        # Compute node Ricci curvature
        for n in self.G.nodes():
            rc_sum = 0  # sum of the neighbor Ricci curvature
            if self.G.degree(n) != 0:
                for nbr in self.G.neighbors(n):
                    if 'ricciCurvature' in self.G[n][nbr]:
                        rc_sum += self.G[n][nbr]['ricciCurvature']

                # Assign the node Ricci curvature to be the average of node's adjacency edges
                self.G.nodes[n]['ricciCurvature'] = rc_sum / self.G.degree(n)
                

    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))

        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))
            
            sum_K_W = sum(self.G[v1][v2]["ricciCurvature"] * self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = (1.0 + step * (sum_K_W - self.G[v1][v2]["ricciCurvature"])) * self.lengths[v1][v2]

            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break
            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)
            
            w = nx.get_edge_attributes(self.G, self.weight)
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))
        
        
#####################################################################################################        
class OllivierUnnormalize(OllivierNormalize):
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2
        
        
    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))

        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))
            
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = (1.0-step * (self.G[v1][v2]["ricciCurvature"])) * self.lengths[v1][v2]

            ## Do normalization on all weights
            w = nx.get_edge_attributes(self.G, self.weight)
            sumw = sum(w.values())
            for k, v in w.items():
                w[k] = w[k] / sumw * self.G.number_of_edges()
                if w[k] < 0: w[k] = min([self.EPSILON, -w[k]])
                assert(w[k]>0)
            nx.set_edge_attributes(self.G, w, self.weight)

            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break

            self.G = G1

            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)
            
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))
        
        
#####################################################################################################        
class aOllivierNormalize(OllivierNormalize):
    
    def __init__(self, G, alpha=0.5, weight="weight", proc=cpu_count()):
        self.G = G.copy()
        self.alpha = alpha
        self.weight = weight
        self.proc = proc
        self.lengths = {}  # all pair shortest path dictionary
        self.densities = {}  # density distribution dictionary
        self.EPSILON = 1e-7  # to prevent divided by zero
        self.base = math.e
        self.exp_power = 2
          
    def compute_ricci_flow(self, iterations=100, step=0.01, delta=1e-6, surgery={'name':'no_surgery', 'portion': 0.02, 'interval': 5}, save_gexf_dir=None):
        if not nx.is_connected(self.G):
            print("Not connected graph detected, compute on the largest connected component instead.")
            self.G = nx.Graph(max([self.G.subgraph(c) for c in nx.connected_components(self.G)], key=len))
            print('---------------------------')
            print(nx.info(self.G))

        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        #Save the original graph
        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "origin.gexf"))

        # Start compute edge Ricci flow
        t0 = time.time()

        if nx.get_edge_attributes(self.G, "original_RC"):
            print("original_RC detected, continue to refine the ricci flow.")
        else:
            self.compute_ricci_curvature()

            for (v1, v2) in self.G.edges():
                self.G[v1][v2]["original_RC"] = self.G[v1][v2]["ricciCurvature"]

        # Start the Ricci flow process
        self.rc_diff = []
        for i in range(iterations):
            # Save current graph
            nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%i))
            
            sum_K_W = sum(self.G[v1][v2]["ricciCurvature"] * self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            a = sum(self.G[v1][v2][self.weight] for (v1, v2) in self.G.edges())
            for (v1, v2) in self.G.edges():
                self.G[v1][v2][self.weight] = (1.0 + step * ((sum_K_W / a)- self.G[v1][v2]["ricciCurvature"])) * self.lengths[v1][v2]

            #Merge really adjacent node
            G1 = self.G.copy()
            merged = True
            while merged:
                merged = False
                for v1,v2 in G1.edges():
                    if G1[v1][v2][self.weight] < delta * 10:
                        G1 = nx.contracted_edge(G1, (v1, v2), self_loops=False)
                        merged = True
                        print("Contracted edge: (%d, %d)" % (v1,v2))
                        break

            self.G = G1


            self.compute_ricci_curvature()
            print("=== Ricciflow iteration % d ===" % i)
                
            w = nx.get_edge_attributes(self.G, self.weight)
            rc = nx.get_edge_attributes(self.G, "ricciCurvature")
            diff = max(rc.values()) - min(rc.values())

            print("Ricci curvature difference: %f" % diff)
            print("max:%f, min:%f | maxw:%f, minw:%f" % (max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

            if diff < delta:
                print("Ricci curvature converged, process terminated.")
                break

            # do surgery or any specific evaluation
            surgery_func = surgery['name']
            do_surgery = surgery['interval']
            portion = surgery['portion']
            if i != 0 and i % do_surgery == 0:
                self.G = getattr(Surgery, surgery_func)(self.G, self.weight, portion)

            # clear the APSP and densities since the graph have changed.
            self.densities = {}

        nx.write_gexf(self.G, os.path.join(save_gexf_dir, "%d.gexf"%iterations))
        print("\n%8f secs for Ricci flow computation." % (time.time() - t0))
        