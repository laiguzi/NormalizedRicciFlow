
import os
import json
from constant_gamma import *
import imageio
from PIL import Image
import pygraphviz as pg
import networkx as nx

def load_config(config_path):
    f = open(config_path, 'r').read()
    config = json.loads(f)
    return config['paths'], config['local_data'], config['graph_param'], config['ricciflow'], config['surgery_param']

def check_dir(target_dir):
    if not os.path.exists(target_dir):
        raise Exception("%s not exists!!!" % target_dir)
    return target_dir

def make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    else:
        for fi in os.listdir(target_dir):
            if fi.endswith('.png'):
                os.remove(os.path.join(target_dir, fi))
    return target_dir

def get_graph_dir(collection, filename=None, rootpath=data_root):
    if not filename:
        filename = "%s.gexf" % collection
    return check_dir(os.path.join(rootpath, collection, filename))

def make_save_dir(collection, mode=None,rootpath=save_root):
    if mode != None:
        return make_dir(os.path.join(rootpath, collection, mode))
    else:
        return make_dir(os.path.join(rootpath, collection))


def DrawGraphWithEdgeLength(G_origin: nx.Graph(), file):
    G = pg.AGraph()
    G.add_nodes_from(G_origin.nodes())
    normalize = float(2 * int(G_origin.number_of_edges()))
    for e in G_origin.edges():
        G.add_edge(e[0], e[1], len = float(normalize * G_origin[e[0]][e[1]]['weight']))
    G.draw(file, format='png', prog='neato')

def create_gif(image_dir, save_name, resize=(400,400)):
    
    image_list = []
    frames = []
    for f in os.listdir(image_dir):
        if f.endswith('.png'):
            image_list.append(os.path.join(image_dir, f))
    image_list.sort(key= lambda x:(int(x[-6:-4]) if x[-6] != '/' else int(x[-5])))
    
    for image_name in image_list:
        if image_name.endswith('.png'):
            image = Image.open(image_name)
            image = image.resize(resize)
            frames.append(image)
    
    imageio.mimsave(save_name, frames, 'GIF', duration = 0.1)
