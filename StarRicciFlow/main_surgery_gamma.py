
from utils_gamma import *
from data_load import Graph
import RicciFlow_gamma
import sys
from optparse import OptionParser
#import torch


class DoRicciFlow:
    
    def __init__(self, config, model_name):
        self.config = config
        self.model_name = model_name
        self.config_path = os.path.join('Configs', config)
        self.paths, self.local_data, self.graph_param, self.ricciflow_param, self.surgery_param  = load_config(self.config_path)

        # Initialize dirs
        if self.local_data:
            self.data_dir = get_graph_dir(self.paths['collection'], self.paths['file'])
        else:
            self.data_dir = None
        self.save_gexf_dir = make_save_dir(self.paths['save_collection'], 'gexf')
        self.save_gexf_with_surgery_dir = make_dir(os.path.join(self.save_gexf_dir, self.model_name, 'with_surgery'))
        self.save_gexf_without_surgery_dir = make_dir(os.path.join(self.save_gexf_dir, self.model_name, 'without_surgery'))


        # Initialize graph
        self.G = Graph(self.graph_param['name'], self.graph_param['param'], self.data_dir).G
        self.G_origin = self.G.copy()

    def process(self):
        #self.sr_G = getattr(RicciFlow_gamma, self.model_name)(self.G)
        #self.sr_G.compute_ricci_flow(iterations=self.ricciflow_param['iterations'],
        #        step=self.ricciflow_param['step'],
        #        delta=self.ricciflow_param['delta'], 
        #        save_gexf_dir=self.save_gexf_without_surgery_dir)
        

        self.sr_G_surgery = getattr(RicciFlow_gamma, self.model_name)(self.G_origin)
        self.sr_G_surgery.compute_ricci_flow(iterations=self.ricciflow_param['iterations'],
                step=self.ricciflow_param['step'],
                delta=self.ricciflow_param['delta'],
                surgery=self.surgery_param, 
                save_gexf_dir=self.save_gexf_with_surgery_dir)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = OptionParser(usage="Usage: main.py used to do star ricci flow on graphs.")

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    config = args[0]
    model_name = args[1]

    ricciflow = DoRicciFlow(config, model_name)

    return ricciflow.process()

if __name__ == "__main__":
    sys.exit(main())
