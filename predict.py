import os
import json
import models
import data_processing

if __name__ == "__main__":
    exp_dir = r'runs/Varying layers and filters/18'
    f = open((exp_dir+'/config.json'))
    exp_para = json.load(f)
    filters = exp_para['n_filters']
    layers = exp_para['n_layers']
    model = models.build_conv_network(layers, filters)
    model.summary()
    model.load_weights((exp_dir + r'/best_weights.ckpt.data-00000-of-00001'))
