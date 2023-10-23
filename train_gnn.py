import yaml
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]='2'

from IntuitiveGraphs.GNN_for_classification_model import *

OVERWRITE=True

def main_run(config_file):
    start_time = time.time()
    init = config_file["node_feature_init"] #BERT, BERT-c, word2vec, glove
    path_root = config_file["input_dir_folder"] 
    path_models = config_file["output_dir_folder"]
    path_results = config_file["results_dir_folder"]   
    num_classes = config_file["num_classes"]
    pre_trained_emb = config_file["pre_trained_embedding"] #'word2vec-google-news-300', 'glove-wiki-gigaword-300'
    filename = config_file["filename"]
        
    num_epoch = config_file["training"]["max_epochs"]
    bs = config_file["training"]["batch_size"]
    dropout = config_file["training"]["dropout"]
    lr = config_file["training"]["lr"] #1e-3 (pre-trained), 1e-5 (FT)
    
    num_execs = config_file["experiment"]["runs"] 
    patience = config_file["experiment"]["patience"]
    type_model = config_file["experiment"]["type_model"]
    n_layers = [config_file["experiment"]["num_layers"]]
    dim_features = [config_file["experiment"]["hidden_dims"]]
    project_name = type_model+"_"+init+"_"+filename
    file_to_save = project_name   #config_file["pre_file"]+project_name


    if init=='BERT':
        dataset= MyGraphDataset(root=path_root, filename=filename) 
        dataset_test= MyGraphDataset(root=path_root, filename=filename+"_test", test=True)
    elif init=='BERT-c':
        dataset= MyGraphDatasetContext(root=path_root, filename=filename) 
        dataset_test= MyGraphDatasetContext(root=path_root, filename=filename+"_test", test=True)
    else:
        dataset= MyGraphDatasetEmb(root=path_root, filename=filename, pre_embedding=pre_trained_emb ) 
        dataset_test= MyGraphDatasetEmb(root=path_root, filename=filename+"_test", test=True, pre_embedding=pre_trained_emb)

    dataset.num_classes = num_classes
    dataset_test.num_classes = num_classes

    run_bunch_experiments(dataset, dataset_test, path_models, path_results, n_layers, dim_features, file_to_save, 
                          type_model, lr, dropout, project_name, num_execs, patience, num_epoch)

    print(f"Finished whole execution of {num_execs} runs in {time.time()-start_time:.2f} secs")    

    
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)
    
    main_run(config_file)