import yaml
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]='2'

from LMs.LM_model import *


OVERWRITE=True

def main_run(config_file):
    start_time = time.time()
    dataframes_path = config_file["input_dir_folder"] 
    path_models = config_file["output_dir_folder"]
    dataset = config_file["dataset"]    
    num_classes = config_file["num_classes"]
    pre_trained_model = config_file["pre_trained_model"] #bert-base-uncased, allenai/longformer-base-4096
    
    max_length = config_file["training"]["max_length"] #1024, 512, ...
    num_epoch = config_file["training"]["max_epochs"]
    bs = config_file["training"]["batch_size"]
    dropout = config_file["training"]["dropout"]
    lr = config_file["training"]["lr"] #1e-3 (pre-trained), 1e-5 (FT)
    
    lm = config_file["experiment"]["language-model"] ##Longformer or BERT
    filename = lm+"_"+dataset
    num_execs = config_file["experiment"]["runs"] 
    patience = config_file["experiment"]["patience"]
    type_train = config_file["experiment"]["type_train"]
    project_name = type_train+"_"+lm+"_"+dataset
    path_metrics = config_file["path_metrics"]
    file_to_save = config_file["pre_file"]+project_name


    if lm=='Longformer':
        tokenizer = LongformerTokenizer.from_pretrained(pre_trained_model)
    else: #BERT
        tokenizer = BertTokenizer.from_pretrained(pre_trained_model) 
      
    loaders = create_loaders(dataset, bs, dataframes_path, tokenizer, max_length)
    
    run_bunch_experiments(lm, pre_trained_model, project_name, file_to_save, type_train, num_classes, path_models, dropout, lr, path_metrics, filename, loaders, num_execs, patience, num_epoch)

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