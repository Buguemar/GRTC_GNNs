import yaml
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]='2'

from BOW_MLP.BOW_model import *

OVERWRITE=True

def main_run(config_file):
    start_time = time.time()
    dataframes_path = config_file["input_dir_folder"] 
    path_models = config_file["output_dir_folder"]
    name_dataset = config_file["dataset"]    
    num_classes = config_file["num_classes"]

    num_epoch = config_file["training"]["max_epochs"]
    bs = config_file["training"]["batch_size"]
    dropout = config_file["training"]["dropout"]
    lr = config_file["training"]["lr"] #1e-3 (pre-trained), 1e-5 (FT)
    dim_hidden = config_file["training"]["dim_hidden"]
    num_execs = config_file["experiment"]["runs"] 
    patience = config_file["experiment"]["patience"]
    path_metrics = config_file["path_metrics"]


    df_train = pd.read_csv(dataframes_path+name_dataset+"/soft/source_processed.csv")
    df_test = pd.read_csv(dataframes_path+name_dataset+"/soft/source_processed_test.csv")

    dataset = Sequences(df_train)  
    train_indices, val_indices = train_test_split(list(range(0,len(dataset))),test_size=0.2,random_state=42)

    print ("Train samples:", len(train_indices))
    print ("Validation samles:", len(val_indices))
    print ("Test samples:", df_test.shape[0])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    training_loader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=bs, sampler=val_sampler)
    dataset_test = Seq_test(df_test, dataset) 
    test_loader = DataLoader(dataset_test, batch_size=bs)
    loaders = training_loader, validation_loader, test_loader 

    dim_features=[dim_hidden] #32, 64, 128, 256

    run_bunch_experiments(path_models, path_metrics, num_classes, dim_features, name_dataset, lr, dropout, dataset, loaders, num_execs, patience, num_epoch)    

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