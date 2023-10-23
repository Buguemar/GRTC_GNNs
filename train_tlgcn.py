import yaml
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]='2'

from TextLevelGCN.tlgcn_model import *

OVERWRITE=True
def main_run(config_file):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    dataset = config_file["dataset"]    
    project_name = config_file["project_name"]
    path_metrics = config_file["path_metrics"]
    file_to_save = config_file["pre_file"]+config_file["dataset"]
    num_execs = config_file["experiment"]["runs"] 
    patience = config_file["experiment"]["patience"]
    num_epoch = config_file["training"]["max_epochs"]
    bs = config_file["training"]["batch_size"]
    ng = config_file["training"]["n-gram"]
    embedding_path = config_file["training"]["embedding"]
    dim_emb = config_file["training"]["dim_emb"]
    bert = config_file["training"]["bert"]
    wandb_path = config_file["wandb_path"]

   
    acc_tests, f1ma_tests, f1_tests = run_loop(wandb_path, path_metrics, project_name, file_to_save, dataset, bs, ng, num_execs, num_epoch, embedding_path, dim_emb, bert=bert)

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