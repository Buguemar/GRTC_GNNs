import os
import argparse
import csv
import dgl
import itertools
import numpy as np
import random, re, os, nltk
import sys, random
import torch
import torch.nn.functional as F
import tqdm
import time, datetime
import word2vec
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import wandb

from generation_module import *
from graph_utils import * 
from itertools import compress
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import logging
logging.set_verbosity_error()
from loader_TLGCN import *


def train_and_evaluate(WANDB_PATH, project_name, dataset, bs, ngram, num_exec, num_epoch, embedding_path, dim_emb, bert=False, dropout = 0.2, edges = True, bar = False):
      
    name = 'temp_model'
    model, total_timec, total_timetrain = train(WANDB_PATH, project_name, ngram, name, bar, dropout, embedding_path, dim_emb, dataset, bs, execn=num_exec, n_epochs=num_epoch, bert=bert, is_cuda=True, edges=edges)
    
    test_acc, test_f1, test_f1ma, total_timetest = test(WANDB_PATH, ngram, embedding_path, model, dataset, bs, execn=num_exec)

    np.set_printoptions(precision=3)
    print('Test acc: %.3f' % test_acc.numpy())
    print('Test F1:', test_f1)
    print('Test F1-ma: %.3f' % test_f1ma)
    
    return test_acc, test_f1, test_f1ma, total_timec, total_timetrain+total_timetest


def run_loop(WANDB_PATH, path_metrics, project_name, file_to_save, dataset, bs, ngram, num_execs, num_epoch, embedding_path, dim_emb, bert=False):
    np.set_printoptions(precision=3)
    acc_tests=[]
    f1_tests=[]
    f1ma_tests=[]
    times_create=0.0
    times_run=0.0
    start = time.time()
    np.set_printoptions(precision=3)
    
    with open(path_metrics+file_to_save+'.txt', 'a') as f:  
    
        for i in range(num_execs):
            print ("\nRunning execution", i)
            test_acc, test_f1, test_f1ma, time_create, time_run = train_and_evaluate(WANDB_PATH, project_name, dataset, bs, ngram, i, num_epoch, embedding_path, dim_emb, bert=bert)

            print("Graph creation time "+str(i)+": "+ str(time_create), file=f)
            print("Graph creation time "+str(i)+": "+ str(time_create))
            print("Running time "+str(i)+": "+ str(time_run), file=f)
            print("Running time "+str(i)+": "+ str(time_run))
            acc_tests.append(test_acc)
            f1_tests.append(test_f1)
            f1ma_tests.append(test_f1ma)
            times_create+=time_create
            times_run+=time_run

        print ("\n************************************************", file=f)
        print ("RESULTS FOR N_GRAM:", ngram, "EMB DIM_FEATURES:", embedding_path, file=f)    
        print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)), file=f)
        print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
        print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
        print ("Mean Creation Time:", times_create/num_execs, file=f)
        print ("Mean Running Time:", times_run/num_execs, file=f)
        print ("Mean Total Execution Time:",  times_create/num_execs + times_run/num_execs , file=f)
        print ("************************************************\n\n", file=f)
        print ("\n************************************************")
        print ("RESULTS FOR N_GRAM:", ngram, "EMB DIM_FEATURES:", embedding_path)  
        print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)))
        print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)))
        print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0))
        print ("Mean Creation Time:", times_create/num_execs)
        print ("Mean Running Time:", times_run/num_execs)        
        print ("Mean Total Execution Time:",  times_create/num_execs + times_run/num_execs)
        print ("************************************************\n\n")
        

    return acc_tests, f1_tests, f1ma_tests