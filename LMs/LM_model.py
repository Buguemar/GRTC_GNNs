import argparse
import csv
import dgl
import itertools
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random, re, os, nltk
import sys, random
import torch
import torch.nn.functional as F
import tqdm
import time, datetime
import torch.optim as optim
import torch.nn as nn
import word2vec
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import wandb


from datasets import list_datasets, load_dataset, list_metrics, load_metric
from itertools import compress
from torchmetrics import F1Score
from transformers import BertModel
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer  #nltk.stem
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, BertModel, BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer, logging
logging.set_verbosity_error()


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_length): 

        self.labels = list(df['label'])
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = max_length, truncation=True,
                                return_tensors="pt") for text in df['content']] ##change max_lentgh

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class LM_LightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("train_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1]))
        return loss["loss"]
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("val_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1])) 
        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr ) #(FT) 1e-5, or 5e-6, (pre-trained) 1e-3
        return optimizer
    
    def forward_performance(self, batch): 
        y_true= batch[1]
        
        out = self(batch[0]['input_ids'].squeeze(1).to(self.device), batch[0]['attention_mask'].to(self.device))  
        loss = self.criterion(out, y_true.long())
        pred = out.argmax(dim=1)  
        acc=(pred == y_true).sum()/len(y_true)        
        f1_score = F1Score(num_classes=self.num_classes, average="macro").to(self.device)
        f1_ma=f1_score(pred, y_true.long())
                
        return {"loss": loss, "f1ma":f1_ma, 'acc':acc}
    
    def predict(self, loader, cpu_store=True, gpu_predict=True):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if gpu_predict:
            self.to(device)
        
        preds=[]
        for batch in loader: 
            out = self(batch[0]['input_ids'].squeeze(1).to(self.device), batch[0]['attention_mask'].to(self.device))  
            pred = out.argmax(dim=1)  
            
            del out 
            torch.cuda.empty_cache()
            
            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds+=list(pred) 
            
        self.train() 
        if not cpu_store:
            preds = torch.Tensor(preds)
        return preds
    
class LM_Classifier(LM_LightningModel):
    def __init__(self, num_classes, dropout, pre_trained_model, lr, type_train, lm):
        super(LM_Classifier, self).__init__()
        
        self.num_classes=num_classes
        if lm=='BERT': ## probar aun 
            self.bert = BertModel.from_pretrained(pre_trained_model) 
        else:
            self.bert = LongformerModel.from_pretrained(pre_trained_model) 
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, self.num_classes)
        self.lr= lr 
        
        if type_train!= 'FT': ##freeze (FZ)
            print ("Pretrained model - Training the classifier layer")
            for name, param in self.bert.named_parameters():
                if 'classifier' not in name:                
                    param.requires_grad = False
       

    def forward(self, input_id, mask):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = F.log_softmax(linear_output, dim=1)
        
        return final_layer

    def criterion(self, pred, label):
        return F.nll_loss(pred, label)
    

def create_loaders(dataset, bs, dataframes_path, tokenizer, max_length): #"allenai/longformer-base-4096"
    df_train=pd.read_csv(dataframes_path+dataset+"/soft/source_processed.csv")
    df_test=pd.read_csv(dataframes_path+dataset+"/soft/source_processed_test.csv")

    np.random.seed(112)
    df_train, df_val = np.split(df_train.sample(frac=1, random_state=42), [int(.8*len(df_train))])
    
    print("Train/Val/Test partitions:", len(df_train),"/",len(df_val),"/", len(df_test))
    train, val = Dataset(df_train, tokenizer, max_length), Dataset(df_val, tokenizer, max_length)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=bs)

    test = Dataset(df_test, tokenizer, max_length)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=bs)
    
    return train_dataloader, val_dataloader, test_dataloader    
    
    
def run_bunch_experiments(lm, pre_trained_model, project_name, file_to_save, type_train, num_classes, path_models, dropout, lr, path_results, filename, loaders, bunch=10, pat=10, ep=100):
    
    start = time.time()
    np.set_printoptions(precision=3)
    
    train_dataloader, val_dataloader, test_dataloader = loaders
    with open(path_results+type_train+"_"+filename+'.txt', 'a') as f:
    
        acc_tests=[]
        f1_tests=[]
        f1ma_tests=[]
        for i in range(bunch):    
            starti = time.time()
            
            model = LM_Classifier(num_classes, dropout, pre_trained_model, lr, type_train, lm)
            
            early_stop_callback = EarlyStopping(monitor="val_f1ma", mode="max", min_delta=1e-3, patience=pat, verbose=True)
            
            wandb_logger = WandbLogger(name=type_train+"_run"+str(i),
                                       id=str(i),
                                       save_dir=path_models+file_to_save,
                                       project= project_name)
            trainer = pl.Trainer(max_epochs=ep, accelerator='gpu', devices=1, 
                         callbacks=[early_stop_callback],
                         logger=wandb_logger,
                         enable_progress_bar=False) #gpus=1, 
            trainer.fit(model, train_dataloader, val_dataloader)

            wandb.finish()
                    
            ###### TESTING
            print ("\n----------- Evaluating model "+str(i)+"-----------\n")
            print ("\n----------- Evaluating model "+str(i)+"-----------\n", file=f)
            print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch)
            preds=model.predict(test_dataloader, cpu_store=False).int()

            trues=[]
            for data in test_dataloader:
                trues.append(data[1])
            trues = torch.concat(trues)

            acc=(trues ==preds).float().mean() 
            f1_score = F1Score(num_classes=num_classes, average=None)
            f1_all = f1_score(preds, trues)
            print ("Acc:", acc, file=f)
            print ("Acc:", acc)
            print ("F1-ma:", f1_all.mean(), file=f)
            print ("F1-ma:", f1_all.mean())
            print ("F1 none:", f1_all, file=f)
            print ("F1 none:", f1_all)
            acc_tests.append(acc.cpu().numpy())
            f1_tests.append(f1_all.cpu().numpy())
            f1ma_tests.append(f1_all.mean().cpu().numpy())

            endi = time.time()
            total_timei = endi - starti
            print("Running time "+str(i)+": "+ str(total_timei), file=f)
            print("Running time "+str(i)+": "+ str(total_timei))

        print ("\n************************************************", file=f)
        print ("RESULTS FOR N_RUN:", bunch, file=f)    
        print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)), file=f)
        print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
        print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
        print ("************************************************\n\n", file=f)
        print ("\n************************************************")
        print ("RESULTS FOR N_RUN:", bunch)   
        print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)))
        print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)))
        print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0))
        print ("************************************************\n\n")
        
    f.close()

    end = time.time()
    total_time = end - start
    print("\nRunning time for all the experiments: "+ str(total_time))
    
    return     
    
    
