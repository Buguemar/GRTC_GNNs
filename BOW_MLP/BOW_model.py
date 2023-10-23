import matplotlib.pyplot as plt
import numpy as np 
import time
import torch
import torch.nn as nn
import pandas as pd
import pytorch_lightning as pl
import re, os
import seaborn as sns
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from graph_utils import from_dataset2df
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torchmetrics import F1Score
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

def dataset_to_dataframe(dataset):    
    dict_datasets={"app_reviews":['review','star'], "imdb":['text','label'], "dbpedia_14":['content','label'],
              "hyperpartisan_news_detection":['text', 'hyperpartisan']}

    if dataset!="bbc":
        try: 
            my_data = load_dataset(dataset)
        except:
            my_data = load_dataset('hyperpartisan_news_detection', 'byarticle')
        columns=dict_datasets[dataset]
        try: 
            df_train=from_dataset2df(dataset, my_data['train'], columns).sample(n=7000)
            print ("\nLabel distribution:\n", Counter(df_train['label']))
            df_test=from_dataset2df(dataset, my_data['test'], columns).sample(n=3000)
            return df_train, df_test
        except:
            try:
                df_temp=from_dataset2df(dataset, my_data['train'], columns).sample(n=10000) #app 
            except: 
                df_temp=from_dataset2df(dataset, my_data['train'], columns) #hnd
                
    else: 
        df_temp=pd.read_csv("../Requirements/Raw.csv")
        df_temp=df_temp.sample(frac=1)
        
    if dataset!="app_reviews":
        df_train, df_test = train_test_split(df_temp, test_size=0.2)
    else:
        df_train, df_test = train_test_split(df_temp, test_size=0.3)        
    return df_train, df_test


def load_return_dataset(dataset): #name of dataset 
    df_train, df_test = dataset_to_dataframe(dataset)
    print ("Shape samples:", df_train.shape)
    print ("Head:", df_train.head())
    return df_train, df_test


class Sequences(Dataset):
    def __init__(self, df):
        #df = pd.read_csv(path)
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        self.sequences = self.vectorizer.fit_transform(df.content.tolist())
        self.labels = df.label.tolist()
        self.token2idx = self.vectorizer.vocabulary_
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]
    
    def __len__(self):
        return self.sequences.shape[0]
    
class Seq_test(Dataset):
    def __init__(self, dft, train_dataset):
        self.sequences = train_dataset.vectorizer.transform(dft.content.tolist())
        self.labels = dft.label.tolist()
        
    def __getitem__(self, i):
        return self.sequences[i, :].toarray(), self.labels[i]
    
    def __len__(self):
        return self.sequences.shape[0]
    
    
class MLPLightningModel(pl.LightningModule):
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
        optimizer = optim.Adam(self.parameters(), lr=self.lr) #0.001 
        return optimizer
    
    def forward_performance(self, batch):
        
        inputs=batch[0]
        y_true= batch[-1].to(self.device) 
        out = self(inputs.squeeze().to(self.device))
        loss = self.criterion(out.squeeze().to(self.device), y_true.long())
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
            out = self(batch[0].squeeze().to(self.device))
            pred = out.argmax(dim=1)  
            if cpu_store:
                pred = pred.detach().cpu().numpy()
            preds+=list(pred) 
            
        if not cpu_store:
            preds = torch.Tensor(preds)
        return preds
        
    
class BOW_Classifier_PT(MLPLightningModel):
    def __init__(self, num_classes, hidden, lr, dropout, vocab_size, criterion = torch.nn.CrossEntropyLoss()): 
        super(BOW_Classifier_PT, self).__init__()
        
        self.num_classes=num_classes
        self.ff1 = nn.Linear(vocab_size, hidden)
        self.ff2 = nn.Linear(hidden, self.num_classes)
        self.dropout = nn.Dropout(dropout)
        self.criterion = criterion
        self.lr = lr        
    
    def forward(self, inputs):        
        x = F.relu(self.ff1(inputs.squeeze(1).float()))
        x = self.dropout(x)
        x = self.ff2(x) 
        return x
            

        
def run_bunch_experiments(path_models, path_metrics, num_classes, dim_features, file_to_save, lr, dropout, dataset, loaders, bunch=10, pat=10, ep=100, progress_bar=False):
    
    start = time.time()
    np.set_printoptions(precision=3)
    
    training_loader, validation_loader, test_loader = loaders
    
    with open(path_metrics+'BOW+MLP_'+file_to_save+'.txt', 'a') as f:

        for dim in dim_features:
            print ("\nTRAINING MODELS #HIDDEN DIM:", dim)
            print ("\nTRAINING MODELS #HIDDEN DIM:", dim, file=f)

            acc_tests=[]
            f1_tests=[]
            f1ma_tests=[]
            for i in range(bunch):             
                starti = time.time()
                model = BOW_Classifier_PT(num_classes, dim, lr, dropout, len(dataset.token2idx))  

                early_stop_callback = EarlyStopping(monitor="val_f1ma", mode="max", min_delta=1e-3, patience=pat, verbose=True)
                logger = TensorBoardLogger(name="BOW_U_"+str(dim), save_dir=path_models+file_to_save)
                trainer = pl.Trainer(max_epochs=ep, accelerator='gpu', devices=1, callbacks=[early_stop_callback],logger=logger, enable_progress_bar=False) 
                trained = trainer.fit(model, training_loader, validation_loader)

                print ("\n----------- Evaluating model "+str(i)+"-----------\n")
                print ("\n----------- Evaluating model "+str(i)+"-----------\n", file=f)
                print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch)
                print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch, file=f)

                preds=model.predict(test_loader, cpu_store=False).int()

                trues=[]
                for data in test_loader:  
                    trues.append(data[-1])
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
            print ("RESULTS FOR HIDDEN DIM_FEATURES:", dim, file=f)    
            print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)), file=f)
            print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
            print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
            print ("************************************************\n\n", file=f)
            print ("\n************************************************")
            print ("RESULTS FOR HIDDEN DIM_FEATURES:", dim)   
            print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)))
            print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)))
            print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0))
            print ("************************************************\n\n")
            
    f.close()

    end = time.time()
    total_time = end - start
    print("\nRunning time for all the experiments: "+ str(total_time))
    
    return     
        