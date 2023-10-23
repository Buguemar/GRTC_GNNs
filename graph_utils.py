import matplotlib.pyplot as plt
import networkx as nx
import numpy as np 
import os
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch.nn as nn

from datasets import list_datasets, load_dataset, list_metrics, load_metric
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Dataset, Data, download_url
from tqdm import tqdm
from transformers import BertModel
from transformers import AutoTokenizer
from torch_geometric.utils import to_networkx
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader

class MyGraphDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(MyGraphDataset, self).__init__(root, transform, pre_transform)
     
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename+".csv"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        from transformers import BertModel
        import torch
        bert_tkz = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        with_edge_attr=False
        try: 
            with_edge = self.data["edges_attr"]
            with_edge_attr= True
        except:
            pass
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):   
            tmpn=mol["nodes"][1:-1].split(",")
            nodes_ids=[int(x) for x in tmpn] 
            try:
                spliteado=mol["edges"].split("], [")
                origen=spliteado[0][2:]
                target=spliteado[1][:-2]
                tmpo=origen.split(',')
                sources_index=[int(x) for x in tmpo]
                tmpd=target.split(',')
                targets_index=[int(x) for x in tmpd]
                indexes=[sources_index,targets_index]
                all_indexes=torch.tensor(indexes).long()
            except:
                indexes=[[],[]] 
                all_indexes=torch.tensor(indexes).long()
            
            tmpf=mol["node_features"][1:-1].split(", ")
            tmpfs=[term[1:-1] for term in tmpf]  
            
            node_fea=[]
            for term in tmpfs:
                
                q_temp=torch.tensor(bert_tkz.encode(term))
                
                if len(q_temp)==3: 
                    q=q_temp[1]
                    q_emb=model.get_input_embeddings()(q)
                    q_emb=q_emb.detach().numpy()
                if len(q_temp)>3:
                    q=[]
                    for sub_w in q_temp[1:-1]: 
                        sub_we= model.get_input_embeddings()(sub_w)
                        q.append(sub_we.detach().numpy())
                    q_emb=np.mean(np.asarray(q), axis=0) 
                    
                if len(q_temp)<3:
                    print ("Key Error")
                    break
                    
                node_fea.append(q_emb) 

            node_fea_emb=torch.tensor(np.asarray(node_fea)).float() 
            
            if with_edge_attr:
                if type(mol["edges_attr"])==str:
                    try:
                        edge_feats=mol["edges_attr"][1:-1].split(',')
                        edge_feats=[[int(x)] for x in edge_feats]
                        edge_feats=torch.tensor(edge_feats)
                    except:
                        edge_feats=None  
            else: 
                edge_feats=None                             
            
            label=self._get_label(mol["label"])
            data=Data(x=node_fea_emb, edge_index=all_indexes, edge_attr=edge_feats, y=label)
            
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64) 

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
    
    
class MyGraphDatasetContext(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(MyGraphDatasetContext, self).__init__(root, transform, pre_transform)
     
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename+".csv"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        from transformers import BertModel
        import torch
        bert_tkz = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                              )
        with_edge_attr=False
        try: 
            with_edge = self.data["edges_attr"]
            with_edge_attr= True
        except:
            pass
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):   
            raw_text=mol["raw_text"]  
            long_flag= False 
            marked_text = "[CLS] " + raw_text + " [SEP]"
            tokenized_text = bert_tkz.tokenize(marked_text)
            if len(tokenized_text) > 512:
                long_flag= True
                tokenized_text = tokenized_text[:512]
                
            indexed_tokens = bert_tkz.convert_tokens_to_ids(tokenized_text)
            dict_otp=dict()
            for tup in zip(tokenized_text, indexed_tokens):
                dict_otp[tup[1]]=tup[0]
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            model.eval()
            with torch.no_grad():
                    outputs = model(tokens_tensor, segments_tensors)
                    hidden_states = outputs[2]
                           
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)
            dict_final=dict()
            for i, token_str in enumerate(tokenized_text):
                if token_str not in dict_final:
                    dict_final[token_str]=[i]
                else: 
                    dict_final[token_str].append(i)
       
                    
            tmpn=mol["nodes"][1:-1].split(",")
            nodes_ids=[int(x) for x in tmpn] 
            try:
                spliteado=mol["edges"].split("], [")
                origen=spliteado[0][2:]
                target=spliteado[1][:-2]
                tmpo=origen.split(',')
                sources_index=[int(x) for x in tmpo]
                tmpd=target.split(',')
                targets_index=[int(x) for x in tmpd]
                indexes=[sources_index,targets_index]
                all_indexes=torch.tensor(indexes).long()
            except:
                indexes=[[],[]] 
                all_indexes=torch.tensor(indexes).long()
            
            tmpf=mol["node_features"][1:-1].split(", ")
            tmpfs=[term[1:-1] for term in tmpf] 
            
            node_fea=[]
            
            for term in tmpfs: 
                q_temp=torch.tensor(bert_tkz.encode(term)).numpy()
                
                if term not in tokenized_text:
                    
                    if long_flag==False:   
                        to_average=[]
                        for qt in q_temp[1:-1]:
                            f_query=dict_final[dict_otp[qt]]
                            to_average+=f_query
                        to_return=[]
                        for fq in to_average:
                            to_return.append(token_embeddings[fq][-1])

                        result=torch.mean(torch.stack(to_return, dim = 0),0)
                        result=result.numpy()
                        
                    else:  
                        if len(q_temp)==3: 
                            q_emb=model.get_input_embeddings()(torch.tensor(q_temp[1]))
                            result=q_emb.detach().numpy()
                        if len(q_temp)>3: 
                            q=[]
                            for sub_w in q_temp[1:-1]: 
                                sub_we= model.get_input_embeddings()(torch.tensor(sub_w))
                                q.append(sub_we.detach().numpy())
                            result=np.mean(np.asarray(q), axis=0)

                        if len(q_temp)<3:
                            print ("Key Error")
                            break
                        
                else: 
                    f_query=dict_final[term]
                    to_return=[]
                    
                    for fq in f_query:
                        to_return.append(token_embeddings[fq][-1]) 

                    result=torch.mean(torch.stack(to_return, dim = 0),0)
                    result=result.numpy()
                    
                node_fea.append(result) 

            node_fea_emb=torch.tensor(np.asarray(node_fea)).float() 
   
            if with_edge_attr:
                if type(mol["edges_attr"])==str:
                    try:
                        edge_feats=mol["edges_attr"][1:-1].split(',')
                        edge_feats=[[int(x)] for x in edge_feats]
                        edge_feats=torch.tensor(edge_feats)
                    except:
                        edge_feats=None   
            else: 
                edge_feats=None                             
            
            label=self._get_label(mol["label"])
            data=Data(x=node_fea_emb, edge_index=all_indexes, edge_attr=edge_feats, y=label)
            
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64) 

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
 
    
class MyGraphDatasetEmb(Dataset):
    def __init__(self, root, filename, pre_embedding='word2vec-google-news-300', test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        self.pre_embedding = pre_embedding
        super(MyGraphDatasetEmb, self).__init__(root, transform, pre_transform)
        """ pre_embedding: 'word2vec-google-news-300', 'glove-wiki-gigaword-300' """
        
    @property
    def raw_file_names(self):
        return self.filename+".csv"

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]
        

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        from transformers import BertModel
        import torch
        emb_vectors = gensim.downloader.load(self.pre_embedding)
        with_edge_attr=False
        try: 
            with_edge = self.data["edges_attr"]
            with_edge_attr= True
        except:
            pass
        
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):   
            tmpn=mol["nodes"][1:-1].split(",")
            nodes_ids=[int(x) for x in tmpn] 
            try:
                spliteado=mol["edges"].split("], [")
                origen=spliteado[0][2:]
                target=spliteado[1][:-2]
                tmpo=origen.split(',')
                sources_index=[int(x) for x in tmpo]
                tmpd=target.split(',')
                targets_index=[int(x) for x in tmpd]
                indexes=[sources_index,targets_index]
                all_indexes=torch.tensor(indexes).long()
            except:
                indexes=[[],[]] 
                all_indexes=torch.tensor(indexes).long()
            
            tmpf=mol["node_features"][1:-1].split(", ")
            tmpfs=[term[1:-1] for term in tmpf]  
            
            node_fea=[]
            for term in tmpfs:
                try:
                    node_fea.append(emb_vectors.get_vector(term)) 
                except:
                    node_fea.append(emb_vectors.get_vector("unk"))

            node_fea_emb=torch.tensor(np.asarray(node_fea)).float() 
            
            if with_edge_attr:
                if type(mol["edges_attr"])==str:
                    try:
                        edge_feats=mol["edges_attr"][1:-1].split(',')
                        edge_feats=[[int(x)] for x in edge_feats]
                        edge_feats=torch.tensor(edge_feats)
                    except:
                        edge_feats=None  
            else: 
                edge_feats=None                            
            
            label=self._get_label(mol["label"])
            data=Data(x=node_fea_emb, edge_index=all_indexes, edge_attr=edge_feats, y=label)
            
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'data_test_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64) 

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data
    
    
def from_dataset2df(name_data, dataset_partition, columns): 
    n_columns=len(columns)
    df_partition={}
    data_iter=np.array(dataset_partition)
    for i in range(n_columns):
        temp=[]
        for x in data_iter:
            temp.append(x[columns[i]])
        df_partition[columns[i]]=temp
    
    df_data= pd.DataFrame(data=df_partition)
    new_columns = ["content", "label"]
    dict_col = {}
    for column,new_column in zip(columns,new_columns):
        dict_col[column]=new_column
        
    df_data.rename(columns=dict_col, inplace=True)
    if name_data=="app_reviews":
        new_df_data = pd.DataFrame({'label': np.asarray(df_data["label"]-1)})
        df_data.update(new_df_data)
    if name_data=="hyperpartisan_news_detection":
        new_df_data = pd.DataFrame({'label': np.asarray([int(lab) for lab in np.asarray(df_data["label"])])})
        df_data.update(new_df_data)
        
      
    return df_data.sample(frac=1)
    
def visualize_graph(G, color, mapping, edge_labels=False):
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    Gn=nx.relabel_nodes(G, mapping)
    pos=nx.spring_layout(Gn, seed=42)
    if edge_labels:           
        edge_labels=dict([((u,v,),d['weight']) for u,v,d in Gn.edges(data=True)])
        nx.draw_networkx_edge_labels(Gn,pos,edge_labels=edge_labels)
        nx.draw(Gn, pos, with_labels=True, node_size=300, node_color=color)
        plt.show()
    else:
        nx.draw_networkx(Gn, pos, with_labels=True,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()
    
