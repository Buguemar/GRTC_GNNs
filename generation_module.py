import pandas as pd
import datasets
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter
from datasets import list_datasets, load_dataset, list_metrics, load_metric
from itertools import groupby, zip_longest, compress
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from sklearn.datasets import fetch_20newsgroups
from tqdm import tqdm
from transformers import AutoTokenizer
from torch_geometric.utils import to_networkx

def num_co(a,b,lista, mode='forward'):
    count=0
    for i in range(len(lista)-1):
        if lista[i]==a and lista[i+1]==b:
            count+=1
    return count

def set_nodes(list_text):
    nodes=set(list_text)
    return nodes

def set_edges(list_text, mode='forward', ws=1, weighted=True, directed=True): 
    #forward mode: left-to-right graph  
    #window mode: outgoing edges to the surrounding terms of a word in a window.
    edges=[]
    if mode=='forward':
        for i in range(len(list_text)-1):
            if directed:
                if weighted:
                    times=num_co(list_text[i],list_text[i+1],list_text)
                    if (list_text[i],list_text[i+1], times) not in edges:
                        edges.append((list_text[i],list_text[i+1], times))
                else:
                    edges.append((list_text[i],list_text[i+1]))
            else:
                if weighted:
                    timesa=num_co(list_text[i],list_text[i+1],list_text)
                    timesb=num_co(list_text[i+1],list_text[i],list_text)
                    tuplar=(list_text[i+1],list_text[i],timesa+timesb)
                    if tuplar not in edges:
                        edges.append((list_text[i],list_text[i+1],timesa+timesb))
                        edges.append((list_text[i+1],list_text[i],timesa+timesb))
                else:
                    tuplar=(list_text[i+1],list_text[i])
                    if tuplar not in edges:
                        edges.append((list_text[i],list_text[i+1]))
                        edges.append((list_text[i+1],list_text[i]))
    
    elif mode=='window':
        dict_edges={}
        for i in range(len(list_text)):
            for w in range(1,ws+1):
                if directed:
                    try:
                        if (list_text[i],list_text[i+w]) not in dict_edges.keys():
                            dict_edges[(list_text[i],list_text[i+w])]=1
                    except:
                        continue
                else:
                    try:
                        if (list_text[i],list_text[i+w]) not in dict_edges.keys() and (list_text[i+w],list_text[i]) not in dict_edges.keys():
                            dict_edges[(list_text[i],list_text[i+w])]=1
                        elif (list_text[i],list_text[i+w]) in dict_edges.keys():
                            dict_edges[(list_text[i],list_text[i+w])]+=1
                        elif (list_text[i+w],list_text[i]) in dict_edges.keys():
                            dict_edges[(list_text[i+w],list_text[i])]+=1
                    except:
                        continue
 
        list_text.reverse()
        for i in range(len(list_text)):
            for w in range(1,ws+1):
                if directed:
                    try:
                        if (list_text[i],list_text[i+w]) not in dict_edges.keys():
                            dict_edges[(list_text[i],list_text[i+w])]=1
                        else: 
                            dict_edges[(list_text[i],list_text[i+w])]+=1
                    except:
                        continue
                            
        if weighted:                  
            edges=[(tupla[0],tupla[1],dict_edges[tupla]) for tupla in dict_edges.keys()]
            invertir=[(tupla[1],tupla[0],dict_edges[tupla]) for tupla in dict_edges.keys()]
        else:
            edges=list(dict_edges.keys())
            invertir=[(tupla[1],tupla[0]) for tupla in dict_edges.keys()]
        edges+=invertir
        
    return edges

def tokenize_sentences(data, labels, tokenizer="nltk"): 
    labels_return= np.asarray(labels)
    docs=np.asarray(data)

    if tokenizer=="bert":
        mytokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  #Change the pre-trained model if you want to try something different 
        temp_tokenized = [mytokenizer(sample) for sample in docs]
        tokens=[np.asarray(text['input_ids']) for text in temp_tokenized]
        lens = [len(set(x)) for x in tokens]  
        all_tokens=np.concatenate([list(set(x)) for x in tokens])
        for temp in temp_tokenized:
            tokenized.append([mytokenizer.convert_ids_to_tokens(doc[1:-1]) for doc in temp['input_ids']])
        
    else:
        all_tokens=[]
        temp_tokenized = [word_tokenize(sample) for sample in docs]
        lens = [len(set(sample)) for sample in temp_tokenized]  
        tokens=np.concatenate([list(set(sample)) for sample in temp_tokenized])
        for tokens_review in tokens:
            all_tokens+=set(tokens_review)

    lens=np.asarray(lens)        
    print ("Vocabulary size (",tokenizer,"):", len(set(all_tokens)))
    print ("\nLenght of text (w.r.t. #tokens):")
    print ("Min:", np.min(lens), "\tAverage:", np.mean(lens),"\tMax:",np.max(lens))
    
    return temp_tokenized, labels_return 

def check(tupla,all_edges): 
    for tup in all_edges:
        if tup[0]==tupla[0] and tup[1]==tupla[1]:
            return True, tup[-1]
    return False, None
    
def update(tupla, cont, all_edges):
    all_edges.remove((tupla[0],tupla[1],cont))
    all_edges.append((tupla[0],tupla[1],cont+1))
    return all_edges


def create_graphs(data_in, labels_in, weighted=False, directed=False, mode="forward", ws=1, plot=False):

    samples, labels = tokenize_sentences(data_in, labels_in)
    
    ngraph=0
    review2labels=[]
    review2nodes=[]
    review2edges=[]
    total=len(samples)
    for doc in samples:        
        nodes=set_nodes(doc)
        edges=set_edges(doc, weighted=weighted, mode=mode, ws=ws, directed=directed)
        
        all_nodes=set(nodes)
        all_edges=set(edges)
        
        review2nodes.append(all_nodes)
        review2edges.append(all_edges)
        review2labels.append(labels[ngraph])
        ngraph+=1
        
        if plot:
            plt.figure(figsize=(10,10))
            if directed:
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            G.add_nodes_from(all_nodes)
            pos=nx.spring_layout(G)
            if weighted:
                G.add_weighted_edges_from(all_edges)
                edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
                nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
            else: 
                G.add_edges_from(all_edges)
            nx.draw(G,pos, with_labels=True, font_weight='bold', node_size=400, node_color='cyan')
            plt.show()
            
    return review2nodes, review2edges, review2labels

def create_graph2Dataframe(df_text, df_labels, directed=True, plot=False, mode='forward', ws=1, weighted=False):
    nodes,edges,labels= create_graphs(df_text, df_labels, directed=directed, plot=plot, mode=mode, ws=ws, weighted=weighted)
    total_nodes_file=0
    cont=0
    sample=0
    dict_nodes_file={}

    for node_list in nodes:
        total_nodes_file+=len(node_list)
        for node in node_list:
            dict_nodes_file[node+'_sample:'+str(sample)]=cont
            cont+=1
        sample+=1
        
    filas=[]
    anterior=0
    for i in range(len(nodes)): 
        text=df_text[i] 
        nodos_sam=list(nodes[i]) 
        edges_sam=list(edges[i])
        label=labels[i] #-1 (if it applies)

        f_nodos=[dict_nodes_file[nodo+"_sample:"+str(i)] for nodo in nodos_sam]
        anterior=f_nodos[0]
        f_nodos_f=[nodo-anterior for nodo in f_nodos]
        list_source=[]
        list_target=[]
        list_attr=[]
        for edge in edges_sam:
            source=edge[0]
            target=edge[1]
            if weighted:
                edge_attr=edge[-1]
                list_attr.append(edge_attr)
            node_source=dict_nodes_file[source+'_sample:'+str(i)]
            node_target=dict_nodes_file[target+'_sample:'+str(i)]
            list_source.append(node_source)
            list_target.append(node_target)                

        list_source_f=[nodo-anterior for nodo in list_source]
        list_target_f=[nodo-anterior for nodo in list_target]

        f_edges=[list_source,list_target]         
        f_edges_f=[list_source_f,list_target_f] 
        fila=[f_nodos_f, nodos_sam, f_edges_f,label]
        if weighted:
            fila=[f_nodos_f, nodos_sam, f_edges_f,list_attr,label]
        filas.append(fila)
    final_columns=['nodes', 'node_features', 'edges', 'label']
    if weighted:
        final_columns=['nodes', 'node_features', 'edges', 'edges_attr', 'label']
        
    df = pd.DataFrame(filas, columns = final_columns)
    return df

