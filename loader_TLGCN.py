import argparse
import csv
import dgl
import gensim
import gensim.downloader
import numpy as np
import itertools
import sys, random
import re, os, nltk
import time, datetime
import torch
import torch.nn.functional as F
import tqdm
import word2vec
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from itertools import compress
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import logging
logging.set_verbosity_error()

from preprocessing import clean_str, remove_short
from generation_module import *
from graph_utils import * 


def stem_corpus(dataset, df, num=7000, n_short=3, mode="training"): 
    stemmer = WordNetLemmatizer()

    labels = [str(lab) for lab in df['label'][:num]] 
    corpus = [clean_str(doc, preprocess="tlgcn") for doc in df["content"][:num]]
    corpus = [remove_short(doc, n_short) for doc in corpus]
    
    if mode!="test":
        print ("Preprocessing for training...")
    else:
        print ("Preprocessing for testing...")
    
    
    tokenized_corpus = [word_tokenize(doc) for doc in corpus]
    results = []
    for doc in tokenized_corpus:
        results.append(' '.join([stemmer.lemmatize(word) for word in doc]))
    
    valid_mask=[]
    for doc in results:
        if doc!='':
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    final_results=list(compress(results, valid_mask))
    final_labels = list(compress(labels, valid_mask))    
    print (len(final_results), "/",num, "samples have been generated.")    
    
    results = list(zip(final_labels, final_results))
    results = ['\t'.join(line) for line in results]
    
    if mode!="test":
        with open(os.path.join('..', 'Data_TLGCN', 'data', dataset, dataset+'-stemmed.txt'), 'w') as f:
            f.write('\n'.join(results))
    else:
        with open(os.path.join('..', 'Data_TLGCN', 'data', dataset, dataset+'-test-stemmed.txt'), 'w') as f:
            f.write('\n'.join(results))      
             
        
def cut_datasets(train_rate, allowed):
    for dataset in allowed: 
        with open(os.path.join('..', 'Data_TLGCN', 'data', dataset, dataset+'-stemmed.txt')) as f:
            all_cases = f.read().split('\n')
            print('Working on Dataset:', dataset, ', with total samples:', len(all_cases))
            cut_index = int(len(all_cases) * train_rate)
            train_cases = all_cases[:cut_index]
            dev_cases = all_cases[cut_index:]

        with open(os.path.join('..', 'Data_TLGCN', 'data', dataset, dataset+'-train-stemmed.txt'), 'w') as f:
            f.write('\n'.join(train_cases))
        with open(os.path.join('..', 'Data_TLGCN', 'data', dataset, dataset+'-dev-stemmed.txt'), 'w') as f:
            f.write('\n'.join(dev_cases))

#columns name in datasets 
dict_datasets={"app_reviews":['review','star'], "imdb":['text','label'], "dbpedia_14":['content','label'],
              "hyperpartisan_news_detection":['text', 'hyperpartisan']}

def dataset_to_dataframe(dataset):
    if dataset!="bbc":
        try: 
            my_data = load_dataset(dataset)
        except:
            my_data = load_dataset('hyperpartisan_news_detection', 'byarticle')
        columns=dict_datasets[dataset]
        try: 
            df_train=from_dataset2df(dataset, my_data['train'], columns)
            df_test=from_dataset2df(dataset, my_data['test'], columns)
            return df_train, df_test
        except:
            df_temp=from_dataset2df(dataset, my_data['train'], columns)
            
    else: #BBC is downloaded from a specific URL -  Set dataframe manually 
        df_temp=pd.read_csv("../Requirements/Raw.csv")
        df_temp=df_temp.sample(frac=1)
                
    df_train, df_test = train_test_split(df_temp, test_size=0.2)

    return df_train, df_test


############################## GLOBALS #################################

my_labels={}
my_labels["app_reviews"]=[0,1,2,3,4]
my_labels["dbpedia_14"]=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
my_labels["imdb"]=[0,1]
my_labels["bbc"]=[0,1,2,3,4]
my_labels["hyperpartisan_news_detection"]=[0,1] 

NUM_ITER_EVAL = 200 
EARLY_STOP_EPOCH = 15

########################################################################

class DataHelper(object):
    def __init__(self, dataset, mode='train', vocab=None):
        allowed_data = ['app_reviews', 'dbpedia_14', 'imdb', 'bbc', 'hyperpartisan_news_detection']

        if dataset not in allowed_data:
            raise ValueError('currently allowed data: %s' % ','.join(allowed_data))
        else:
            self.dataset = dataset

        self.mode = mode
        self.base = os.path.join('/home/mbugueno/TC_graphs/', 'Data_TLGCN', 'data', self.dataset)
        self.current_set = os.path.join(self.base, '%s-%s-stemmed.txt' % (self.dataset, self.mode))
        content, label = self.get_content()

        self.label = self.label_to_onehot(label)
        if vocab is None:
            self.vocab = []

            try:
                self.get_vocab()
            except FileNotFoundError:
                self.build_vocab(content, min_count=5)
        else:
            self.vocab = vocab

        self.d = dict(zip(self.vocab, range(len(self.vocab))))

        self.content = [list(map(lambda x: self.word2id(x), doc.split(' '))) for doc in content]

    def label_to_onehot(self, label_str):
        return [int(lab) for lab in label_str]

    def get_content(self):
        with open(self.current_set) as f:
            all_lines = f.read()
            content = [line.split('\t') for line in all_lines.split('\n')]
            cleaned = content

        label, content = zip(*cleaned)

        return content, label

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['[UNK]']

        return result

    def get_vocab(self):
        with open(os.path.join(self.base, 'vocab-5.txt')) as f:
            vocab = f.read()
            self.vocab = vocab.split('\n')

    def build_vocab(self, content, min_count=10):
        vocab = []

        for c in content:
            words = c.split(' ')
            for word in words:
                if word not in vocab:
                    vocab.append(word)
                    
        freq = dict(zip(vocab, [0 for i in range(len(vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        results = []
        for word in freq.keys():
            if freq[word] < min_count:
                continue
            else:
                results.append(word)

        results.insert(0, '[UNK]')
        with open(os.path.join(self.base, 'vocab-5.txt'), 'w') as f:
            f.write('\n'.join(results))

        self.vocab = results

    def count_word_freq(self, content):
        freq = dict(zip(self.vocab, [0 for i in range(len(self.vocab))]))

        for c in content:
            words = c.split(' ')
            for word in words:
                freq[word] += 1

        with open(os.path.join(self.base, 'freq.csv'), 'w') as f:
            writer = csv.writer(f)
            results = list(zip(freq.keys(), freq.values()))
            writer.writerows(results)

    def batch_iter(self, batch_size, num_epoch):
        for i in range(num_epoch):
            num_per_epoch = int(len(self.content) / batch_size)
            for batch_id in range(num_per_epoch):
                start = batch_id * batch_size
                end = min((batch_id + 1) * batch_size, len(self.content))

                content = self.content[start:end]
                label = self.label[start:end]

                yield content, torch.tensor(label).cuda(), i

def cal_PMI(dataset: str, window_size=20):
    helper = DataHelper(dataset=dataset, mode="train")
    content, _ = helper.get_content()
    pair_count_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    word_count =np.zeros(len(helper.vocab), dtype=int)
    
    
    for doc in content:
        doc = doc.split(' ')
        
        for i, word in enumerate(doc):
            try:
                word_count[helper.d[word]] += 1
            except KeyError:
                continue
                
            start_index = max(0, i - window_size)
            end_index = min(len(doc), i + window_size)
            for j in range(start_index, end_index):
                if i == j:
                    continue
                else:
                    target_word = doc[j]
                    try:
                        pair_count_matrix[helper.d[word], helper.d[target_word]] += 1
                    except KeyError:
                        continue
        
    total_count = np.sum(word_count)
    word_count = word_count / total_count 
        
    pair_count_matrix = pair_count_matrix / total_count
    
    pmi_matrix = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=float)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            try:
                pmi_matrix[i, j] = np.log(pair_count_matrix[i, j] / (word_count[i] * word_count[j]))
            except:
                print ("Error")
    
    pmi_matrix = np.nan_to_num(pmi_matrix)
    
    pmi_matrix = np.maximum(pmi_matrix, 0.0)

    edges_weights = [0.0]
    count = 1
    edges_mappings = np.zeros((len(helper.vocab), len(helper.vocab)), dtype=int)
    for i in range(len(helper.vocab)):
        for j in range(len(helper.vocab)):
            if pmi_matrix[i, j] != 0:
                edges_weights.append(pmi_matrix[i, j])
                edges_mappings[i, j] = count
                count += 1

    edges_weights = np.array(edges_weights)
    edges_weights = edges_weights.reshape(-1, 1)
    edges_weights = torch.Tensor(edges_weights)
    
    return edges_weights, edges_mappings, count
    
    
def gcn_msg(edge):
    return {'m': edge.src['h'], 'w': edge.data['w']}

def gcn_reduce(node):
    w = node.mailbox['w']
    new_hidden = torch.mul(w, node.mailbox['m'])
    new_hidden,_ = torch.max(new_hidden, 1)
    node_eta = torch.sigmoid(node.data['eta'])
    return {'h': new_hidden}


class Model(torch.nn.Module):
    def __init__(self,
                 class_num,
                 hidden_size_node,
                 vocab,
                 n_gram,
                 drop_out,
                 edges_num,
                 edges_matrix,
                 emb_path="../TextLevelGCN/glove.6B.200d.vec.txt", #'bert-base-uncased' if bert initialization is desired
                 max_length=350,  
                 trainable_edges=True,
                 pmi=None,
                 cuda=True
                 ):
        super(Model, self).__init__()

        self.is_cuda = cuda
        self.vocab = vocab
        self.seq_edge_w = torch.nn.Embedding(edges_num, 1)
        self.node_hidden = torch.nn.Embedding(len(vocab), hidden_size_node)
        self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)
        self.edges_num = edges_num
        
        if trainable_edges:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(torch.ones(edges_num, 1), freeze=False)
        else:
            self.seq_edge_w = torch.nn.Embedding.from_pretrained(pmi, freeze=True)

        self.hidden_size_node = hidden_size_node
        self.node_hidden.weight.data.copy_(torch.tensor(self.load_word2vec(emb_path))) 
        self.node_hidden.weight.requires_grad = True
        self.len_vocab = len(vocab)
        self.ngram = n_gram
        self.d = dict(zip(self.vocab, range(len(self.vocab))))
        self.max_length = max_length
        self.edges_matrix = edges_matrix
        self.dropout = torch.nn.Dropout(p=drop_out)
        self.activation = torch.nn.ReLU()
        self.Linear = torch.nn.Linear(hidden_size_node, class_num, bias=True)

    def word2id(self, word):
        try:
            result = self.d[word]
        except KeyError:
            result = self.d['[UNK]']

        return result

    def load_word2vec(self, word2vec_file):
        bert=False
        
        try:
            print ("Loading", word2vec_file )
            model = gensim.downloader.load(word2vec_file)  
        except: 
            bert_tkz = AutoTokenizer.from_pretrained(word2vec_file) 
            model = BertModel.from_pretrained(word2vec_file)
            bert=True
            
        embedding_matrix = []
        
        
        for word in self.vocab:
            try:
                if not bert:
                    try: 
                        embedding_matrix.append(model.get_vector(word))
                    except: 
                        embedding_matrix.append(model.get_vector("unk"))
                else:
                    q=bert_tkz.encode(word)
                    
                    if len(q)==3: 
                        q_emb=model.get_input_embeddings()(torch.tensor(q[1]))
                        q_emb=q_emb.detach().numpy()
                        
                    if len(q)>3: 
                        q_mean=[]
                        for sub_w in q[1:-1]:
                            sub_we= model.get_input_embeddings()(torch.tensor(sub_w))
                            q_mean.append(sub_we.detach().numpy())
                        q_mean=np.asarray(q_mean)
                        q_emb=np.mean(q_mean, axis=0)
            
                    if len(q)<3:
                        unk=bert_tkz.encode('[UNK]')[1]
                        q_unk=model.get_input_embeddings()(torch.tensor(unk))
                        q_emb=q_unk.detach().numpy()
                        
                    embedding_matrix.append(q_emb)
                    
            except KeyError:
                if not bert:
                    embedding_matrix.append(model['the']) 
                else: 
                    unk=bert_tkz.encode('[UNK]')[1]
                    q_unk=model.get_input_embeddings()(torch.tensor(unk))
                    q_unk=q_unk.detach().numpy()
                    if len(q_emb)!=768:
                        print ("Error - dims do not match")
                    embedding_matrix.append(q_unk) 
        embedding_matrix = np.asarray(embedding_matrix) 
        
        if bert: 
            for emb in embedding_matrix:
                if len(emb)!=768:
                    print ("Error - dims do not match")
                
        return embedding_matrix

    def add_all_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []

        local_vocab = list(set(doc_ids))

        for i, src_word_old in enumerate(local_vocab):
            src = old_to_new[src_word_old]
            for dst_word_old in local_vocab[i:]:
                dst = old_to_new[dst_word_old]
                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def add_seq_edges(self, doc_ids: list, old_to_new: dict):
        edges = []
        old_edge_id = []
        for index, src_word_old in enumerate(doc_ids):
            src = old_to_new[src_word_old]
            for i in range(max(0, index - self.ngram), min(index + self.ngram + 1, len(doc_ids))):
                dst_word_old = doc_ids[i]
                dst = old_to_new[dst_word_old]

                edges.append([src, dst])
                old_edge_id.append(self.edges_matrix[src_word_old, dst_word_old])

            edges.append([src, src])
            old_edge_id.append(self.edges_matrix[src_word_old, src_word_old])

        return edges, old_edge_id

    def seq_to_graph(self, doc_ids: list) -> dgl.DGLGraph():  
        
        if len(doc_ids) > self.max_length:
            doc_ids = doc_ids[:self.max_length]

        local_vocab = set(doc_ids)
        old_to_new = dict(zip(local_vocab, range(len(local_vocab))))

        if self.is_cuda:
            local_vocab = torch.tensor(list(local_vocab)).cuda()
        else:
            local_vocab = torch.tensor(list(local_vocab))

        sub_graph = dgl.DGLGraph()
        if self.is_cuda:
            sub_graph = sub_graph.to('cuda')

        sub_graph.add_nodes(len(local_vocab))
        local_node_hidden = self.node_hidden(local_vocab)
        sub_graph.ndata['h'] = local_node_hidden
        seq_edges, seq_old_edges_id = self.add_seq_edges(doc_ids, old_to_new)

        edges, old_edge_id = [], []
        edges.extend(seq_edges)
        old_edge_id.extend(seq_old_edges_id)

        if self.is_cuda:
            old_edge_id = torch.LongTensor(old_edge_id).cuda()
        else:
            old_edge_id = torch.LongTensor(old_edge_id)

        srcs, dsts = zip(*edges)
        sub_graph.add_edges(srcs, dsts)
        try:
            seq_edges_w = self.seq_edge_w(old_edge_id)
        except RuntimeError:
            print("Error", old_edge_id)
        sub_graph.edata['w'] = seq_edges_w
        
        return sub_graph

    def forward(self, doc_ids, is_20ng=None):
        sub_graphs = [self.seq_to_graph(doc) for doc in doc_ids]

        batch_graph = dgl.batch(sub_graphs)
        batch_graph.update_all(
            message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
            reduce_func=dgl.function.max('weighted_message', 'h')
        ) 
        h1 = dgl.sum_nodes(batch_graph, feat='h')
        drop1 = self.dropout(h1)
        act1 = self.activation(drop1)
        l = self.Linear(act1)
        return l


def edges_mapping(vocab_len, content, ngram):
    count = 1
    mapping = np.zeros(shape=(vocab_len, vocab_len), dtype=np.int32)
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram+1)):
                dst = doc[dst_id]

                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                    count += 1

    for word in range(vocab_len):
        mapping[word, word] = count
        count += 1

    return count, mapping


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset, bs):
    data_helper = DataHelper(dataset, mode='dev')

    total_pred = 0
    correct = 0
    iterr = 0
    all_labels=[]
    all_preds=[]
    for content, label, _ in data_helper.batch_iter(batch_size=bs, num_epoch=1):
        iterr += 1
        model.eval()
        logits = model(content)
        pred = torch.argmax(logits, dim=1)
        correct_pred = torch.sum(pred == label)
        correct += correct_pred
        total_pred += len(content)        
        all_labels+=list(label.to('cpu').numpy())        
        all_preds+=list(pred.to('cpu').numpy())

    total_pred = float(total_pred)
    correct = correct.float()
    val_acc= torch.div(correct, total_pred)
    val_f1ma = f1_score(all_labels, all_preds, average=None)
    val_f1ma=np.mean(val_f1ma)
    
    return val_acc, val_f1ma


def test(WANDB_PATH, ngram, embedding_path, model_name, dataset, bs, execn=0):
    
    starttest = time.time()
    model = torch.load(os.path.join(WANDB_PATH, embedding_path, dataset, str(ngram), model_name  + "_"+ str(execn) +'.pkl'))

    data_helper = DataHelper(dataset, mode='test')

    total_pred = 0
    correct = 0
    iterr = 0
    all_labels=[]
    all_preds=[]
    for content, label, _ in data_helper.batch_iter(batch_size=bs, num_epoch=1):
        iterr += 1
        model.eval()

        logits = model(content)
        pred = torch.argmax(logits, dim=1)
        correct_pred = torch.sum(pred == label)
        correct += correct_pred
        total_pred += len(content)        
        all_labels+=list(label.to('cpu').numpy())        
        all_preds+=list(pred.to('cpu').numpy())
        
    total_pred = float(total_pred)
    correct = correct.float()
    acc=torch.div(correct, total_pred).to('cpu')
    my_f1=f1_score(all_labels, all_preds, average=None)
    my_f1ma=np.mean(my_f1)
    
    endtest = time.time()
    total_timetest = endtest - starttest    
    return acc, my_f1, my_f1ma, total_timetest

def train(WANDB_PATH, project_name, ngram, name, bar, drop_out, embedding_path, dim_emb, dataset, bs, execn=0, n_epochs=200, bert=False, is_cuda=False, edges=True):
    if bert: 
        dim_emb = 768
        
    import wandb    
    wandb.init(project=project_name, name=embedding_path+" "+dataset+"_n"+str(ngram)+"_run"+str(execn))

    
    print('Loading data helper in training mode.')
    data_helper = DataHelper(dataset, mode='train')

    if os.path.exists(os.path.join(WANDB_PATH, embedding_path, dataset, str(ngram), name+'.pkl')) and name != 'temp_model':
        print('Loading model from file.')
        model = torch.load(os.path.join(WANDB_PATH, embedding_path, dataset, str(ngram), name+'.pkl'))
    else:
        print('Creating a new model.')
        startc = time.time()
        if name == 'temp_model':
            name = 'temp_model_%s' % dataset
            
        edges_weights, edges_mappings, count = cal_PMI(dataset=dataset)        
        model = Model(class_num=len(set(data_helper.label)), hidden_size_node=dim_emb, vocab=data_helper.vocab, 
                      n_gram=ngram, drop_out=drop_out, edges_matrix=edges_mappings, emb_path=embedding_path, 
                      edges_num=count, trainable_edges=edges, pmi=edges_weights, cuda=is_cuda)
        
        endc = time.time()
        total_timec = endc - startc
        
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ("Model parameters", pytorch_total_params )
    
    
    starttrain = time.time()
    if is_cuda:
        print('\nCuda available')
        model.cuda()
    loss_func = torch.nn.CrossEntropyLoss() 
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)

    iterr = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_acc = 0.0
    best_f1ma = 0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total = 0
    
    wandb.watch(model)
                
    for content, label, epoch in data_helper.batch_iter(batch_size=bs, num_epoch=n_epochs):
        improved = ''
        model.train()

        logits = model(content)
        loss = loss_func(logits, label)
        pred = torch.argmax(logits, dim=1)
        correct = torch.sum(pred == label)
        total_correct += correct
        total += len(label)
        total_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc = float(total_correct) / float(total)
                
        iterr += 1
        if bar:
            pbar.update()
        if iterr % (NUM_ITER_EVAL) == 0:            
            wandb.log({"loss": loss, "acc":acc})
                        
            if bar:
                pbar.close()

            val_acc, val_f1ma = dev(model, dataset, bs)                 
            if val_f1ma > best_f1ma:
                best_f1ma = val_f1ma
                last_best_epoch = epoch
                improved = '*'

                torch.save(model, WANDB_PATH + embedding_path + "/" + dataset + "/" + str(ngram) + "/" + name + "_"+ str(execn) +'.pkl')

            if epoch - last_best_epoch >= EARLY_STOP_EPOCH:
                endtrain = time.time()
                total_timetrain = endtrain - starttrain
                return name, total_timec ,total_timetrain
            
            wandb.log({"val_F1ma":val_f1ma , "val_acc":val_acc})
            
            
            msg = 'Epoch:{0:>4} -- Iter:{1:>6}, Train Loss:{5:>6.2}, Train Acc:{6:>6.2%}' \
                  + ' Val Acc:{2:>6.2%}, Val_F1ma:{7:>6.2%}, Time:{3}{4}' \
               
            print(msg.format(epoch, iterr, val_acc, get_time_dif(start_time), improved, total_loss/ NUM_ITER_EVAL,
                             float(total_correct) / float(total), val_f1ma))

            total_loss = 0.0
            total_correct = 0
            total = 0
            if bar:
                pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    endtrain = time.time()
    total_timetrain = endtrain - starttrain
    
    return name, total_timec, total_timetrain 
