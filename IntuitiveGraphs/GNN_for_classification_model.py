import os
import wandb
import time
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

from graph_utils import *
from torchmetrics import F1Score
from torch.nn import Linear
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool 
from torch import optim, nn, utils, Tensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


class GNN_LightingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Train_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1])) 
        return loss["loss"]
    
    def validation_step(self, batch, batch_idx):
        loss = self.forward_performance(batch) 
        for k, v in loss.items():
            self.log("Val_" + k, v, prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch[1]))  
        return loss["loss"]

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr) #1e-3
        return optimizer
    
    def forward_performance(self, data):        
        try: 
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.batch.to(self.device))  
        except: 
            out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None, data.batch.to(self.device))  
        
        loss = self.criterion(out, data.y)
        pred = out.argmax(dim=1) 
        acc=(pred == data.y).sum()/len(data.y)
        f1_score = F1Score(num_classes=self.num_classes, average="macro").to(self.device)
        f1_ma=f1_score(pred, data.y)
        return {"loss": loss, "f1ma":f1_ma, 'acc':acc} 
    
    def predict(self, loader, cpu_store=True):
        self.eval()
        preds=[]
        for data in loader:  
            try: 
                out = self(data.x.float().to(self.device), data.edge_index.to(self.device), data.edge_attr.to(self.device), data.batch.to(self.device))  
            except:
                out = self(data.x.float().to(self.device), data.edge_index.to(self.device), None, data.batch.to(self.device))  
                
            pred = out.argmax(dim=1) 
            if cpu_store:
                pred = pred.detach().cpu().numpy() 
            preds+=list(pred) 
        self.train() 
        if not cpu_store:
            preds = torch.Tensor(preds)
        return preds
    
    
class GAT_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, num_heads=4, dropout=False, criterion = torch.nn.CrossEntropyLoss()):
        print ("Creating GAT model")
        super(GAT_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()        
        self.convs.append(GATConv(in_features, num_hidden_conv, heads=num_heads))
        for l in range(num_layers-1):
            self.convs.append(GATConv(num_hidden_conv * num_heads, num_hidden_conv, heads=num_heads))

        self.lin = Linear(num_hidden_conv * num_heads, out_features)
        self.criterion = criterion
        
        
    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr==None: 
                x = self.convs[i](x, edge_index)
            else: 
                x = self.convs[i](x, edge_index, edge_attr.float())
            
            emb = x
            x = F.relu(x)  
            if self.dropout!=False:
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

    
class GCN_model(GNN_LightingModel):
    def __init__(self, in_features, num_hidden_conv, out_features, num_layers, lr, dropout=False, criterion = torch.nn.CrossEntropyLoss()):
        print ("Creating GCN model")
        super(GCN_model, self).__init__()
        self.num_classes = out_features
        self.dropout = dropout
        self.lr = lr
        self.num_layers = num_layers
        self.convs = nn.ModuleList()        
        self.convs.append(GCNConv(in_features, num_hidden_conv))
        for l in range(num_layers-1):
            self.convs.append(GCNConv(num_hidden_conv, num_hidden_conv))

        self.lin = Linear(num_hidden_conv, out_features)
        self.criterion = criterion

    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr==None: 
                x = self.convs[i](x, edge_index)
            else: 
                x = self.convs[i](x, edge_index, edge_attr.float())
                
            emb = x
            x = F.relu(x)
            if self.dropout!=False:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return x    
    

class GIN_model(GNN_LightingModel):
    def __init__(self, input_dim, hidden_dim, out_features, num_layers, lr, dropout, with_edges_attr = False):
        super(GIN_model, self).__init__()
        if with_edges_attr:
            print ("Creating GINE model")
        else: 
            print ("Creating GIN model")
        self.with_edges_attr = with_edges_attr
        self.num_classes = out_features
        self.lr = lr
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(num_layers-2):
            self.lns.append(nn.LayerNorm(hidden_dim))
        self.num_layers = num_layers
        self.dropout = dropout 
        for l in range(num_layers-1):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(self.dropout), 
            nn.Linear(hidden_dim, out_features))
        

    def build_conv_model(self, input_dim, hidden_dim):
        if self.with_edges_attr:
            return pyg_nn.GINEConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                 nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)), edge_dim=1) 
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        return 
 
    def forward(self, x, edge_index, edge_attr, batch):
        for i in range(self.num_layers):
            if edge_attr==None: 
                x = self.convs[i](x, edge_index)
            else: 
                x = self.convs[i](x, edge_index, edge_attr.float())                
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
                
        x = pyg_nn.global_mean_pool(x, batch)
        x = self.post_mp(x)
        return F.log_softmax(x, dim=1)

    def criterion(self, pred, label):
        return F.nll_loss(pred, label)    
    
    
def partitions(dataset, dataset_test, bs, trainp=0.8, valp=0.2):
    if np.round(trainp+valp)!=1.0:
        print ("Partitions don't fit. Please specify partitions that sum 1.")
        return 
    
    dataset = dataset.shuffle()
    total=len(dataset)
    a=int(total*trainp)
    train_dataset = dataset[:a]
    val_dataset = dataset[a:]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1) 
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=1) 
    test_loader = DataLoader(dataset_test, batch_size=bs, shuffle=False, num_workers=1)
            
    return train_loader, val_loader, test_loader


def run_bunch_experiments(dataset, dataset_test, path_models, path_results, n_layers, dim_features, file_to_save, type_model, lr, dropout, bs, project_name, with_edges_attr, bunch=10, pat=10, ep=100, progress_bar=False):
    
    start = time.time()
    np.set_printoptions(precision=3)
    
    train_loader, val_loader, test_loader = partitions(dataset, dataset_test, bs)
    
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes 
    
    with open(path_results+file_to_save+'.txt', 'a') as f:
    
        for nl in n_layers:

            for dim in dim_features:
                
                print ("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim)
                print ("\nTRAINING MODELS SETTING #LAYERS:", nl, " HIDDEN DIM:", dim, file=f)
                acc_tests=[]
                f1_tests=[]
                f1ma_tests=[]
                for i in range(bunch):    
                    starti = time.time()
                    
                    if type_model=="GAT":
                        model = GAT_model(num_node_features, dim, num_classes,  nl, lr, dropout=dropout)    
                    
                    elif type_model=="GIN":
                        model = GIN_model(num_node_features, dim, num_classes, nl, lr, dropout=dropout,
                                          with_edges_attr=with_edges_attr)
                        
                    elif type_model=="GCN":  
                        model = GCN_model(num_node_features, dim, num_classes,  nl, lr, dropout=dropout)  
                    
                    else: 
                        print ("Type model error: No GNN was intended")                      
                        return 
                        
                    early_stop_callback = EarlyStopping(monitor="Val_f1ma", 
                                                        mode="max", min_delta=1e-3, patience=pat, verbose=True)
                    wandb_logger = WandbLogger(name=type_model+"Model_L"+str(nl)+"_U"+str(dim), 
                                               save_dir=path_models+file_to_save, project= project_name)
                    trainer = pl.Trainer(max_epochs=ep, accelerator='gpu', devices=1, 
                                         callbacks=[early_stop_callback],
                                         logger=wandb_logger, 
                                         enable_progress_bar=progress_bar) #gpus=1, 
                    trainer.fit(model, train_loader, val_loader)

                    ###### TESTING
                    print ("\n----------- Evaluating model "+str(i)+"-----------\n")
                    print ("\n----------- Evaluating model "+str(i)+"-----------\n", file=f)
                    print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch)
                    print ("\nTraining stopped on epoch:", trainer.callbacks[0].stopped_epoch, file=f)
                    
                    preds=model.predict(test_loader, cpu_store=False).int()

                    trues=[]
                    for data in test_loader: 
                        trues.append(data.y)
                    trues = torch.concat(trues)

                    acc=(trues ==preds).float().mean() 
                    f1_score = F1Score(num_classes, average=None)
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
                print ("RESULTS FOR N_LAYERS:", nl, " HIDDEN DIM_FEATURES:", dim, file=f)    
                print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)), file=f)
                print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)), file=f)
                print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0), file=f)
                print ("************************************************\n\n", file=f)
                print ("\n************************************************")
                print ("RESULTS FOR N_LAYERS:", nl, " HIDDEN DIM_FEATURES:", dim)   
                print ("Test Acc: %.3f"% np.mean(np.asarray(acc_tests)), "-- std: %.3f" % np.std(np.asarray(acc_tests)))
                print ("Test F1-macro: %.3f"%np.mean(np.asarray(f1ma_tests)), "-- std: %.3f" % np.std(np.asarray(f1ma_tests)))
                print ("Test F1 per class:", np.mean(np.asarray(f1_tests), axis=0))
                print ("************************************************\n\n")
        
        f.close()

    end = time.time()
    total_time = end - start
    print("\nRunning time for all the experiments: "+ str(total_time))
    
    return 