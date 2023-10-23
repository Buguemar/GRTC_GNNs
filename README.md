# GRTC_GNNs
Public repository of our paper accepted to the Findings of EMNLP 2023: *Graph Representations for Text Classification Using GNNs: Exploring Advantages and Limitations.*


## Training 
### Graph Models
For training a GNN for Intuitive Graph constructions:
```
python train_GNN.py -s config/GNNClassifier_example.yaml
```
For training TextLevelGCN:
```
python train_tlgcn.py -s config/tlgcn_example.yaml
```
### Baselines 
For training a Transformer-based LM:
```
python train_language.py -s config/longformer_example.yaml
```
For training BOW MLP:
```
python train_bow_mlp.py -s config/bow_mlp_example.yaml
```
Note that all the config ```yaml``` files are provided as a mere example. You can set the corresponding model hyperparameters as you need.   

## Paper Results
The results reported in our EMNLP paper can be found in the corresponding sub-folder ```Results Paper```.

## Requirements
Please install the required packages with the following command:
```
pip install -r requirements.txt
```
