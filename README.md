# GOAL_Graph_Complementary_Learning_ICML2023
The Open Source Code For ICML 2023 Paper "Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs"
![goal](https://github.com/zyzisastudyreallyhardguy/GOAL-Graph-Complementary-Learning/assets/75228223/51f8538b-f4ce-4fed-b6fb-95cc35aabec0)

The arxiv link for our paper is [https://arxiv.org/submit/4952503 ](https://arxiv.org/abs/2306.07608)


# Overview
Our implementation of Graph Complementary Learning(GOAL)is based on pytorch. 
For finding the missing-half topology for a dataset, you need to run graph_complement.py first. Then, you can run goal_conv.py to train on the complemented graph.

Actor 
```
python goal_complement.py --dataset 'actor' --graph_gen --k_heter 100 --dataset 'actor' --batch_test --num_layers_gen 2 --epochs_gen 30
python goal_conv.py --dataset 'actor' --alpha 1.5 --beta 0 --gamma 3.5 --delta 2 --epochs 50 --homo_gen
```

Chameleon
```
python goal_complement.py --dataset 'chameleon' --use_gnn_high --graph_gen --epochs_gen 50 --num_layers_gen 5 --k_homo 10 --batch_test --hidden 20 --num_layers_gnn 2 --pretrain
python goal_conv.py --alpha 0 --beta 2 --gamma 2 --delta 0 --epochs 200 --patience 200 --dataset 'chameleon' --homo_gen
```

CiteSeer
```
python goal_complement.py --pretrain --graph_gen --k_heter 100 --dataset 'citeseer'
python goal_conv.py --dataset 'citeseer' --alpha 1 --beta 0.5 --gamma 1.5 --delta 0.5 --epochs 100
```

Computers
```
python goal_complement.py --pretrain --graph_gen --k_heter 100 --dataset 'computers' --epochs_gnn 1000 --batch_test --num_layers_gen 4 --num_layers_gen 2 --hidden 50
python goal_conv.py --dataset 'computers' --alpha 2 --beta 4 --gamma 0 --delta 1 --epochs 250
```

Photo
```
python goal_complement.py --pretrain --graph_gen --k_heter 100 --dataset 'photo' --epochs_gnn 1000 --batch_test --num_layers_gen 4
python goal_conv.py --dataset 'photo' --alpha 1.5 --beta 1.5 --gamma 2.5 --delta 1 --epochs 400
```

PubMed
```
python goal_complement.py --pretrain --graph_gen --k_heter 1 --dataset 'pubmed' --batch_test --num_layers_gen 2 --hidden 50 --patience 200 --epochs_gnn 500 --epochs_gen 100
python goal_conv.py --dataset 'pubmed' --alpha 2.5 --beta 2.5 --gamma 0.5 --delta 0 --epochs 700 --hid_units 256 --n_layers 2
```

Squirrel
```
python goal_complement.py --dataset 'squirrel' --use_gnn_high --graph_gen --epochs_gen 100 --num_layers_gen 5 --k_homo 20 --batch_test --hidden 50 --pretrain
python goal_conv.py --alpha 0 --beta 5 --gamma 4 --delta 0 --epochs 500 --dataset 'squirrel' --homo_gen
```


## Requirement


## Reference
```
@inproceedings{zheng2023finding,
  title={Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs},
  authors={Zheng, Yizhen and Zhang, He and Lee, Vincent and Zheng, Yu and Wang, Xiao and Pan, Shirui},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}}
```

## License
```
MIT
```
