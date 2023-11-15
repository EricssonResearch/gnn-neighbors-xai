# GNN Neighbors XAI
This if the official implementation of [Evaluating Neighbor Explainability for Graph Neural Networks](https://arxiv.org/abs/2311.08118)).

## Structure

All the code of this repo is inside the src folder. This folder contains a utils file for generic functions and classes and two modules, train and explain
* train: This module contains all the code for training the models. It contains the following files:
  - models.py: This file contains the different models used.
  - utils.py: This file contains functions and classes that will be needed during the training process.
  - main.py: This file contains the main function for training.<br/><br/>


* explain: This module contains all the code for computing explanations and the metrics presented in the paper. It contains the following files:
  - methods.py: This file contains the different explainability methods used.
  - algorithms.py: This file contains an auxiliary class for the computation of PGExplainer method.
  - main.py: This file contains the main function for computing the metrics results over all methods.
  - examples.ipynb: This file contains two examples of visualizations of the explanations of two explainability methods.

## Dependencies

Are listed in the requirements file. For installing them just run:

```
pip install -r requirements.txt
```


## How to run the experiments

To replicate the experiments the first thing would be training the models. For that we will run the train.py file of the train module the following way (treating the file as a module too):

```
python -m src.train.main
```

For the training there are three variables that we can modify at the beginning of the main function:

- dataset_name: It can be Cora, CiteSeer or PubMed. This will choose what dataset will be used for training. 
- model_name: It can be gcn or gat. The model will be a two layer GNN, and the type of GNN can be GCN or GAT, depending on this variable.
- self_loops: If the model uses self loops

Once the training has finished, the program will save the model in a models folder that will be created once the training starts. The name for saving will be based on the three variables explained before.

The second step will be computing the metrics results running the main.py file in the explain module the following way:

```
python -m src.explain.main
```

At the beginning of this main function the same three variables are present. Therefore, here the model trained before will be loaded. Note that if a specific combination of the three variables has not been trained, the main function of the explain module will throw an error. This main function will create a results folder to save the results. In these results folder there will be the following sub-folders:

- graphs: where all the visualizations of the graphs will be saved.
- aucs: where the results of the area under the curve of the *loyalty* and the *inverse loyalty* will be saved.
- aucs_probs: where the results of the area under the curve of the *loyalty probabilities* and the *inverse loyalty probabilities* will be saved.
- last_loyalties: where the results of the *loyalty* and the *inverse loyalty* once all the important neighbors have been deleted.
- last_loyalties_probs: where the results of the *loyalty probabilities* and the *inverse loyalty probabilities* once all the important neighbors have been deleted.


