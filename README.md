# DATING ITALIAN DOCUMENTS USING BERT

Natural Language Processing course project 

## How to use the software
1. Download the dataset from [the DaDoEval competition website](https://dhfbk.github.io/DaDoEval)
2. Put the training set, test set, and gold file folder into the `data` folder in this repository. 
3. Run the preprocessing scripts into `src/utils` to preprocess the data. Example (from the `src` folder): `python -m utils.preprocess_train_data`
4. Train the model with `python train.py`. The default strategy is Umberto with truncation of the first 512 tokens and embedding derived from the sum of the embedding of the [CLS] token of the last four layers. 

## Notes
- By now the change of parameters is manual and embedded in the code. I'll provide an automatic way of training a given model without directly changing the code. 
- A GPU is not required but recommended. 
- On a NVidia Titan X GPU the training took about 10 minutes. 

## Details 
If you want to know more details about the project and the tests, have a look at the report here. 
