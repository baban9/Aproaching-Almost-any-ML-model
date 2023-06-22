import io 
import torch 

import numpy as np 
import pandas as pd 

import tensorflow as tf 
from sklearn import metrics 
 
import config 
import dataset
import engine 
import lstm 

def load_vectors(fname):
    # taken from: https://fasttext.cc/docs/en/english-vectors.html
    fin = io.open(
                fname,
                'r',
                encoding='utf-8',
                newline='\n',
                errors='ignore'
                )
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix

def run(df, fold, embedding_dict):
    print(f" For : {fold}")
    train_df = df[df.kfold != fold].reset_index(drop=True)
    valid_df = df[df.kfold == fold].reset_index(drop=True) 

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df.review.values.tolist())

    xtrain = tokenizer.texts_to_sequences(train_df.review.values)
    xtest = tokenizer.texts_to_sequences(valid_df.review.values)
    xtrain = tf.keras.preprocessing.sequence.pad_sequences(xtrain, maxlen=config.MAX_LEN)
    xtest = tf.keras.preprocessing.sequence.pad_sequences(xtest, maxlen=config.MAX_LEN)
    
    print("loading the data")
    train_dataset = dataset.IMDBDataset(
                                    reviews=xtrain,
                                    targets=train_df.sentiment.values
                                    )
    train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config.TRAIN_BATCH_SIZE)
    
    valid_dataset = dataset.IMDBDataset(
                                    reviews=xtest,
                                    targets=valid_df.sentiment.values
                                    )
    valid_data_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.VALID_BATCH_SIZE)
    
    
    embedding_matrix = create_embedding_matrix(
                        tokenizer.word_index, embedding_dict
                        )
    device = torch.device("cpu")
    model = lstm.LSTM(embedding_matrix)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_accuracy = 0
    early_stopping_counter = 0
    for epoch in range(config.EPOCHS):
        engine.train(train_data_loader, model, optimizer, device)
        outputs, targets = engine.evaluate(
                    valid_data_loader, model, device
                    )
        
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(
        f"FOLD:{fold}, Epoch: {epoch}, Accuracy Score = {accuracy}"
        )
        # simple early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        else:
            early_stopping_counter += 1
        if early_stopping_counter > 2:
            break

if __name__ == "__main__":
    # load data
    df = pd.read_csv("datasets/imdb_folds.csv")
    print("Loading embeddings")
    embed_dict = load_vectors(config.EMBED_VEC)
    # train for all folds
    run(df, fold=0, embedding_dict= embed_dict)
    run(df, fold=1, embedding_dict= embed_dict)
    run(df, fold=2, embedding_dict= embed_dict)
    run(df, fold=3, embedding_dict= embed_dict)
    run(df, fold=4, embedding_dict= embed_dict)