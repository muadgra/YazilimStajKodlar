# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 10:46:56 2021

@author: Mertcan
"""
import random
import json
import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32)
    
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


with open("C:/Users/Mertcan/Desktop/mydata/nlp/intents.json", 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        #don't want an array of arrays, so I don't use append
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '!', '.', ',', '...']
all_words = [stem(w) for w in all_words if w not in ignore_words]

#set is used for achieving all unique words and tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#26 sentences, 54 words
X_train = []
#26 tags, same as sentences
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) #not one hot encoded, cross entropy doesn't care in pytorch

X_train = np.array(X_train)
y_train = np.array(y_train)
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
        
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
    
    def __len__(self):
        return self.n_samples

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        #no activation
        return out

#hyperparameters
batch_size = 8
learning_rate = 0.001
num_epochs = 1000
#they all have all size
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)


dataset = ChatDataset()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        labels = labels.to(dtype=torch.long)
        #forwards pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss = {loss.item():.4f}') 


data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "all_words": all_words,
            "tags": tags
        }
f = "data.pth"
torch.save(data, f)
print(f"Training complete. File saved to {f}")





