#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:10:07 2020

@author: hadoopuser
"""

from transformers import BertTokenizer,BertModel,AdamW,get_linear_schedule_with_warmup
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

import unicodedata

  
import re

import pickle

from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#from pylab import rcParams
#import matplotlib.pyplot as plt
#import seaborn as sns

#from matplotlib import rc

class_names=["P","E","A"]  
n_classes= 3


    
def textclean(data):
   # print(data[0])
    text=data[1]
    entity=data[2]
    text=text.lower()
    text= re.sub(r'[\']',"",text)
    text=re.sub(r'\W'," ",text)
    textlst=text.split(" ")
    words = [unicodedata.normalize('NFKD', str(w)).encode('ascii','ignore').decode("utf-8") for w in textlst]
    lenword=len(words)
    #entity_lst=["{0}-{1}".format("I",entity) if i>0 else"{0}-{1}".format("B",entity) for i in range(lenword)] 
    if entity=="P":
        
        entity_lst=0
        
    else:
        entity_lst=1
        
    newtext=" ".join(words)
    finaldf=pd.Series([newtext,words, entity_lst])
   
    return finaldf



filename='/home/hadoopuser/Desktop/project/transformers/sanctionlist.csv'
filename="sanctionlist.csv"
def tragetProcess(data):
   
    tags = list(set((data["Tag"].values)))
    n_tags = len(tags)
    # Creating words to indices dictionary.
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: w for w, i in tag2idx.items()}
#    sentences = get_tagged_sentences(data)
    data["Tag1"]=data["Tag"]
    target_sent =[data["Tag"][i] for i in range(len(data))]
    y = [tag2idx[ str(s)]for s in target_sent]
    data["Tag"]=y
        
    return data,n_tags,tag2idx,idx2tag

def loadData(filename):
        file_df=pd.read_csv(filename, encoding="unicode_escape")
        file_df = file_df.sample(frac=1).reset_index(drop=True).dropna()
        
        file_df.reset_index(inplace=True)
              
        file_df[["CleanName",'Word','Tag']] = file_df.apply(textclean,axis=1)
       # data=file_df[["index","Word","Tag"]]
#        dt=data.head(50)
#        print(dt)
        print("Data Loaded Successfully")
        return(file_df)
        


  #require Internet

file_df=loadData(filename)
#file_df,_,_,_=tragetProcess(file_df)
#file_df=file_df[["CleanName","Tag"]]

query_data_train=file_df["CleanName"].values.tolist()
#text_str=[bert_tokenizer_transformer.encode(txt) for txt in query_data_train]
#
#len_str=[len(txt) for txt in text_str]
#
#maxlen=max(len_str)
#
#import numpy as np
#
#np.mean(len_str)
#
#np.median(len_str)


class Dataset(Dataset):
    def __init__(self,x,y,bert_tokenizer,maxlen):
        self.x=x
        self.y=y
        self.bert_tokenizer=bert_tokenizer
        self.maxlen=maxlen
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        sent=self.x[index]
        target=self.y[index]
        
        encoding=self.bert_tokenizer.encode_plus(
                                                  sent,
                                                  max_length=self.maxlen,
                                                  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                                  padding='max_length',
                                                  return_tensors='pt',  # Return PyTorch tensors
                                                  truncation=True
                                                )
        item={}
        target=target.flatten()
        item["sent"]=str(sent)
        item["target"]=torch.tensor(target,dtype=torch.float)
        item["input_id"]=encoding["input_ids"].flatten()
        item["attention_mask"]=encoding["attention_mask"].flatten()
        
        return item
    

bert_tokenizer_transformer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
maxlen=32
def createData(df):
    x_torch=df["CleanName"].values
    y_torch=df["Tag"].values
    bert_tokenizer=bert_tokenizer_transformer
    
    
    dataset=Dataset(x_torch,y_torch,bert_tokenizer,maxlen)
    
    loaddt=DataLoader(dataset=dataset,batch_size=32, shuffle=True)
    return loaddt

df_train, df_test = train_test_split(file_df,test_size=0.1,random_state=42)
df_val, df_test = train_test_split(df_test,test_size=0.5,random_state=42)


df_train.shape, df_val.shape, df_test.shape

traindt=createData(df_train)
testdt=createData(df_test)
valdt=createData(df_val)


traindf=next(iter(traindt))


#bert_model = BertModel.from_pretrained('bert-base-uncased')

#hidden_state,output=bert_model(input_ids=encoding["input_ids"],attention_mask=encoding["attention_mask"])
#bert_model.config.hidden_size



class BertModelClassify(nn.Module):
    def __init__(self,n_classes):
         super(BertModelClassify,self).__init__()
         self.bert=BertModel.from_pretrained('bert-base-uncased',return_dict=False)
         self.dropout=nn.Dropout(0.2)
         self.linearlayer=nn.Linear(self.bert.config.hidden_size,1)
    
    def forward(self,input_ids,attention_mask):
        output=self.bert(input_ids=input_ids,
                                      attention_mask=attention_mask)
        hidden_state = output[0]  # (bs, seq_len, dim)
        output = hidden_state[:, 0]
        #hidden_state = output[0]  # (bs, seq_len, dim)
        #output = hidden_state[:, 0]
       # print(hidden_state)
        
        output=self.dropout(output)
        output=self.linearlayer(output)
        return output
    
model=BertModelClassify(n_classes)
optimizer=AdamW(model.parameters(),lr=2e-5,correct_bias=False)

EPOCHS=1
total_steps = len(traindt) * EPOCHS


scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
#loss_fn = nn.CrossEntropyLoss()
loss_fn = torch.nn.BCEWithLogitsLoss()      
##################TRAINING###########################################

def trainBert(model,traindt,loss_fn,optimizer,scheduler):
    losses = []
    correct_predictions = 0
    model=model.train()
    
    for dt in traindt:
        inputids=dt["input_id"]
        target=dt["target"]
        attention_mask=dt["attention_mask"]
        output=model(inputids,attention_mask)
        _, preds = torch.max(output, dim=1)
        
        
        lss=loss_fn(output,target)
        correct_predictions += torch.sum(preds == target)
        losses.append(lss.item() )     
        lss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        train_acc=correct_predictions.double() / len(df_train)
        train_loss=np.mean(losses)
        print("train_acc:",train_acc,"train_loss:",train_loss)
        
        
        
    return train_acc,train_loss
        

def evaluateBert(model,valdt,loss_fn):
    losses = []
    correct_predictions = 0
    model=model.eval()
    
    with torch.no_grad():
        for dt in valdt:
            inputids=dt["input_id"]
            target=dt["target"]
            attention_mask=dt["attention_mask"]
            output=model(inputids,attention_mask)
            _, preds = torch.max(output, dim=1)
             
            lss=loss_fn(output,target)
            correct_predictions += torch.sum(preds == target)
            losses.append(lss.item() )     
            
            val_acc=correct_predictions.double() / len(valdt)
            val_loss=np.mean(losses)
            
            
        
    return val_acc,val_loss


best_accuracy=0        
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_acc,train_loss=trainBert(model,traindt,loss_fn,optimizer,scheduler)
    
    print('Train loss {0} accuracy {1}'.format(train_loss,train_acc))
    
    val_acc,val_loss=evaluateBert(model,valdt,loss_fn)
    
    print('Val loss {0} val accuracy {1}'.format(val_loss,val_acc))
    
    if val_acc>best_accuracy:
        best_accuracy=val_acc
        torch.save(model.state_dict(), 'best_model_state.bin')
