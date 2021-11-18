#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pandas as pd
import json

from sklearn import ensemble
from sklearn.externals import joblib


# In[ ]:


#Define the Parameter in envirorment variable 


# In[7]:


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters used for Training
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=5)
    
    # path for  Input, output and model directory 
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    # Setting variable to host the model
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


# In[ ]:


# Load data


# In[6]:


def load_data(file_path, channel):
    input_files = [ os.path.join(file_path, file) for file in os.listdir(file_path) ]
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    df = pd.concat(raw_data)    
    
    features = df.iloc[:,1:].values
    label = df.iloc[:,0].values
   
    return features, label


# In[ ]:


# Define Model


# In[5]:


def model(args, x_train, y_train, x_test, y_test):   
    model = ensemble.RandomForestClassifier(n_estimators=args.n_estimators,max_depth=args.max_depth)
    model.fit(x_train, y_train)
    
    print("Training Accuracy: {:.3f}".format(model.score(x_train,y_train)))
    print("Testing Accuracy: {:.3f}".format(model.score(x_test,y_test)))
    
    return model



def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    classifier = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("###### Uploaded model########")
    return classifier



# In[ ]:


# Main Function


# In[8]:


if __name__ == '__main__':
    print("####Starting Main Program#########")
    args, unknown = parse_args()
    print("####ARG#########",args)
    train_data, train_labels = load_data(args.train,'train')
    print("#######Successfully Loaded Train")
    eval_data, eval_labels = load_data(args.test,'test')
    print("####### Test Successfully Loaded")
    
    print("####### Model Started Training")
    classifier = model(args, train_data, train_labels, eval_data, eval_labels)
    joblib.dump(classifier, os.path.join(args.model_dir, "model.joblib"))


# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]: