import string
import re
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import src.const as const

class Model():
    def __init__(self):
        pass
    
    def preprocessor(self, text):
        translate_table = dict((ord(char), None) for char in string.punctuation)
        text = text.lower()
        text = re.sub(r'\d+','',text)
        text = re.sub(r'https\S+','',text)
        text = " ".join(text.split())
        text = text.translate(translate_table)
        return text

    def train(self):
        df = pd.read_csv(os.path.join(os.getcwd(),const.INPUT_FILE))
        df['Clean_Text'] = df['Text'].apply(self.preprocessor)
        X = df['Clean_Text']
        y = df['Language']

        X_train, _ , y_train, _ = train_test_split(X, y, test_size= 0.2, stratify= y, random_state= const.RANDOM_STATE)

        vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')
        pipe_lr = Pipeline([
            ('vectorizer', vectorizer),
            ('clf', LogisticRegression(random_state = const.RANDOM_STATE))
        ])
        pipe_lr.fit(X_train,y_train)

        lr_file = open(const.MODEL_FILE, 'wb')
        pickle.dump(pipe_lr, lr_file)
        lr_file.close()
    
    def get_random(self):
        df = pd.read_csv(const.INPUT_FILE)
        df['Clean_Text'] = df['Text'].apply(self.preprocessor)
        X = df['Clean_Text']
        y = df['Language']

        _, X_test, _, y_test = train_test_split(X, y, test_size= 0.2, stratify= y, random_state= const.RANDOM_STATE)
        random_index = np.random.randint(0, X_test.shape[0]+1) 
        text = X_test.iloc[random_index]
        label = y_test.iloc[random_index]
        
        return text, label                                                       
