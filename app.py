# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import division, print_function
import jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'model.h5'

model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()          


def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

st_words = stopwords.words('english')

# Function to preprocess the name

def name_process(text):             
    text = decontracted(text)
    text = re.sub("[^A-Za-z0-9 ]","",text)  
    text = text.lower()
    text =  " ".join([i for i in text.split() if i not in st_words])
    if len(text)==0:
        text = "missing"
    return text

# Function to preprocess the "brand_name"

def brand_process(text):
    text = re.sub("[^A-Za-z0-9 ]","",text)
    text = text.lower() 
    return text

# Assigning the score correrponding to each "brand_name" equals to number of occurences for that brand
df = pd.read_csv('mercari_train.csv')
brand_score = dict(df[df.brand_name.notnull()]["brand_name"].apply(brand_process).value_counts())

def brand_name_processed_func(brand_name, name_processed):
    if(len(brand_name)!=0):
        return brand_name
    words = name_processed.split()
    score = []
    for j in words:
        if j in brand_score.keys():  #if the words in name is present in the keys of brand score dict
            score.append(brand_score[j]) 
        else:                        #if the word is not a brand name
            score.append(-1)
                
    if max(score) > 0: 
        return words[score.index(max(score))]
    else:                           # no brand name was found
        return "missing"

#Preprocessing the "category_name"

def category_name_preprocessing(text):
    if(len(text)==0):
        return "missing"
    text = re.sub("[^A-Za-z0-9/ ]","",text)
    text = re.sub("s "," ",text) 
    text = re.sub("s/","/",text)
    text = re.sub("  "," ",text)
    text = text.lower()
    return text



def tier_2_func(category_name_preprocessed):
    x = category_name_preprocessed
    if(len(x.split("/"))>1):
        return x.split("/")[1]
    return "missing"
    

def tier_3_func(category_name_preprocessed):
    x = category_name_preprocessed
    if(len(x.split("/"))>1):
        return x.split("/")[2]
    return "missing"


#Preprocessing "item_description"

def processing_item_description(text):
    text = re.sub("\[rm\] ","",str(text))
    text = decontracted(text)
    text = re.sub("[^A-Za-z0-9 ]","",str(text))
    text = str(text).lower()
    text =  " ".join([i for i in text.split() if i not in st_words])
    if len(text)==0:
        text = "missing"
    return text

# Tokenizing and Padding

def text_vectorizer(feature):

    tk = Tokenizer()
    tk.fit_on_texts(str(feature))
    
    tk_new = tk.texts_to_sequences(str(feature))

    max_length = len(str(feature).split())
    
    pad= pad_sequences(tk_new,padding="post",maxlen = max_length)
    
    return pad 

def func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  arg = np.expand_dims(arg,axis=(0,1))
  return arg

@app.route('/')
def index():
    # Main page
    return "Hello"


@app.route('/predict')
def predicted_price():
    name=request.args.get("name")
    item_condition_id=request.args.get("item_condition_id")
    category_name=request.args.get("category_name")
    brand_name=request.args.get("brand_name")
    shipping=request.args.get("shipping")
    seller_id=request.args.get("seller_id")
    item_description=request.args.get("item_description")
    
    name_processed=name_process(name)
    category_name_processed=category_name_preprocessing(category_name)
    brand_name_processed=brand_name_processed_func(brand_name, name_processed)
    item_description_processed=processing_item_description(item_description)
    tier_2=tier_2_func(category_name_processed)
    tier_3=tier_3_func(category_name_processed)
    
    # Tokenizing Brand_name-processed and padding

    brand_name_pad = text_vectorizer(brand_name_processed)
    tier2_pad = text_vectorizer(tier_2)
    tier3_pad = text_vectorizer(tier_3)
    name_processed_pad = text_vectorizer(name_processed)
    desc_pad = text_vectorizer(item_description_processed)
    
    
    input_processed = [func(float(item_condition_id)),func(float(shipping)),func(brand_name_pad),func(tier2_pad),func(tier3_pad),func(name_processed_pad),func(desc_pad)]

    prediction = model.predict(input_processed,verbose=1,steps=1)
    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    




