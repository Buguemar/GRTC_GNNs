from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pickle
import random,re, os, nltk
from nltk.corpus import stopwords

def clean_rep(text):
    temp=text.split()
    new=[]
    for i in range(len(temp)):
        try:
            if temp[i]!=temp[i+1]:
                new.append(temp[i])
        except:
            new.append(temp[i])

    return ' '.join(new)


def clean_str(string, preprocess): 
    """
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if preprocess=="soft": 
        
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) 
        string = re.sub(r",", " ", string)
        string = re.sub(r"\. ", " ", string) 
        string = re.sub(r"\_", " ", string) 
        string = re.sub(r"!", " ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " ", string)
        string = re.sub(r"\'", " ", string)
        string = string.strip().lower()
        #stopwords 
        stop_words = stopwords.words('english')
        text = word_tokenize(string)
        text = [word for word in text if word not in stop_words]
        return ' '.join(text)
    
    
    if preprocess=="tlgcn":
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        # string = re.sub(r"[0-9]", " ", string)
        string = re.sub(r"\'s", " is", string)
        string = re.sub(r"\'ve", " have", string)
        string = re.sub(r"n\'t", " not", string)
        string = re.sub(r"\'re", " are", string)
        string = re.sub(r"\'d", " would", string)
        string = re.sub(r"\'ll", " will", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"!", " ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " ", string)
        string = re.sub(r"\'", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


def remove_short(string, limit):
    results = []
    for word in string.split(' '):
        if len(word) < limit:  
            continue
        else:
            results.append(word)

    return ' '.join(results)


def clean_text(text, as_TLGCN=True):
    
    stemmer = WordNetLemmatizer()
    text = clean_str(text, preprocess="tlgcn")
    
    if as_TLGCN:
        text= remove_short(text, 3)
        text = word_tokenize(text)
        text = [stemmer.lemmatize(word) for word in text]
    
    else: 
        text = word_tokenize(text)
    
    text=' '.join(text)
    
    return clean_rep(text)

