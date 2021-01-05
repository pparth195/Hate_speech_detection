# https://stackoverflow.com/a/47091490/4084039
import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def process(rawtext):
    preprocessed_essays = []
    # tqdm is for printing the status bar
    for sentance in tqdm(rawtext):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\"', ' ')
        sent = sent.replace('\\n', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split() if e not in set(stopwords.words('english')))
        preprocessed_essays.append(sent.lower().strip())
    return preprocessed_essays

def stemming(preprocessed_essays):
    processsedsentence=[]   
    ps = PorterStemmer() 
   
    for sentence in preprocessed_essays:
        words = word_tokenize(sentence) 
        processsedtext=""
        for w in words: 
            processsedtext=processsedtext + ps.stem(w)+" " 
        #print(processsedtext)    
        processsedsentence.append(processsedtext)    
    return processsedsentence



def gridsearch(X_train,y_train,clf,grid,scorer):
 
    rf_cv=GridSearchCV(clf,grid,cv=2,scoring=scorer,verbose=10,return_train_score=True)
    rf_cv.fit(X_train, y_train)

    print("best parameters:",rf_cv.best_params_)
    return rf_cv

#https://www.geeksforgeeks.org/python-get-first-element-of-each-sublist/
def Extract(lst): 
    return [item[1] for item in lst] 

def plot_roc_train_for_best_param(X_tr,y_train,X_te,clf,y_test):
    
    clf.fit(X_tr, y_train)

    y_train_pred_prob = clf.predict_proba(X_tr) 
    y_test_pred_prob = clf.predict_proba(X_te)

    y_train_pred=Extract(y_train_pred_prob)
    y_test_pred=Extract(y_test_pred_prob)

    train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("False positive rate")
    plt.ylabel("True Positive rate")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()
    return train_fpr, train_tpr, tr_thresholds,test_fpr,test_tpr,te_thresholds,y_train_pred,y_test_pred_prob

def find_best_threshold(threshould, fpr, tpr):
    t = threshould[np.argmax(tpr*(1-fpr))]
    # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
    print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
    return t

def predict_with_best_t(proba, threshould):
    predictions = []
    for i in proba:
        if i>=threshould:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

#code copied from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
def plt_confusion_matrix(y_train,y_pred):
    df_cm = pd.DataFrame(confusion_matrix(y_train,y_pred), range(2), range(2))
    # plt.figure(figsize=(10,7))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size
    plt.xlabel('Actual classes')
    plt.ylabel('Predicted classes')
    plt.show()
