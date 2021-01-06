import pickle
import re
import sklearn
import gzip


with gzip.open('Models/transform_f.pkl', 'rb') as ifp:
    tfidf = pickle.load(ifp)

#tfidf = pickle.load(open('Models/transform_f.pkl', "rb"))

# print(tfidf_text[0].todense().shape)
svc = pickle.load(open("Models/svc.pkl", "rb"))
lr = pickle.load(open("Models/LR.pkl", "rb"))
rf = pickle.load(open("Models/rf.pkl", "rb"))

def predictt(text):
    text = str(text)

    text = [''.join(re.sub('[^A-Za-z]',' ',t) for t in text)]

    tfidf_text = tfidf.transform(text)

    svc_result = svc.predict(tfidf_text)
    lr_result = lr.predict(tfidf_text)
    rf_result = rf.predict(tfidf_text)
    r = {
        "svc" : svc_result[0],
        "lr"  : lr_result[0],
        "rf"  : rf_result[0]
    }
    return r
