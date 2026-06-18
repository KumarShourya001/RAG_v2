from chunker import slidingWindow 
from rank_bm25 import BM25Okapi 
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def BM_idx():
    emb_dict=slidingWindow()
    key_x=list(emb_dict.keys())
    corpus=[]
    for key in key_x:
        sent=emb_dict.get(key)
        tokenize_corpus=[w for w in sent.lower().split(" ") if w not in stop]
        corpus.append(tokenize_corpus)
    bm25=BM25Okapi(corpus)
    return bm25,key_x

def search_BM(query,bm25):
    query=[w.lower()for w in query if w.lower() not in stop]
    scores=bm25.get_scores(query)
    rank=np.argsort(scores)[::-1]
    return scores,rank

def main():
    bm25,_=BM_idx()
    query=input("Enter Your sentence here ")
    token_query=query.split(" ")
    srch,rnk=search_BM(token_query,bm25)
    print(f"{rnk}->{srch[rnk]}") 
     

if __name__=="__main__":
    main()