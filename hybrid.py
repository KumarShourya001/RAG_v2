from vector_store import indexFaiss,searchFaiss
from BM25 import BM_idx,search_BM
from embedder import embeddings_user
from itertools import chain
from chunker import slidingWindow
def Rank(text):
    query=text.split(" ")
    idx_fias,key_x,embeddings=indexFaiss()
    emb = embeddings_user(text).reshape(1, -1)
    idx_bm25,_=BM_idx()
    _,rnk_BM=search_BM(query,idx_bm25)
    key_fiass,_,I=searchFaiss(idx_fias,key_x,emb,k=(len(key_x)))
    I= list(chain.from_iterable(I))
    faiss_rank={}
    for rank,pos in enumerate(I):
        faiss_rank[pos]=rank
    BM_rank={}
    for rank,pos in enumerate(rnk_BM):
        BM_rank[pos]=rank
    key_BMrnk=list(BM_rank.keys())
    key_fiassrnk=list(faiss_rank.keys())
    score={}
    all_positions = set(faiss_rank.keys()) | set(BM_rank.keys())
    k=60
    default=len(key_x)
    chk=10
    for pos in all_positions:
        score[pos]=1/(k+faiss_rank.get(pos,default))+1/(k+BM_rank.get(pos,default))
    top=sorted(score,key=score.get,reverse=True)[:chk]
    top_keys=[key_x[pos] for pos in top]
    
    top_n = 10
    faiss_keys = set(key_x[p] for p in I[:10])
    bm25_keys = set(key_x[p] for p in rnk_BM[:10])
    agreement = len(faiss_keys & bm25_keys) / 10
   
    return top_keys,agreement

def main():
    text=input("Enter Your text here ")
    top_keys,agreement=Rank(text)
    # print(top_keys)
    print(agreement)
    

if __name__=="__main__":
    main()