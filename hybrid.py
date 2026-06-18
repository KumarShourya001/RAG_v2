from vector_store import indexFaiss,searchFaiss
from BM25 import BM_idx,search_BM
from embedder import embeddings_user
from itertools import chain
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
    for pos in all_positions:
        score[pos]=1/(k+faiss_rank.get(pos,default))+1/(k+BM_idx.get(pos,default))
    
def main():
    text=input("Enter Your text here ")
    Rank(text)

if __name__=="__main__":
    main()