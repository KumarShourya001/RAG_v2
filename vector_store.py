from embedder import getEmbeddings
import faiss
import numpy as np
def indexFaiss():
    embeddings,key_x=getEmbeddings()
    key_x=list(key_x)

    x,y=embeddings.shape
    
    index=faiss.IndexFlatIP(y)
    index.add(embeddings)
    return index,key_x,embeddings


def searchFaiss(index,key_x,embeddings,k):
    D,I=index.search(embeddings,k)
    key_lst=[]
    for row in I:
        for pos in row:
            key_lst.append(key_x[pos])
    
    print(I)
    print(D)
    print(key_lst)
    return key_lst,D,I
def main():
    Index,key_x,embeddings=indexFaiss()
    k=len(key_x)
    searchFaiss(Index,key_x,embeddings[:5,:],k)
if __name__=="__main__":
    main()