from embedder import getEmbeddings
import faiss
import numpy as np
import os
import pickle
def folder_state():
    return sorted((f, os.path.getmtime(os.path.join("source", f)))
                  for f in os.listdir("source"))
def needsRebuild():
     if not os.path.exists("faiss.index") or not os.path.exists("state.pkl"):
          return True
     with open("state.pkl","rb")as f:
          saved_state=pickle.load(f)
     return saved_state!=folder_state() 
def indexFaiss():
    
    if(not needsRebuild()):
            print("Loading cached index")
            index=faiss.read_index("faiss.index")
            key_x=[]
            with open('file.pkl','rb') as key_file:
                key_x=pickle.load(key_file)
            return index,key_x,None
    else:
        print("Rebuilding index")
        embeddings,key_x=getEmbeddings()
        key_x=list(key_x)

        x,y=embeddings.shape
        t=os.path.getmtime("source")
        index=faiss.IndexFlatIP(y)
        index.add(embeddings)
        faiss.write_index(index,"faiss.index")
        with open ('file.pkl','wb') as file:
                pickle.dump(key_x,file)
        with open("state.pkl", "wb") as f:
            pickle.dump(folder_state(), f)
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
    if(embeddings is not None):
         searchFaiss(Index,key_x,embeddings[:5,:],k)
   
if __name__=="__main__":
    main()