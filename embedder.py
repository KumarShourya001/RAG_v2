from sentence_transformers import SentenceTransformer
from chunker import slidingWindow
import numpy as np
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embeddings_user(text):
    embeddings=model.encode(text).astype(np.float32)
    return embeddings
def getEmbeddings():
    emb_dict=slidingWindow()
    key_x=list(emb_dict.keys())
    sentences=[]
    for i in (key_x):
        sentences.append(emb_dict.get(i))
    embeddings = model.encode(sentences)
    embeddings=embeddings.astype(np.float32)
    return embeddings,key_x

def main():
    embeddings,key_x=getEmbeddings()
    print(embeddings.shape)
    print(key_x)

if __name__=="__main__":
    main()