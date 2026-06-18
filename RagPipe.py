from vector_store import searchFaiss,indexFaiss
from embedder import embeddings_user
import numpy as np
from chunker import slidingWindow
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


def Groq_client(s,text):
    
    client = Groq(
        api_key = os.getenv("GROQ_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":f"Answer this question: {text} using this context: {s}",
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    print(chat_completion.choices[0].message.content)

def result(text):
    embeddings=embeddings_user(text)
    embeddings=embeddings.reshape(1,-1)
    index,key_x,_=indexFaiss()
    k=4
    key_lst,D=searchFaiss(index,key_x,embeddings,k)

    return key_lst,D

if __name__ == "__main__":
    text=input("Give your inital sentence ")
    key_lst, D = result(text)
    print(key_lst)
    print(D)
    s=""
    emb_dict = slidingWindow()
    for key in key_lst:
        chunk = emb_dict.get(key)
        s += f"[Source: {key[0]}, Page {key[1]}]: {chunk}\n"
    Groq_client(s,text)
