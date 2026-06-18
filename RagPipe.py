from vector_store import searchFaiss,indexFaiss
from embedder import embeddings_user
import numpy as np
from chunker import slidingWindow
from groq import Groq
import os
from dotenv import load_dotenv
from hybrid import Rank
from groq import AuthenticationError, RateLimitError
load_dotenv()


def Groq_client(s,text):
    
    client = Groq(
        api_key = os.getenv("GROQ_API_KEY")
    )
    try: 
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content":f"Answer this question: {text} using this context: {s}",
                }
            ],
            model="llama-3.3-70b-versatile",
        )

        
        return chat_completion.choices[0].message.content
    except AuthenticationError:
        print("Api key invalid or expired")
        return None
    except RateLimitError:
        print("Rate Limit hit")
        return None
    except Exception as e:
        print(f"unexpected error: {e}")
        return None

def result(text):
    rnk_keys=Rank(text)

    return rnk_keys

def main(): 
    text=input("Give your inital sentence ")
    rnk_keys=result(text)
    s=""
    emb_dict = slidingWindow()
    for key in rnk_keys:
        chunk = emb_dict.get(key)
        s += f"[Source: {key[0]}, Page {key[1]}]: {chunk}\n"
    answer=Groq_client(s,text)
    if answer:
        print (answer)

if __name__ == "__main__":
    main()