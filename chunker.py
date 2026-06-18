from loader import loader
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt_tab')
def slidingWindow():
    txt_dict=loader()
    windowSize=650
    overlap=180
    key_x=(txt_dict.keys())
    emb_dict={}
    for i in key_x:
        txt_dict[i]=sent_tokenize(txt_dict[i])
 
    for i in key_x:
        senteces=txt_dict.get(i)
        buffer=""
        chunk_size=0
        curr=[]
        for sent in senteces:
            size=len(buffer)+len(sent)
            if(size>windowSize):
                emb_dict[(i, chunk_size)] = buffer
                chunk_size+=1
                curr=curr[-2:]
                buffer=" ".join(curr)
            curr.append(sent)
            buffer+=sent+" "
        if buffer:
            emb_dict[(i,chunk_size)]=buffer
    return emb_dict

def main():
    emb_dict=slidingWindow()
    print(emb_dict)
if __name__=="__main__":
    main()