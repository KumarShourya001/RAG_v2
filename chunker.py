from loader import loader

def slidingWindow():
    txt_dict=loader()
    windowSize=650
    overlap=180
    key_x=(txt_dict.keys())
    emb_dict={}
    for i in key_x:
        for j in range(0, len(txt_dict[i]), windowSize-overlap):
            emb_dict[i,j]=txt_dict.get(i)[j:j+windowSize]
    return emb_dict

def main():
    emb_dict=slidingWindow()
    print(emb_dict)
if __name__=="__main__":
    main()