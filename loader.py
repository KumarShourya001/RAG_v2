import os 
import pymupdf

def loader():
    pdf_lst=[f for f in os.listdir("source")if f.lower().endswith('.pdf')]
    txt_dict={}
    for i in pdf_lst:
        doc=pymupdf.open(os.path.join("source",i))
        for j,page in enumerate(doc):
            txt_dict[(i,j)]=page.get_text()
           
    return txt_dict

def main():
    txt_dict=loader()
    print(txt_dict)
if __name__=="__main__":
    main()