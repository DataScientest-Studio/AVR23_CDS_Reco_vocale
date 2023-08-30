import streamlit as st
import pandas as pd
import numpy as np
import os


title = "Traduction mot à mot"
sidebar_name = "Traduction mot à mot"

def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

df_data_en = load_corpus('../data/preprocess_txt_en')
df_data_fr = load_corpus('../data/preprocess_txt_fr')

def calcul_dic(Lang,Algo,Metrique):

    if Algo=='Manuel':
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index()
    else:
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index()
    return df_dic

def display_translation(n1,dict, Lang):
    global df_data_src, df_data_tgt

    for i in range(n1,n1+5):
        s = df_data_src.iloc[i]['Phrase']
        source = Lang[:2]
        target = Lang[-2:]
        st.write("**"+source+"   :**  "+ s)
        st.write("**"+target+"   :**  "+(' '.join(dict[col].iloc[0] for col in s.split())))
        st.write("**ref. :** "+df_data_tgt.iloc[i][0])
        st.write("")

def display_dic(df_dic):
    st.dataframe(df_dic.T, height=800)



def run():
    global df_data_src, df_data_tgt

    st.title(title)
    st.write("## **Données d'entrée :**\n")
    Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
    Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')
    Algo = st.radio('Algorithme :',('Manuel', 'KMeans','KNN','Random Forest',' Word Embedding'), horizontal=True)
    Metrique = ''
    if (Algo == 'KNN'):
        Metrique = st.radio('Metrique:',('minkowski', 'cosine', 'chebyshev', 'manhattan', 'euclidean'), horizontal=True)

    if (Lang=='en_fr'):
        df_data_src = df_data_en
        df_data_tgt = df_data_fr
    else:
        df_data_src = df_data_fr
        df_data_tgt = df_data_en
    df_data_src.columns = ['Phrase']

    sentence1 = st.selectbox("Première des 5 phrase à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4])
    n1 = df_data_src[df_data_src['Phrase']==sentence1].index.values[0]
    df_dic = calcul_dic(Lang,Algo,Metrique)
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        display_dic(df_dic)
    with col2:
        display_translation(n1, df_dic, Lang)   

