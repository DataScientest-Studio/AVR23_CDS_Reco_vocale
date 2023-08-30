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


def calcul_dic(Lang,Algo,Metrique):

    if Algo=='Manuel':
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index()
    else:
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index()
    return df_dic

def display_dic(df_dic, type, target_lang):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.dataframe(df_dic.T.iloc[:84], height=1500)
    with col2:
        st.dataframe(df_dic.T.iloc[85:169], height=1500)
    with col3:
        st.dataframe(df_dic.T.iloc[170:254], height=1500)
    with col4:
        if len(df_dic.T)>254:
             st.dataframe(df_dic.T.iloc[255:], height=1500)
        else: st.write('')

def run():

    st.title(title)
    st.write("## **Données d'entrée :**\n")
    Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
    Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')
    Algo = st.radio('Algorithme :',('Manuel', 'KMeans','KNN','Random Forest',' Word Embedding'), horizontal=True)
    Metrique = ''
    if (Algo == 'KNN'):
        Metrique = st.radio('Metrique:',('minkowski', 'cosine', 'chebyshev', 'manhattan', 'euclidean'), horizontal=True)

    if (Lang=='en_fr'):
        df_data = load_corpus('../data/preprocess_txt_en').iloc[:-4]
    else:
        df_data = load_corpus('../data/preprocess_txt_fr').iloc[:-4]
    df_data.columns = ['Phrase']

    sentence1 = st.selectbox("1ere phrase à traduire avec le dictionnaire sélectionné", df_data)
    st.write(sentence1)
    n1 = df_data[df_data['Phrase']==sentence1].index.values
    df_dic = calcul_dic(Lang,Algo,Metrique)
    st.write("No phrase:"+str(n1[0]))
    display_dic(df_dic,'ref',Lang)
