import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Traduction mot à mot"
sidebar_name = "Traduction mot à mot"

def calcul_dic(Lang,Algo,Metrique):


    if Algo=='Manuel':
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T
    else:
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T
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
    else:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
    df_dic = calcul_dic(Lang,Algo,Metrique)
    display_dic(df_dic,'ref',Lang)
    