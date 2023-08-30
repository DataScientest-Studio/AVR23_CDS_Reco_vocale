import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


title = "Traduction mot à mot"
sidebar_name = "Traduction mot à mot"

def display_dic(type, target_lang):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header(type)
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header(target_lang)
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")

def run():

    st.title(title)
    st.write("## **Données d'entrée :**\n")
    Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
    Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')
    Algo = st.radio('Algorithme :',('KMeans','KNN','Random Forest',' Word Embedding'), horizontal=True)
    if (Algo == 'KNN'):
        Metrique = st.radio('Metrique:',('minkowski', 'cosine', 'chebyshev', 'manhattan', 'euclidean'), horizontal=True)
    else:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
    display_dic('ref',Lang)
    