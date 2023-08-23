import streamlit as st
from PIL import Image
import os
import time
import random
import ast
import contextlib
import numpy as np
import pandas as pd
import collections
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords


title = "Data Vizualization"
sidebar_name = "Data Vizualization"

with contextlib.redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')

# Première ligne à charger
first_line = 0
# Nombre maximum de lignes à charger
max_lines = 140000

if ((first_line+max_lines)>137860):
    first_line = max(137860-max_lines,0)
    
def load_data(path):
    
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        
    # On convertit les majuscules en minulcule
    data = data.lower()
    
    data = data.split('\n')
    return data[first_line:min(len(data),first_line+max_lines)]
    
def plot_word_cloud(text, title, masque, stop_words, background_color = "white"):
    
    mask_coloring = np.array(Image.open(str(masque)))
    # Définir le calque du nuage des mots
    wc = WordCloud(background_color=background_color, max_words=200, 
                   stopwords=stop_words, mask = mask_coloring, 
                   max_font_size=50, random_state=42)
    # Générer et afficher le nuage de mots
    fig=plt.figure(figsize= (20,10))
    plt.title(title, fontsize=25, color="green")
    wc.generate(text)
    
    # getting current axes
    a = plt.gca()
 
    # set visibility of x-axis as False
    xax = a.axes.get_xaxis()
    xax = xax.set_visible(False)
 
    # set visibility of y-axis as False
    yax = a.axes.get_yaxis()
    yax = yax.set_visible(False)
    
    plt.imshow(wc)
    # plt.show()
    st.pyplot(fig)

def run():
    
    global max_lines, first_line
    
    st.title(title)

    #Chargement des textes complet dans les 2 langues
    first_line=0
    max_lines = 140000
    full_txt_en = load_data('../data/small_vocab_en')
    full_txt_fr = load_data('../data/small_vocab_fr')
    # 
    st.write("## **Données d'entrée :**\n")
    Langue = st.radio('Langue:',('Anglais','Français'), horizontal=True)
    first_line = st.slider('No de la premiere ligne à analyser :',1,137860)-1
    max_lines = st.select_slider('Nombre de lignes à analyser :',
                              options=[1,5,10,15,100, 1000,'Max'])
    if max_lines=='Max':
        max_lines=137860
    if ((first_line+max_lines)>137860):
        first_line = max(137860-max_lines,0)
     
    # Chargement des textes sélectionnés (max lignes = max_lines)
    if (Langue == 'Anglais'):
        txt_en = load_data('../data/small_vocab_en')
    else:
        txt_fr = load_data('../data/small_vocab_fr')
    for i in range(min(15,max_lines)):
        if (Langue == 'Anglais'):
            st.write(str(first_line+i),": ", full_txt_en[first_line+i])
        else:
            st.write(str(first_line+i),": ", full_txt_fr[first_line+i])
        
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["World Cloud", "Frequence","Distribution longueur", "Co-occurence", "Proximité"])
    with tab1:
        st.subheader("World Cloud")
    with tab2:
        st.subheader("Frequence d'apparition des mots")
    with tab3:
        st.subheader("Distribution des longueurs de phases")
    with tab4:
        st.subheader("Co-occurence des mots dans une phrase") 
    with tab5:
        st.subheader("Proximité sémantique des mots") 
        
    with tab1:
        if (Langue == 'Anglais'):
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('english'))
            for e in txt_en : text += e
            plot_word_cloud(text, "English words corpus", "../images/coeur.png", stop_words)
        else:
            text = ""
            # Initialiser la variable des mots vides
            stop_words = set(stopwords.words('french'))
            for e in txt_fr : text += e
            plot_word_cloud(text,"Mots français du corpus", "../images/coeur.png", stop_words)