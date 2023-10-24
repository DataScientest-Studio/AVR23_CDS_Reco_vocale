import streamlit as st
import  streamlit_toggle as tog
import pandas as pd
import numpy as np
import os

title = "Identification de langue"
sidebar_name = "Identification de langue"


def run():
    st.title(title)
    st.write("Comme préambule à la traduction, il apparaît nécessaire de pouvoir détecter la langue d'origine, ne serait-ce que pour ne traduire que des langues concernées par nos modèles. De plus, les principes de la détection de langue offrent une bonne introduction aux outils qui seront employés pour la traduction.\n ")
    st.write("")
    
    st.write("### Tokenisation\n")
    st.write("""Réaliser un bag of words - c'est à dire la présence ou non du mot dans la phrase - directement sur le dataset atteint rapidement des limites de performances. En effet, ce qui revient à réaliser un one hot encoding de mots, ne l'est que sur les mots présent dans le dataset.
             Ainsi, toutes les déclinaisons d'un mot ou d'un verbe non incluses dans le dataset ne seront pas détectées, et le modèle sera au final peu efficace.   
            """)
    st.write("")
    st.write("""La tokenisation permet de pallier à ce problème. En découpant les mots en token, comme le mot manger par exemple avec les tokens man, an, ang, ger, ra, ait... et ainsi de suite, nous sommes capable de détecter toutes les formes possible du verbe grâce à la combinaison de tokens.
             Différents tokenisers existent aujourd'hui, chacun découpant les mots suivant un dictionnaire de tokens qui lui est propre, avec des tokens plus ou moins nombreux.
             Une fois tokenisé, l'opération de bag of words (BOW) peut être appliquée afin de détecter la présence ou non de chaque token dans la phrase.  
             """)

    st.write("### Entraînement de modèles \n")
    st.write(""" une fois le BOW créé, il est possible d'en extraire les différents dataset train et test, sans oublier d'avoir les features (le BOW des tokens) et la variable cible qui est le label de la langue (Français, English...).
             Cela revient donc à un problème de classification où telle ou telle liste de token, correspond à telle ou telle langue.
             Différents classifiers peuvent être employés, comme le GradientBoosting ou le Naivebayes, voir les réseaux de neurones. 
             Comme le dataset comprend de nombreuses phrases dans de nombreuses langues, ces modèles sont assez long en calcul, vous pouvez donc les charger directements.
             """)
    
    
    tokenizer = "Bert"
 
    
    
    selected_option = st.empty()
    tog_token = tog.st_toggle_switch( 
                                    key="OptionSwitch",
                                    default_value=False,
                                    label_after=True,
                                    inactive_color='#FACE06',
                                    active_color="#11567f",
                                    track_color="#29B5E8")

    if tog_token:
        selected_option.write("Option A sélectionnée.")
    else:
        selected_option.write("Option B sélectionnée.")
    
 
 
 
    st.button("Reset", type="primary")
    if st.button('Say hello'):
        st.write('Why hello there')
    else:
        st.write('Goodbye')





    st.markdown(
        """
        <H1> Test </H1>
        """)
    
    
    st.write("")
    st.write("Waiting for Keynes & Tia's code...")
    st.write("")
    st.write("## **Paramètres :**\n")
    st.write("")
    st.write("## **Résultats :**\n")
 