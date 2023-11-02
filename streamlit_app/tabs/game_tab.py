import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import tiktoken
import random
import joblib
import json
import csv
from filesplit.merge import Merge
from extra_streamlit_components import tab_bar, TabBarItemData
import matplotlib.pyplot as plt
import seaborn as sns

title = "Jouez avec nous !"
sidebar_name = "Jeu"


def read_list_lan():

    with open('../data/multilingue/lan_code.csv', 'r') as fichier_csv:
        reader = csv.reader(fichier_csv)
        lan_code = next(reader)
        return lan_code

@st.cache_data
def init_game():

    new = int(time.time())
    sentence_test = pd.read_csv('../data//multilingue/sentence_test_extract.csv')
    list_lan = read_list_lan()
        # Lisez le contenu du fichier JSON
    with open('../data/multilingue/lan_to_language.json', 'r') as fichier:
        lan_to_language = json.load(fichier)
    lan_identified = [lan_to_language[l] for l in list_lan]
    return sentence_test, list_lan, lan_identified, lan_to_language, new

def find_indice(sent_selected):
    l = list(lan_to_language.keys())
    for i in range(len(l)):
        if l[i] == sentence_test['lan_code'].iloc[sent_selected]:
            return i

@st.cache_data
def set_game(new):
    nb_st = len(sentence_test)
    sent_sel = []
    # Utilisez une boucle pour générer 5 nombres aléatoires différents
    while len(sent_sel) < 5:
        nombre = random.randint(0, nb_st)
        if nombre not in sent_sel:
            sent_sel.append(nombre)

    rep_possibles=[]
    for i in range(5):
        rep_possibles.append([find_indice(sent_sel[i])])
        while len(rep_possibles[i]) < 5:
            rep_possible = random.randint(0, 95)
            if rep_possible not in rep_possibles[i]:
                rep_possibles[i].append(rep_possible)
        random.shuffle(rep_possibles[i])
    return sent_sel, rep_possibles,new

def run():
    global sentence_test, list_lan, lan_identified, lan_to_language

    sentence_test, list_lan, lan_identified, lan_to_language, new = init_game()

    st.write("")
    st.title(title)
    st.write("#### **Etes vous un expert es Langues ?**\n")
    st.markdown(
        """
        Essayer de trouvez, sans aide, la langue des 5 phrases suivantes.  
        Attention : Vous devez être le plus rapide possible !  
        """, unsafe_allow_html=True
        )
    st.write("")

    sent_sel, rep_possibles, new = set_game(new)
    answer = [""] * 5
    l = list(lan_to_language.values())
    for i in range(5):
        answer[i] = st.radio("**:blue["+sentence_test['sentence'].iloc[sent_sel[i]]+"]**\n",[l[rep_possibles[i][0]],l[rep_possibles[i][1]],l[rep_possibles[i][2]], \
                                                                                      l[rep_possibles[i][3]],l[rep_possibles[i][4]]], horizontal=True, key=i)
    if st.button(label="Valider", type="primary"):
        st.cache_data.clear()

        score = 0
        nb_bonnes_reponses = 0
        for i in range(5):
            if lan_to_language[sentence_test['lan_code'].iloc[sent_sel[i]]]==answer[i]:
                score +=200
                nb_bonnes_reponses +=1
        if nb_bonnes_reponses >=4:
            st.write(":red[**Félicitations, vous avez "+str(nb_bonnes_reponses)+" bonnes réponses !**]")
            st.write(":red[Votre score est de "+str(score)+" points]")
        else:
            if nb_bonnes_reponses >1 : s="s" 
            else: s=""
            st.write("**:red[Vous avez "+str(nb_bonnes_reponses)+" bonnes réponse"+s+".]**")
            if nb_bonnes_reponses >0 : s="s"
            else: s=""
            st.write(":red[Votre score est de "+str(score)+" point"+s+"]")

        st.write("Bonne réponses:")
        for i in range(5):
            st.write("- "+sentence_test['sentence'].iloc[sent_sel[i]]+" -> :blue[**"+lan_to_language[sentence_test['lan_code'].iloc[sent_sel[i]]]+"**]")
        new = int(time.time())
        st.button(label="Play again ?", type="primary")
    return


        





    
