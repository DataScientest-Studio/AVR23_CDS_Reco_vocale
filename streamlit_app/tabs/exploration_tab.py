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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import subprocess
import sys
import spacy_streamlit



import warnings
warnings.filterwarnings('ignore')


title = "Exploration et Preprocessing"
sidebar_name = "Exploration et Preprocessing"

# Indiquer si l'on veut enlever les stop words. C'est un processus long
stopwords_to_do = True
# Indiquer si l'on veut lemmatiser les phrases, un fois les stop words enlevés. C'est un processus long (approximativement 8 minutes)
lemmatize_to_do = True
# Indiquer si l'on veut calculer le score Bleu pour tout le corpus. C'est un processus très long long (approximativement 10 minutes pour les 10 dictionnaires)
bleu_score_to_do = True
# Première ligne à charger
first_line = 0
# Nombre maximum de lignes à charger
max_lines = 140000
if ((first_line+max_lines)>137860):
    first_line = max(137860-max_lines,0)
    
with contextlib.redirect_stdout(open(os.devnull, "w")):
    nltk.download('stopwords')

@st.cache_data(ttl='1h')  
def load_data(path):
    
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        
    # On convertit les majuscules en minulcule
    data = data.lower()
    
    data = data.split('\n')
    return data[first_line:min(len(data),first_line+max_lines)]

    
def remove_stopwords(text, lang): 
    stop_words = set(stopwords.words(lang))
    # stop_words will contain  set all english stopwords
    filtered_sentence = []   
    for word in text.split(): 
        if word not in stop_words: 
            filtered_sentence.append(word) 
    return " ".join(filtered_sentence)

def clean_undesirable_from_text(sentence, lang):

    # Removing URLs 
    sentence  = re.sub(r"https?://\S+|www\.\S+", "", sentence )
    
    # Removing Punctuations (we keep the . character)
    REPLACEMENTS = [("..", "."),
                    (",", ""),
                    (";", ""),
                    (":", ""),
                    ("?", ""),
                    ('"', ""),
                    ("-", " "),
                    ("it's", "it is"),
                    ("isn't","is not"),
                    ("'", " ")
                   ]
    for old, new in REPLACEMENTS:
        sentence = sentence.replace(old, new)
    
    # Removing Digits
    sentence= re.sub(r'[0-9]','',sentence)
    
    # Removing Additional Spaces
    sentence = re.sub(' +', ' ', sentence)

    return sentence

def clean_untranslated_sentence(data1, data2):
    i=0
    while i<len(data1):
        if data1[i]==data2[i]:
            data1.pop(i)
            data2.pop(i)
        else: i+=1
    return data1,data2

import spacy

nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')


def lemmatize(sentence,lang):
    # Create a Doc object
    if lang=='en':
        nlp=nlp_en
    elif lang=='fr':
        nlp=nlp_fr
    else: return
    doc = nlp(sentence)

    # Create list of tokens from given string
    tokens = [] 
    for token in doc:
        tokens.append(token)

    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
 
    return lemmatized_sentence


def preprocess_txt (data, lang):
    
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    
    tab1, tab2, tab3, tab4 = st.tabs(["Résumé", "Tokenisation","Lemmatisation", "Sans Stopword"])
    with tab1:
        st.subheader("Résumé du pré-processing")
    with tab2:
        st.subheader("Tokenisation")
    with tab3:
        st.subheader("Lemmatisation")
    with tab4:
        st.subheader("Sans Stopword")
    
    word_count = collections.Counter()
    word_lem_count = collections.Counter()
    word_wosw_count = collections.Counter()
    corpus = []
    data_split = []
    sentence_length = []
    data_split_wo_stopwords = []
    data_length_wo_stopwords = []
    data_lem = []
    data_lem_length = []
    
    txt_en_one_string= ". ".join([s for s in data])
    txt_en_one_string = txt_en_one_string.replace('..', '.')
    txt_en_one_string = " "+clean_undesirable_from_text(txt_en_one_string, 'lang')
    data = txt_en_one_string.split('.')
    if data[-1]=="":
        data.pop(-1)
    for i in range(len(data)): # On enleve les ' ' qui commencent et finissent les phrases 
        if data[i][0] == ' ':
            data[i]=data[i][1:]
        if data[i][-1] == ' ':
            data[i]=data[i][:-1]
    nb_phrases = len(data)
    
    # Création d'un tableau de mots (sentence_split)
    for i,sentence in enumerate(data):
        sentence_split = word_tokenize(sentence)
        word_count.update(sentence_split)
        data_split.append(sentence_split)
        sentence_length.append(len(sentence_split))

    # La lemmatisation et le nettoyage des stopword va se faire en batch pour des raisons de vitesse
    # (au lieu de le faire phrase par phrase)
    # Ces 2 processus nécéssitent de connaitre la langue du corpus
    if lang == 'en': l='english'
    elif lang=='fr': l='french'
    else: l="unknown"

    if l!="unknown":
        # Lemmatisation en 12 lots (On ne peut lemmatiser + de 1 M de caractères à la fois)
        if lemmatize_to_do:
            n_batch = 12
            batch_size = round((nb_phrases/ n_batch)+0.5)
            data_lemmatized=""
            for i in range(n_batch):
                to_lem = ".".join([s for s in data[i*batch_size:(i+1)*batch_size]])
                data_lemmatized = data_lemmatized+"."+lemmatize(to_lem,lang).lower()

            data_lem_for_sw = data_lemmatized[1:]  
            data_lemmatized = data_lem_for_sw.split('.')
            for i in range(nb_phrases):
                data_lem.append(data_lemmatized[i].split())
                data_lem_length.append(len(data_lemmatized[i].split()))
                word_lem_count.update(data_lem[-1])
                               
        # Elimination des StopWords en un lot
        # On élimine les Stopwords des phrases lémmatisés, si cette phase a eu lieu
        # (wosw signifie "WithOut Stop Words")
        if stopwords_to_do:
            if lemmatize_to_do:
                data_wosw = remove_stopwords(data_lem_for_sw,l)
            else:
                data_wosw = remove_stopwords(txt_en_one_string,l)
                               
            data_wosw = data_wosw.split('.')
            for i in range(nb_phrases):
                data_split_wo_stopwords.append(data_wosw[i].split())
                data_length_wo_stopwords.append(len(data_wosw[i].split()))
                word_wosw_count.update(data_split_wo_stopwords[-1])

    corpus = list(word_count.keys())
    nb_mots = sum(word_count.values())
    nb_mots_uniques = len(corpus)


    # Affichage du nombre de mot en fonction du pré-processing réalisé   
    with tab1:
        st.write("**Nombre de phrases                     : "+str(nb_phrases)+"**")
        st.write("**Nombre de mots                        : "+str(nb_mots)+"**")
        st.write("**Nombre de mots uniques                : "+str(nb_mots_uniques)+"**")
    with tab3:
        if lemmatize_to_do:
            mots_lem = list(word_lem_count.keys())
            nb_mots_lem = len(mots_lem)
            st.write("**Nombre de mots uniques lemmatisés     : "+str(nb_mots_lem)+"**")
    with tab4:
        if stopwords_to_do:
            mots_wo_sw = list(word_wosw_count)
            nb_mots_wo_stopword = len(mots_wo_sw)
            st.write("**Nombre de mots uniques sans stop words: "+str(nb_mots_wo_stopword)+"**")
    st.write("")

    # Affichage des 5 premiers txt_split
    for i in range(min(5,len(data_split))):
        with tab2:
            st.markdown('**Texte "splited"     '+str(i)+'** : '+str(data_split[i]) ) 
        with tab3:
            if lemmatize_to_do:
                st.markdown('**Texte lemmatisé     '+str(i)+'** : '+str(data_lem[i]))
                if lang == 'en':
                    st.markdown("**Texte avec Tags     "+str(i)+"** : "+str(nltk.pos_tag(data_split[i])))
        with tab4:
            if stopwords_to_do:
                st.markdown('**Texte sans stopwords '+str(i)+'** : '+str(data_split_wo_stopwords[i]))
            # Si langue anglaise, affichage du taggage des mots

   
    
    with tab2:
        # Affichage du corpus de mots uniques    
        st.write("\n**Mots uniques:**")
        st.markdown(corpus[:500])
    with tab3:
        if lemmatize_to_do:
            st.write("\n**Mots uniques lemmatisés:**")
            st.markdown(mots_lem[:500])
    with tab4:
        if stopwords_to_do:
            st.write("\n**Mots uniques sans stop words:**")
            st.markdown(mots_wo_sw[:500])

    
    # Création d'un DataFrame txt_n_unique_val :
    #      colonnes = mots
    #      lignes = phases
    #      valeur de la cellule = nombre d'occurence du mot dans la phrase

    
    ## BOW
    from sklearn.feature_extraction.text import CountVectorizer
    count_vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"[^' ']+" )
    
    # Calcul du nombre d'apparition de chaque mot dans la phrases
    countvectors = count_vectorizer.fit_transform(data)
    corpus = count_vectorizer.get_feature_names_out()

    txt_n_unique_val=  pd.DataFrame(columns=corpus,index=range(nb_phrases), data=countvectors.todense()).astype(float)
    with tab1:
        st.write("\n**Nombre d'apparitions de chaque mot dans chaque phrase (Bag Of Words):**")
        st.dataframe(txt_n_unique_val.head(50)) # .iloc[:,:40].head(10)
    with tab2:
        st.write("\n**Nombre d'apparitions de chaque mot dans chaque phrase (Bag Of Words):**")
        st.dataframe(txt_n_unique_val.head(50))
    
  
    
    
    return data, corpus, data_split, txt_n_unique_val, sentence_length, data_length_wo_stopwords, data_lem_length  
 

def run():
    global max_lines, first_line, lemmatize_to_do, stopwords_to_do
    
    st.title(title)
    
    #Chargement des textes complet dans les 2 langues
    first_line=0
    max_lines = 140000
    full_txt_en = load_data('../data/small_vocab_en')
    full_txt_fr = load_data('../data/small_vocab_fr')
    # 
    st.write("## **Données d'entrée :**\n")
    Langue = st.radio('Langue:',('Anglais','Français'), horizontal=True)
    first_line = st.slider('No de la premiere ligne à analyser'':',0,137859)
    max_lines = st.select_slider('Nombre de lignes à analyser (Attention, si Max pas de lemmatisation)'':',
                              options=[1,5,10,15,100, 1000,'Max'])
    if max_lines=='Max':
        max_lines=137860
    if ((first_line+max_lines)>137860):
        first_line = max(137860-max_lines,0)
    if ((max_lines-first_line)>1000): 
        lemmatize_to_do = False
    else:
        lemmatize_to_do = True
        
    last_line = first_line+max_lines
    if (Langue=='Anglais'):
        st.write(pd.DataFrame(data=full_txt_en,columns=['Texte']).loc[first_line:last_line-1])
    else:
        st.dataframe(pd.DataFrame(data=full_txt_fr,columns=['Texte']).loc[first_line:last_line-1])
    st.write("")

    #Chargement des textes sélectionnés dans les 2 langues (max lignes = max_lines)
    txt_en = full_txt_en[first_line:first_line+max_lines]
    txt_fr = full_txt_fr[first_line:first_line+max_lines]   
    # Elimination des phrases non traduites
    txt_en, txt_fr = clean_untranslated_sentence(txt_en, txt_fr)
    
    # Lancement du préprocessing du texte qui va spliter nettoyer les phrases et les spliter en mots 
    # et calculer nombre d'occurences des mots dans chaque phrase
    if (Langue == 'Anglais'):
        st.write("## **Préprocessing de small_vocab_en :**\n")
        txt_en, corpus_en, txt_split_en, df_count_word_en,sent_len_en, sent_wo_sw_len_en, sent_lem_len_en  = preprocess_txt (txt_en,'en')
    else:
        st.write("## **Préprocessing de small_vocab_fr :**\n")
        txt_fr, corpus_fr, txt_split_fr, df_count_word_fr,sent_len_fr, sent_wo_sw_len_fr, sent_lem_len_fr  = preprocess_txt (txt_fr,'fr')








    # DEFAULT_TEXT = """Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."""
    """
    spacy_model = "en_core_web_sm"

    text = st.text_area("Text to analyze", DEFAULT_TEXT, height=200)
    doc = spacy_streamlit.process_text(spacy_model, text)

    spacy_streamlit.visualize_ner(
        doc,
        labels=["PERSON", "DATE", "GPE"],
        show_table=False,
        title="Persons, dates and locations",
        )
    st.text(f"Analyzed using spaCy model {spacy_model}")
    """

    # models = ["en_core_web_sm"]
    # default_text = "Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002."
    # spacy_streamlit.visualize(models, default_text)









