import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


title = "Traduction mot à mot"
sidebar_name = "Traduction mot à mot"

def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

def load_BOW(path, l):
    input_file = os.path.join(path)
    df1 = pd.read_csv(input_file+'1_'+l, encoding="utf-8", index_col=0)
    df2 = pd.read_csv(input_file+'2_'+l, encoding="utf-8", index_col=0)
    df_count_word  = pd.concat([df1, df2]) 
    return df_count_word

df_data_en = load_corpus('../data/preprocess_txt_en')
df_data_fr = load_corpus('../data/preprocess_txt_fr')
df_count_word_en = load_BOW('../data/preprocess_df_count_word', 'en')
df_count_word_fr = load_BOW('../data/preprocess_df_count_word', 'fr')
n1 = 0

nb_mots_en = 199 # len(corpus_en)
nb_mots_fr = 330 # len(corpus_fr)

# On modifie df_count_word en indiquant la présence d'un mot par 1 (au lieu du nombre d'occurences)
df_count_word_en = df_count_word_en[df_count_word_en==0].fillna(1)
df_count_word_fr = df_count_word_fr[df_count_word_fr==0].fillna(1)

# On triche un peu parce que new et jersey sont toujours dans la même phrase et donc dans la même classe
if ('new' in df_count_word_en.columns):
    df_count_word_en['new']=df_count_word_en['new']*2
    df_count_word_fr['new']=df_count_word_fr['new']*2

# ============

def calc_kmeans(l_src,l_tgt):
    global df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt

    # Algorithme de K-means
    init_centroids = df_count_word_tgt.T
    kmeans = KMeans(n_clusters = nb_mots_tgt, n_init=1, max_iter=1, init=init_centroids, verbose=0)

    kmeans.fit(df_count_word_tgt.T)

    # Centroids and labels
    centroids= kmeans.cluster_centers_
    labels = kmeans.labels_

    # Création et affichage du dictionnaire
    df_dic = pd.DataFrame(data=df_count_word_tgt.columns[kmeans.predict(df_count_word_src.T)],index=df_count_word_src.T.index,columns=[l_tgt])
    df_dic.index.name= l_src
    df_dic = df_dic.T
    # print("Dictionnaire Anglais -> Français:")
    # translation_quality['Précision du dictionnaire'].loc['K-Means EN->FR'] =round(accuracy(dict_EN_FR_ref,dict_EN_FR)*100, 2)
    # print(f"Précision du dictionnaire = {translation_quality['Précision du dictionnaire'].loc['K-Means EN->FR']}%")
    # display(dict_EN_FR)
    return df_dic

def calc_knn(l_src,l_tgt, metric):
    global df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt

    #Définition de la metrique (pour les 2 dictionnaires
    knn_metric = metric   # minkowski, cosine, chebyshev, manhattan, euclidean

    # Algorithme de KNN
    X_train = df_count_word_tgt.T
    y_train = range(nb_mots_tgt)

    # Création du classifieur et construction du modèle sur les données d'entraînement
    knn = KNeighborsClassifier(n_neighbors=1, metric=knn_metric)
    knn.fit(X_train, y_train)

    # Création et affichage du dictionnaire
    df_dic = pd.DataFrame(data=df_count_word_tgt.columns[knn.predict(df_count_word_src.T)],index=df_count_word_en.T.index,columns=[l_tgt])
    df_dic.index.name = l_src
    df_dic = df_dic.T

    # print("Dictionnaire Anglais -> Français:")
    # translation_quality['Précision du dictionnaire'].loc['KNN EN->FR'] =round(accuracy(dict_EN_FR_ref,knn_dict_EN_FR)*100, 2)
    # print(f"Précision du dictionnaire = {translation_quality['Précision du dictionnaire'].loc['KNN EN->FR']}%")
    # display(knn_dict_EN_FR)
    return df_dic

def calcul_dic(Lang,Algo,Metrique):

    if Lang[:2]=='en': 
        l_src = 'Anglais'
        l_tgt = 'Francais'
    else:
        l_src = 'Francais'
        l_tgt = 'Anglais'

    if Algo=='Manuel':
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index(axis=1)
    elif Algo=='KMeans':
         df_dic = calc_kmeans(l_src,l_tgt)
    elif Algo=='KNN':
        df_dic = calc_knn(l_src,l_tgt, Metrique)
    else:
        df_dic = pd.read_csv('../data/dict_ref_'+Lang+'.csv',header=0,index_col=0, encoding ="utf-8", sep=';',keep_default_na=False).T.sort_index(axis=1)
    return df_dic
# ============

def display_translation(n1,dict, Lang):
    global df_data_src, df_data_tgt

    for i in range(n1,n1+5):
        s = df_data_src.iloc[i][0]
        source = Lang[:2]
        target = Lang[-2:]
        # for col in s.split():
        #     st.write('col: '+col)
        #     st.write('dict[col]! '+dict[col])
        st.write("**"+source+"   :**  "+ s)
        st.write("**"+target+"   :**  "+(' '.join(dict[col].iloc[0] for col in s.split())))
        st.write("**ref. :** "+df_data_tgt.iloc[i][0])
        st.write("")

def display_dic(df_dic):
    st.dataframe(df_dic.T, height=600)


def run():
    global df_data_src, df_data_tgt, df_count_word_src, df_count_word_tgt, nb_mots_src, nb_mots_tgt, n1
    global df_data_en, df_data_fr, nb_mots_en, df_count_word_en, df_count_word_fr, nb_mots_en, nb_mots_fr

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
        df_count_word_src = df_count_word_en
        df_count_word_tgt = df_count_word_fr
        nb_mots_src = nb_mots_en
        nb_mots_tgt = nb_mots_fr
    else:
        df_data_src = df_data_fr
        df_data_tgt = df_data_en
        df_count_word_src = df_count_word_fr
        df_count_word_tgt = df_count_word_en
        nb_mots_src = nb_mots_fr
        nb_mots_tgt = nb_mots_en

    # df_data_src.columns = ['Phrase']
    sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrase à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
    n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
    st.write("## **Dictionnaire calculé et traduction mot à mot :**\n")
    df_dic = calcul_dic(Lang,Algo,Metrique)
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.write("#### **Dictionnaire**")
        display_dic(df_dic)
    with col2:
        st.write("#### **Traduction**")
        display_translation(n1, df_dic, Lang)   
