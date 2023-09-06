import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sacrebleu import corpus_bleu
from transformers import pipeline
import whisper
import sounddevice as sd

title = "Traduction Sequence à Sequence"
sidebar_name = "Traduction Seq2Seq"

# !pip install transformers
# !pip install sentencepiece

@st.cache_data(ttl='1h00s')
def load_corpus(path):
    input_file = os.path.join(path)
    with open(input_file, "r",  encoding="utf-8") as f:
        data = f.read()
        data = data.split('\n')
        data=data[:-1]
    return pd.DataFrame(data)

@st.cache_resource(ttl='1h00s')
def load_all_data():
    df_data_en = load_corpus('../data/preprocess_txt_en')
    df_data_fr = load_corpus('../data/preprocess_txt_fr')
    lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")
    translation_en_fr = pipeline('translation_en_to_fr', model="t5-base") #, model="Helsinki-NLP/opus-mt-en-fr"
    translation_fr_en = pipeline('translation_fr_to_en', model="Helsinki-NLP/opus-mt-fr-en") #, model="t5-base"
    model_speech = whisper.load_model("base")
    return df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech

n1 = 0
df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier, model_speech = load_all_data() 
lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")

def display_translation(n1, Lang):
    global df_data_src, df_data_tgt, placeholder
    
    with st.status(":sunglasses:", expanded=True):
        s = df_data_src.iloc[n1:n1+5][0].tolist()
        s_trad = []
        s_trad_ref = df_data_tgt.iloc[n1:n1+5][0].tolist()
        source = Lang[:2]
        target = Lang[-2:]
        for i in range(5):
            s_trad.append(translation_model(s[i], max_length=500)[0]['translation_text'].lower())
            st.write("**"+source+"   :**  "+ s[i])
            st.write("**"+target+"   :**  "+s_trad[-1])
            st.write("**ref. :** "+s_trad_ref[i])
            st.write("")
    with placeholder:
        st.write("<p style='text-align:center;background-color:red; color:white')>Score Bleu = "+str(int(round(corpus_bleu(s_trad,[s_trad_ref]).score,0)))+"%</p>", \
            unsafe_allow_html=True)

def run():

    global n1, df_data_src, df_data_tgt, translation_model, placeholder, model_speech
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en

    st.title(title)
    #
    st.write("## **Explications :**\n")

    st.markdown(
        """
        Enfin, nous avons réalisé une traduction :red[**Seq2Seq**] ("Sequence-to-Sequence") avec des :red[**réseaux neuronaux**].  
        La traduction Seq2Seq est une méthode d'apprentissage automatique qui permet de traduire des séquences de texte d'une langue à une autre en utilisant 
        un :red[**encodeur**] pour capturer le sens du texte source, un :red[**décodeur**] pour générer la traduction, et un :red[**vecteur de contexte**] pour relier les deux parties du modèle.
        """
    )
    #
    lang = { 'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el':'modern greek', 'en': 'english', 'es': 'spanish', 'fr': 'french', \
            'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', \
            'th': 'thai', 'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}

    st.write("## **Paramètres :**\n")
    
    choice = st.radio("Choisissez le type de traduction:",["small vocab","Phrases à saisir","Phrases à dicter"], horizontal=True)

    if choice == "small vocab":
        Sens = st.radio('Sens :',('Anglais -> Français','Français -> Anglais'), horizontal=True)
        Lang = ('en_fr' if Sens=='Anglais -> Français' else 'fr_en')

        if (Lang=='en_fr'):
            df_data_src = df_data_en
            df_data_tgt = df_data_fr
            translation_model = translation_en_fr
        else:
            df_data_src = df_data_fr
            df_data_tgt = df_data_en
            translation_model = translation_fr_en

        sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrases à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
        n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
        placeholder = st.empty()
        display_translation(n1, Lang)
    elif choice == "Phrases à saisir":

        custom_sentence = st.text_area(label="Saisir le texte à traduire")
        st.button(label="Valider", type="primary")
        if custom_sentence!="":
            Lang_detected = lang_classifier (custom_sentence)[0]['label']
            st.write('Langue détectée : **'+lang.get(Lang_detected)+'**')
        else: Lang_detected=""
         
        if (Lang_detected=='en'):
            st.write("**fr :**  "+translation_en_fr(custom_sentence, max_length=400)[0]['translation_text'])
        elif (Lang_detected=='fr'):
            st.write("**en  :**  "+translation_fr_en(custom_sentence, max_length=400)[0]['translation_text'])
    elif choice == "Phrases à dicter":
            st.write("Chaque phrase dure 10 secondes maximum")
            st.write("Démarrage de la reconnaissance vocale en temps réel...")
            duration = 10 #seconds
            sampling_rate = 16000
            while True:
                try:
                    audio_input = sd.rec(frames=int(sampling_rate * duration), samplerate=sampling_rate, channels=1)
                    sd.wait()
                    audio_input = audio_input.reshape(sampling_rate * duration,)
                    result = model_speech.transcribe(audio_input)
                    Lang_detected = result["language"]
                    if (Lang_detected=='en'):
                        st.write("**en :**  :blue["+result["text"]+"]")
                        st.write("**fr :**  "+translation_en_fr(result["text"], max_length=400)[0]['translation_text'])
                        st.write("")
                    elif (Lang_detected=='fr'):
                        st.write("**fr :**  :blue["+result["text"]+"]")
                        st.write("**en  :**  "+translation_fr_en(result["text"], max_length=400)[0]['translation_text'])
                        st.write("")
                    else:
                        st.write("**Langue détectée :**  "+lang.get(Lang_detected))
                        st.write("**"+Lang_detected+"  :**  :blue["+result["text"]+"]")
                        st.write("")
                    st.write("Pret pour la phase suivante..")
                except KeyboardInterrupt:
                    st.write("Arrêt de la reconnaissance vocale en temps réel.")
                    break



