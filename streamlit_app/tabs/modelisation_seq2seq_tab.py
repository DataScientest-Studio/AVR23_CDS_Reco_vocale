import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from transformers import pipeline

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
    return df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier

n1 = 0
df_data_en, df_data_fr, translation_en_fr, translation_fr_en, lang_classifier = load_all_data() 
lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")

def display_translation(n1, Lang):
    global df_data_src, df_data_tgt

    for i in range(n1,n1+5):
        s = df_data_src.iloc[i][0]
        source = Lang[:2]
        target = Lang[-2:]
        # for col in s.split():
        #     st.write('col: '+col)
        #     st.write('dict[col]! '+dict[col])
        st.write("**"+source+"   :**  "+ s)
        st.write("**"+target+"   :**  "+translation_model(s, max_length=400)[0]['translation_text'].lower())
        st.write("**ref. :** "+df_data_tgt.iloc[i][0])
        st.write("")


def run():

    global n1, df_data_src, df_data_tgt, translation_model
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en

    st.title(title)
    #
    st.write("## **Explications :**\n")

    st.markdown(
        """
        Enfin, nous avons réalisé une traduction **Seq2Seq** ("Sequence-to-Sequence") avec des réseaux neuronnaux.  
        La traduction Seq2Seq est une méthode d'apprentissage automatique qui permet de traduire des séquences de texte d'une langue à une autre en utilisant 
        un encodeur pour capturer le sens du texte source, un décodeur pour générer la traduction, et un vecteur de contexte pour relier les deux parties du modèle.
        """
    )
    #
    lang = { 'ar': 'arabic', 'bg': 'bulgarian', 'de': 'german', 'el':'modern greek', 'en': 'english', 'es': 'spanish', 'fr': 'french', \
            'hi': 'hindi', 'it': 'italian', 'ja': 'japanese', 'nl': 'dutch', 'pl': 'polish', 'pt': 'portuguese', 'ru': 'russian', 'sw': 'swahili', \
            'th': 'thai', 'tr': 'turkish', 'ur': 'urdu', 'vi': 'vietnamese', 'zh': 'chinese'}

    st.write("## **Paramètres :**\n")
    
    on = st.toggle("Traduction d'une phrase à saisir")

    if not on:
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

    
        sentence1 = st.selectbox("Selectionnez la 1ere des 5 phrase à traduire avec le dictionnaire sélectionné", df_data_src.iloc[:-4],index=int(n1) )
        n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
        display_translation(n1, Lang)
    else:
        custom_sentence = st.text_area(label="Saisir le texte à traduire")
        st.button(label="Valider", type="primary")
        if custom_sentence!="":
            Lang_detected = lang_classifier (custom_sentence)[0]['label']
        else: Lang_detected=""
        if Lang_detected!="":
            st.write('Langue détectée : **'+lang.get(Lang_detected)+'**')
        if (Lang_detected=='en'):
            st.write("**fr :**  "+translation_en_fr(custom_sentence, max_length=400)[0]['translation_text'])
        elif (Lang_detected=='fr'):
            st.write("**en  :**  "+translation_fr_en(custom_sentence, max_length=400)[0]['translation_text'])




    """
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Prepare the text
text = "translate English to French: The quick brown fox jumps over the lazy dog"

# Tokenize the input
inputs = tokenizer.encode(text, return_tensors='pt')

# Generate translation
outputs = model.generate(inputs, max_length=40, num_beams=4, early_stopping=True)

# Decode the output
translated_text = tokenizer.decode(outputs[0])

print(translated_text)
"""
"""
from transformers import AutoTokenizer, AutoModelWithLMHead, TranslationPipeline

pipeline = TranslationPipeline(
model=AutoModelWithLMHead.from_pretrained("SEBIS/legal_t5_small_trans_fr_en"),
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "SEBIS/legal_t5_small_trans_fr_en", do_lower_case=False, 
                                            skip_special_tokens=True),
    device=0
)

fr_text = "quels montants ont été attribués et quelles sommes ont été effectivement utilisées dans chaque État membre? 4."

pipeline([fr_text], max_length=512)
"""