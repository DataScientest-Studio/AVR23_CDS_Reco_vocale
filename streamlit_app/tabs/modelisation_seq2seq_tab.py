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

n1 = 0
df_data_en = load_corpus('../data/preprocess_txt_en')
df_data_fr = load_corpus('../data/preprocess_txt_fr')
translation_en_fr = pipeline('translation_en_to_fr', model="t5-base")
translation_fr_en = pipeline('translation_fr_to_en', model="t5-base")


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
        st.write("**"+target+"   :**  "+translation_model(s))
        st.write("**ref. :** "+df_data_tgt.iloc[i][0])
        st.write("")


def run():

    global n1, df_data_src, df_data_tgt, translation_model
    global df_data_en, df_data_fr, lang_classifier, translation_en_fr, translation_fr_en

    st.title(title)

    st.write("## **Données d'entrée :**\n")
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


    n1 = df_data_src[df_data_src[0]==sentence1].index.values[0]
    lang_classifier = pipeline('text-classification',model="papluca/xlm-roberta-base-language-detection")

    Lang_detected = lang_classifier (sentence1)
    st.write(Lang_detected)
    display_translation(n1, Lang)


    st.markdown(
        """
        This is your app's second tab. Fill it in `tabs/second_tab.py`.
        You can and probably should rename the file.

        ## Test

        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse gravida urna vel tincidunt vestibulum. Nunc malesuada molestie odio, vel tincidunt arcu fringilla hendrerit. Sed leo velit, elementum nec ipsum id, sagittis tempus leo. Quisque viverra ipsum arcu, et ullamcorper arcu volutpat maximus. Donec volutpat porttitor mi in tincidunt. Ut sodales commodo magna, eu volutpat lacus sodales in. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Aliquam interdum libero non leo iaculis bibendum. Suspendisse in leo posuere risus viverra suscipit.

        Nunc eu tortor dolor. Etiam molestie id enim ut convallis. Pellentesque aliquet malesuada ipsum eget commodo. Ut at eros elit. Quisque non blandit magna. Aliquam porta, turpis ac maximus varius, risus elit sagittis leo, eu interdum lorem leo sit amet sapien. Nam vestibulum cursus magna, a dapibus augue pellentesque sed. Integer tincidunt scelerisque urna non viverra. Sed faucibus leo augue, ac suscipit orci cursus sed. Mauris sit amet consectetur nisi.
        """
    )

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))

    st.line_chart(chart_data)

    st.markdown(
        """
        ## Test 2

        Proin malesuada diam blandit orci auctor, ac auctor lacus porttitor. Aenean id faucibus tortor. Morbi ac odio leo. Proin consequat facilisis magna eu elementum. Proin arcu sapien, venenatis placerat blandit vitae, pharetra ac ipsum. Proin interdum purus non eros condimentum, sit amet luctus quam iaculis. Quisque vitae sapien felis. Vivamus ut tortor accumsan, dictum mi a, semper libero. Morbi sed fermentum ligula, quis varius quam. Suspendisse rutrum, sapien at scelerisque vestibulum, ipsum nibh fermentum odio, vel pellentesque arcu erat at sapien. Maecenas aliquam eget metus ut interdum.
        
        ```python

        def my_awesome_function(a, b):
            return a + b
        ```

        Sed lacinia suscipit turpis sit amet gravida. Etiam quis purus in magna elementum malesuada. Nullam fermentum, sapien a maximus pharetra, mauris tortor maximus velit, a tempus dolor elit ut lectus. Cras ut nulla eget dolor malesuada congue. Quisque placerat, nulla in pharetra dapibus, nunc ligula semper massa, eu euismod dui risus non metus. Curabitur pretium lorem vel luctus dictum. Maecenas a dui in odio congue interdum. Sed massa est, rutrum eu risus et, pharetra pulvinar lorem.
        """
    )

    st.area_chart(chart_data)

    st.markdown(
        """
        ## Test 3

        You can also display images using [Pillow](https://pillow.readthedocs.io/en/stable/index.html).

        ```python
        import streamlit as st
        from PIL import Image

        st.image(Image.open("assets/sample-image.jpg"))

        ```

        """
    )

    st.image(Image.open("assets/sample-image.jpg"))
