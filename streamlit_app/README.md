![Alt Whaooh!](./streamlit_app/assets/miss-honey-glasses-off.gif)
# Installation Guide of the Stremalit App

You can  run the **Streamlit App directly on the Cloud**: 
**[Translation System for Connected Glasses](https://demosthene-or-avr23-cds-translation.hf.space/)**

To run the **app locally on your computer** (be careful with the paths of the files in the app):
```shell
conda create --name Avr23-cds-translation python=3.10
conda activate Avr23-cds-translation
cd "folder where the file app.py is located"
pip install -r requirements.txt
# if your operating system is Windows 11, run the following line:
pip install protobuf==3.20.3 streamlit==1.28.0

streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
