import streamlit as st 
from fastai.vision.all import *
import plotly.express as px
import pathlib
temp =pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Mevalar")

file = st.file_uploader("Rasm yuklash",type=['png','jpeg','svg','jfif'])


if file:
    st.image(file)
    img=PILImage.create(file)

    model1 = load_learner('mevalar.pkl')

    pred, pred_id,probs = model1.predict(img)
    st.success(f"Bashorat: {pred}")
    sf.info(f"Aniqlilik darajasi: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x= probs*100, y= model1.dls.vocab)
    st.plotly_chart(fig)
