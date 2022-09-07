import streamlit as st
import os
import numpy as np

from src.modelling import Model
from src.result import Detect
import src.const as const

def prepopulate_textarea():
    if st.session_state.check:
        populate()
    else:
        st.session_state.prepopulated = ""
        st.session_state.label = ""

def populate():
    mdl = Model()
    prepopulated, label = mdl.get_random()
    st.session_state.prepopulated = prepopulated
    st.session_state.label = label

st.set_page_config(page_title='LangD', page_icon='', layout='wide')
st.header('Language Detector')

mdl = Model()
det = Detect()

if not os.path.exists(os.path.join(os.getcwd(),const.MODEL_FILE)):
    mdl.train()

if 'prepopulated' not in st.session_state or 'label' not in st.session_state:
    st.session_state.prepopulated = ""
    st.session_state.label = ""

c1, _, c2 = st.columns([2,0.5,1.5])
with c1:
    text = st.text_area(label='Text to Detect', value=st.session_state.prepopulated, height=280)
c3, c4, _ = st.columns([4,1,5])
with c3:
    rand = st.checkbox(label='Populate random text', on_change=prepopulate_textarea, key='check')
if rand:
    with c4:
        nxt = st.button(label='Next', on_click=populate)
    with c3:
        st.info(f"The language of auto populated text is : {st.session_state.label}")
    
with c2:
    if text != "":
        lang, fig = det.detect(text)
        st.markdown(f"<h3 style='text-align:center;'>{lang}</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

