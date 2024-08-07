import streamlit as st
from PIL import Image
import numpy as np
import json
from session_state import SessionState

if 'savestate' not in st.session_state:
    st.session_state['savestate'] = SessionState()

st.title('Interpretability of Saliency-Based Models')

st.write("""The goal of this project is to explore the interpretability of saliency-based models.
        The first step is to choose some basic options related to the explainability model you will be using
        This includes choosing the saliency method, the model and the image you want to explain. This is done
        on the page titled 'Initialise the Demo - I'. Therafter, you can customise the demo further in the
        second part of the demo on the page titled 'Initialise the Demo - II' where you can choose options
        like number of steps, baseline, etc. Finally, you can run the demo on the page titled 'Run the Demo'.
        You can also directly run the demo by clicking on the 'Run the Demo' button on the sidebar, in which case
        a default selection of parameters will be chosen""")

def init_demo():
    st.switch_page("pages/Initialise the Demo - I.py")
def run_demo():
    st.switch_page("pages/Run the Demo.py")

if st.button("Initialise the Demo - I"):
    init_demo()
if st.button("Run the Demo"):
    run_demo()
