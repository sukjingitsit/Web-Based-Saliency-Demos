import streamlit as st
from PIL import Image
import numpy as np
import json
from session_state import SessionState

if 'savestate' not in st.session_state:
    st.session_state['savestate'] = SessionState()
savestate = st.session_state['savestate']

st.title('Initialising the Demo - II')

st.write("The second step is to choose some additional options related to the explainability model you will be using")

st.text(f"""The currently selected options for the saliency map are:
            Model: {savestate.model}
            Method: {savestate.method}
            SmoothGrad: {('Yes' if savestate.smoothgrad else 'No')}
            IDGI: {('Yes' if savestate.idgi else 'No')}
            Image: {("Uploaded" if savestate.image else "Default")}
            Class: {savestate.classchoice}
If the options need to be changed, please go to the 'Initialise the Demo - I' page.
Based on the selected options, you can now choose the following additional options:""")

if (savestate.model == 'Integrated Gradients'):
    baseline = st.radio("Which baseline do you want to use for Integrated Gradients:",
             options=['black', 'white', 'random'],
            index=(0 if savestate.baseline == 'black' else 1 if savestate.baseline == 'white' else 2))
    steps = st.slider("Number of steps for Integrated Gradients (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=savestate.steps)
    savestate.update(steps=steps, baseline=baseline)

if (savestate.model == 'Blur Integrated Gradients'):
    steps = st.slider("Number of steps for Blur Integrated Gradients (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=savestate.steps)
    max_sig = st.slider("Maximum sigma for the Gaussian blur in Blur Integrated Gradients", min_value=0.1, max_value=1.0, value=savestate.max_sig)
    grad_step = st.slider("Gradient step size for Blur Integrated Gradients (smaller step size may lead to better results but also increased chances of numerical error)", min_value=0.001, max_value=0.1, value=savestate.grad_step)
    sqrt = st.radio("Do you want to use square root adjust or non-adjusted Blur Integrated Gradients (setting it to Yes may lead to erroneous behavior in some case)?",
             options=['Yes', 'No'],
             index=(0 if savestate.sqrt else 1))
    savestate.update(steps=steps, max_sig=max_sig, grad_step=grad_step, sqrt=sqrt)

if (savestate.smoothgrad):
    noise_steps = st.slider("Number of noise steps for SmoothGrad (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=savestate.noise_steps)
    noise_var = st.slider("Noise variance for SmoothGrad (higher variance may lead to worse results but if it is low, SmoothGrad may not give the desired results)", min_value=0.01, max_value=1.0, value=savestate.noise_var)
    savestate.update(noise_steps=noise_steps, noise_var=noise_var)

st.write("Once you have selected all the options, you can run the demo on the page titled 'Run the Demo' by clicking on the button below")
if st.button("Run the Demo"):
    st.switch_page("pages/Run the Demo.py")
