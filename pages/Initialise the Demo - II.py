import streamlit as st
from PIL import Image
import numpy as np
import json

st.title('Initialising the Demo - II')

st.write("The second step is to choose some additional options related to the explainability model you will be using")

with open('savestate.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    modelstr = data['model']
    methodstr = data['method']
    smoothgradstr = data['smoothgrad']
    idgistr = data['idgi']
    img = data['image']
    classstr = data['class']
    classnum = data['classnum']

st.text(f"""The currently selected options for the saliency map are:
            Model: {modelstr}
            Method: {methodstr}
            SmoothGrad: {smoothgradstr}
            IDGI: {idgistr}
            Image: {"Uploaded" if img is not None else "Default"}
            Class: {classstr}
If the options need to be changed, please go to the 'Initialise the Demo - I' page.
Based on the selected options, you can now choose the following additional options:""")

if (methodstr == 'Integrated Gradients'):
    baseline = st.radio("Which baseline do you want to use for Integrated Gradients:",
             options=['black', 'white', 'random'],
            index=0)
    steps = st.slider("Number of steps for Integrated Gradients (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=20)
    with open('savestate.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.update({
            "baseline": baseline,
            "steps": steps
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=4)

if (methodstr == 'Blur Integrated Gradients'):
    steps = st.slider("Number of steps for Blur Integrated Gradients (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=20)
    max_sig = st.slider("Maximum sigma for the Gaussian blur in Blur Integrated Gradients", min_value=0.1, max_value=1.0, value=0.5)
    grad_step = st.slider("Gradient step size for Blur Integrated Gradients (smaller step size may lead to better results but also increased chances of numerical error)", min_value=0.001, max_value=0.1, value=0.01)
    sqrt = st.radio("Do you want to use square root adjust or non-adjusted Blur Integrated Gradients (setting it to Yes may lead to erroneous behavior in some case)?",
             options=['Yes', 'No'],
             index=1)
    with open('savestate.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.update({
            "grad_step": grad_step,
            "steps": steps,
            "max_sig": max_sig,
            "sqrt": sqrt
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=4)

if (smoothgradstr == 'Yes'):
    noise_steps = st.slider("Number of noise steps for SmoothGrad (more steps may lead to better results but also increased inference time)", min_value=1, max_value=100, value=20)
    noise_var = st.slider("Noise variance for SmoothGrad (higher variance may lead to worse results but if it is low, SmoothGrad may not give the desired results)", min_value=0.01, max_value=1.0, value=0.1)
    with open('savestate.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.update({
            "noise_steps": noise_steps,
            "noise_var": noise_var
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=4)

st.write("Once you have selected all the options, you can run the demo on the page titled 'Run the Demo' by clicking on the button below")
if st.button("Run the Demo"):
    st.switch_page("pages/Run the Demo.py")
