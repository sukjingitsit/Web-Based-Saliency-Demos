from utils.model_init import model_init
from utils.visualise import *
from utils.ig import IG
from utils.blurig import BlurIG
import matplotlib.pyplot as plt
from utils.vg import VG
from utils.smoothgrad import SmoothGrad
import streamlit as st
from PIL import Image
import numpy as np
import json
import torch
from session_state import SessionState

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else device
device = torch.device("cpu") if device == torch.device("mps") else device

if 'savestate' not in st.session_state:
    st.session_state['savestate'] = SessionState()
savestate = st.session_state['savestate']

st.title("Running the demo")

def run_demo():
    st.write("The saliency map is being generated. Please wait for a few seconds...")
    modelstr = savestate.model
    preprocess, model, load = model_init(modelstr.lower())
    st.write("Model loaded successfully: ", modelstr)
    image_ = savestate.image_arr
    st.write("Image loaded successfully")
    if image_:
        st.image(image_, caption="Image for Saliency Map", use_column_width=True)
    else:
        st.image("doberman.png", caption="Image for Saliency Map", use_column_width=True)
    classnum = savestate.classnum
    if classnum == -1:
        classnum = model(preprocess(torch.Tensor(load(image_, False))).to(device)).argmax().item() if image_ is not None else 236
    if savestate.method == "Integrated Gradients":
        if (savestate.smoothgrad):
            ig = SmoothGrad(model, load, preprocess, IG)
            saliency_base, saliency_idgi = ig.smoothsaliency(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_, steps=savestate.steps, baseline=savestate.baseline, noise_steps=savestate.noise_steps, noise_var=savestate.noise_var)
        else:
            ig = IG(model, load, preprocess)
            saliency_base, saliency_idgi = ig.saliency(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_, steps=savestate.steps, baseline=savestate.baseline)
    elif savestate.method == "Blur Integrated Gradients":
        if (savestate.smoothgrad):
            blurig = SmoothGrad(model, load, preprocess, BlurIG)
            saliency_base, saliency_idgi = blurig.smoothsaliency(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_, steps=savestate.steps, max_sig=savestate.max_sig, grad_step=savestate.grad_step, sqrt=savestate.sqrt, noise_steps=savestate.noise_steps, noise_var=savestate.noise_var)
        else:
            blurig = BlurIG(model, load, preprocess)
            saliency_base, saliency_idgi = blurig.saliency(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_, steps=savestate.steps, max_sig=savestate.max_sig, grad_step=savestate.grad_step, sqrt=savestate.sqrt)
    elif savestate.method == "Vanilla Gradients":
        if (savestate.smoothgrad):
            vg = SmoothGrad(model, load, preprocess, VG)
            saliency_base = vg.smoothvg(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_, noise_steps=savestate.noise_steps, noise_var=savestate.noise_var)
        else:
            vg = VG(model, load, preprocess)
            saliency_base = vg.saliency(x_values_paths=["doberman.png"], prediction_class=[classnum], x_values=image_)
    st.write("Saliency map generated successfully")
    if savestate.idgi and savestate.method != "Vanilla Gradients":
        title = "Saliency Map for " + savestate.method
        if savestate.smoothgrad:
            title += " and SmoothGrad"
        title += " and IDGI"
        title += f" (Class : {classnum})"
        st.pyplot(Visualise(saliency_idgi, title = title))
    else:
        title = "Saliency Map for " + savestate.method
        if savestate.smoothgrad:
            title += " and SmoothGrad"
        title += f" (Class : {classnum})"
        st.pyplot(Visualise(saliency_base, title = title))
    st.write("The saliency map has been visualised successfully")
    st.write("You can now choose to run the demo again with the same options or go back to the initialisation page to change the options")
    if st.button("Run the Demo Again"):
        st.switch_page("pages/Run the Demo.py")
    if st.button("Initialise the Demo - I"):
        st.switch_page("pages/Initialise the Demo - I.py")
    if st.button("Initialise the Demo - II"):
        st.switch_page("pages/Initialise the Demo - II.py")

if st.button("Run the Demo"):
    run_demo()
    pass