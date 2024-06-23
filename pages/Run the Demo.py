from model_init import model_init
from visualise import *
from ig import IG
from blurig import BlurIG
import matplotlib.pyplot as plt
from vg import VG
from smoothgrad import SmoothGrad
import streamlit as st
from PIL import Image
import numpy as np
import json
import torch

st.title("Running the demo")



def run_demo():
    with open('savestate.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        modelstr = data['model']
        methodstr = data['method']
        smoothgradstr = data['smoothgrad']
        idgistr = data['idgi']
        img = data['image']
        classstr = data['class']
        classnum = data['classnum']
        steps = data['steps']
        baseline = data['baseline']
        max_sig = data['max_sig']
        grad_step = data['grad_step']
        sqrt = data['sqrt']
        noise_steps = data['noise_steps']
        noise_var = data['noise_var']
        steps_at = data['steps_at']
    st.write("The saliency map is being generated. Please wait for a few seconds...")
    preprocess, model, load = model_init(modelstr.lower())
    st.write("Model loaded successfully")
    image_ = np.load("image.npy") if img == "Uploaded" else None
    st.write("Image loaded successfully")
    classnum = classnum
    if classnum == -1:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        classnum = model(preprocess(torch.Tensor(load(image_, False))).to(device)).argmax().item() if image_ is not None else 236
    if methodstr == "Integrated Gradients":
        if (smoothgradstr == 'Yes'):
            ig = SmoothGrad(model, load, preprocess, IG)
            saliency_base, saliency_idgi = ig.smoothsaliency(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_, steps = steps, baseline = baseline, noise_steps = noise_steps, noise_var = noise_var)
        else:
            ig = IG(model, load, preprocess)
            saliency_base, saliency_idgi = ig.saliency(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_, steps=steps, baseline=baseline)
    elif methodstr == "Blur Integrated Gradients":
        if (smoothgradstr == 'Yes'):
            blurig = SmoothGrad(model, load, preprocess, BlurIG)
            saliency_base, saliency_idgi = blurig.smoothsaliency(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_)
        else:
            blurig = BlurIG(model, load, preprocess)
            saliency_base, saliency_idgi = blurig.saliency(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_)
    elif methodstr == "Vanilla Gradients":
        if (smoothgradstr == 'Yes'):
            vg = SmoothGrad(model, load, preprocess, VG)
            saliency_base = vg.smoothvg(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_)
        else:
            vg = VG(model, load, preprocess)
            saliency_base = vg.saliency(x_values_paths = ["doberman.png"], prediction_class = [classnum], x_values=image_)
    st.write("Saliency map generated successfully")
    if idgistr == 'Yes' and methodstr != "Vanilla Gradients":
        title = "Saliency Map for " + methodstr
        if smoothgradstr == 'Yes':
            title += " and SmoothGrad"
        if idgistr == 'Yes':
            title += " and IDGI"
        title += f" (Class : {classnum})"
        st.pyplot(Visualise(saliency_idgi, title = title))
    else:
        title = "Saliency Map for " + methodstr
        if smoothgradstr == 'Yes':
            title += " and SmoothGrad"
        title += f" (Class : {classnum})"
        st.pyplot(Visualise(saliency_base, title = title))
    st.write("The saliency map has been visualised successfully")
    st.write("You can now choose to run the demo again with different options or go back to the initialisation page to change the options")
    if st.button("Run the Demo Again"):
        st.switch_page("pages/Run the Demo.py")
    if st.button("Initialise the Demo - I"):
        st.switch_page("pages/Initialise the Demo - I.py")
    if st.button("Initialise the Demo - II"):
        st.switch_page("pages/Initialise the Demo - II.py")

if st.button("Run the Demo"):
    run_demo()
    pass