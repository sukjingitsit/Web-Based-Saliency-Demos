import streamlit as st
from PIL import Image
import numpy as np
import json

st.title('Initialising the Demo - I')

st.write("""The first step is to choose some basic options related to the explainability model you will be using
        This includes choosing the saliency method, the model and the image you want to explain.""")

modelstr = st.radio(label="Which CNN Model do you want to explain:",
        options=['InceptionV3', 'ResNet50V2', 'ResNet101V2', 'Resnet152V2', 'MobileNetV2', 'VGG16', 'VGG19', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'Xception'],
        index=0)

methodstr = st.radio(label="Which Saliency Method do you want to use:",
            options=['Integrated Gradients', 'Blur Integrated Gradients', 'Vanilla Gradients'],
            index=0)

smoothgradstr = st.radio(label="Do you want to use the SmoothGrad method in addition to the chosen method?",
                    options=['Yes', 'No'],
                    index=1)

idgistr = st.radio(label="Do you want to use the IDGI method in addition to the chosen method (ignored for Vanilla Gradients)?",
                    options=['Yes', 'No'],
                    index=1)

img = st.file_uploader("Upload the image for which you want to create the saliency map here...", type=['png', 'jpeg'])
st.write("An image of a doberman dog will be chosen as the default image for the saliency map if none is uploaded")
if img is not None:
    image = st.image(img, caption="Image chosen for saliency")
    pilimage = Image.open(img).convert("RGB")
    data = np.asarray(pilimage)
    np.save("image.npy", data)
else:
    image = st.image("doberman.png", caption="Default Image")

classstr = st.radio(label="What class do you want to evaluate the saliency map for?",
            options=['Top Class', 'Random Class', 'User Defined Class'],
            index=0)

if (classstr == 'User Defined Class'):
    classnum = st.number_input("Enter the class number you want to evaluate the saliency map for:", min_value=0, max_value=999, value=0)
if (classstr == 'Top Class'):
    classnum = -1
if (classstr == 'Random Class'):
    classnum = np.random.randint(0, 1000)
    st.write(f"Random class number chosen: {classnum}")

st.text(f"""Have you selected all the options correctly to generate the saliency map:
                    Model: {modelstr}
                    Method: {methodstr}
                    SmoothGrad: {smoothgradstr}
                    IDGI: {idgistr}
                    Image: {"Uploaded" if img is not None else "Default"}
                    Class: {classstr}""")

def init_ii():
    with open('savestate.json', 'r+', encoding='utf-8') as f:
        data = json.load(f)
        data.update({
            "model": modelstr,
            "method": methodstr,
            "smoothgrad": smoothgradstr,
            "idgi": idgistr,
            "image": "Uploaded" if img is not None else "Default",
            "class": classstr,
            "classnum": classnum,
        })
        f.seek(0)
        f.truncate()
        json.dump(data, f, ensure_ascii=False, indent=4)
    st.switch_page("pages/Initialise the Demo - II.py")

if st.button("Next step of Initialisation"):
    init_ii()
if st.button("Run the Demo"):
    st.switch_page("pages/Run the Demo.py")