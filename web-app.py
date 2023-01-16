
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model


st.title("Malaria Detection Using blood smear image")
file = st.file_uploader("Upload file",type=["png"])
show_file = st.empty()
if not file:
    show_file.info("Please upload a file")
else:

    show_file.image(file)

if st.button("View Result"):
    
    model = load_model("C:\\Users\\M\\Desktop\\MinorApp\\epoches5.h5")
    x = Image.open(file)
    size = (224,224)
    x = ImageOps.fit(x, size, Image.ANTIALIAS)

    # x = image.img_to_array(x)
    x = np.array(x)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)

    # x = preprocess_input(x)

    preds1 = model.predict(x)

    
    preds=np.argmax(preds1, axis=1)
    if preds==0:
        st.error("The Person is Infected With malaria")
    else:
        st.success("The Person is not Infected With malaria")


