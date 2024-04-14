import streamlit as st
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import torch as t
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from interpretathon import get_cosine_similarity, get_dot_product, run, str_to_feature_pair

            

if __name__ == "__main__":

    st.title("Track to the Feature")

    if "MODEL_LOADED" not in st.session_state:
        with st.spinner("Model loading in progress..."):
            #device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
            saes, _ = get_gpt2_res_jb_saes()
            gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
            
            st.session_state["SAES"] = saes
            st.session_state["MODEL"] = gpt2_small
            st.session_state["MODEL_LOADED"] = True
    
    
    prompt = st.text_input("Please enter your prompt", value="This movie is amazing! The best I have ever seen")
    feature = st.text_input("Please enter a feature", value="8.23510")
    
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_options = ["Dot Product", "Cosine Similarity"]
        similarity_func = st.selectbox("Select similarity function", options=similarity_options, index=0)
        eos = st.checkbox("Include EOS token", value=False)
    
    with col2:
        model_options = ["Decoder", "Encoder"]
        model_selection = st.selectbox("Select model", options=model_options, index=0)
    
    

    use_dot = similarity_func == "Dot Product"
    dec = model_selection == "Decoder"
    
    func = get_dot_product if use_dot else get_cosine_similarity
    
    if prompt:
        with st.spinner("Creating Plot..."):
            
            f = str_to_feature_pair(feature)
            fig = run(
                st.session_state["MODEL"],
                st.session_state["SAES"],
                prompt,
                f,
                func,
                dec=dec,
                normalized=(not use_dot),
                eos=eos
            )  
            st.plotly_chart(fig, True)
    # st.button("Calculate!", key="run")