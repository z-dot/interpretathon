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
    
    prompt = st.text_input("please enter your prompt", value="This movie is amazing! The best I have ever seen")
    feature = st.text_input("please enter a feature", value="8.23510")
    
    
    use_dot = st.toggle("Use Dot Product or Cosine Similarity", value=True)
    eos = st.toggle("Include EOS token", value=True)
    st.write(f"{'Dot product' if use_dot else 'Cosine Similarity'} selected")
    func = get_dot_product if use_dot else get_cosine_similarity
    
    st.button("Calculate!", key="run")
    
    if st.session_state["run"]:
        
        with st.spinner("Creating Plot..."):
            
            f = str_to_feature_pair(feature)
            fig = run(
                st.session_state["MODEL"],
                st.session_state["SAES"],
                prompt,
                f,
                func,
                normalized=(not use_dot),
                eos=eos
            )  
            st.plotly_chart(fig, True)
            