import streamlit as st
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import torch as t
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from interpretathon import get_cosine_similarity, get_dot_product, run, str_to_feature_pair
from fetchDescriptions import generate_url_sae, request, scrape_data


if __name__ == "__main__":

    st.title("Track to the Feature")
    
    if "URL_CACHE" not in st.session_state:
        st.session_state["URL_CACHE"] = {}

    if "MODEL_LOADED" not in st.session_state:
        with st.spinner("Model loading in progress..."):
            #device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
            saes, _ = get_gpt2_res_jb_saes()
            gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
            
            st.session_state["SAES"] = saes
            st.session_state["MODEL"] = gpt2_small
            st.session_state["MODEL_LOADED"] = True
    
    prompt = st.text_input("please enter your prompt", value="This movie is amazing! The best I have ever seen")
    features = st.text_input("please enter a feature", value="8.23510")
    
    col1, col2 = st.columns(2)
    
    with col1:
        similarity_options = ["Dot Product", "Cosine Similarity"]
        similarity_func = st.selectbox("Select similarity function", options=similarity_options, index=0)
        eos = st.checkbox("Include EOS token", value=False)
    
    with col2:
        model_options = ["Decoder", "Encoder"]
        model_selection = st.selectbox("Select direction type", options=model_options, index=0)
    
    

    use_dot = similarity_func == "Dot Product"
    dec = model_selection == "Decoder"
    
    func = get_dot_product if use_dot else get_cosine_similarity
    
    if prompt:
        
        with st.spinner("Creating Plot..."):
            
            f = str_to_feature_pair(features)
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
        st.subheader("Feature descriptions from neuronpedia.org")
        for feature in features.split(","):
            l, f = feature.split(".")
            #url = generate_url("gpt2-small", l.strip(), f.strip())
            
            sae_with_layer = f"{l.strip()}-res-jb"
            url = generate_url_sae("gpt2-small", sae_with_layer, f.strip())

            if url not in st.session_state["URL_CACHE"]:
                d = request(url).text
                explanation = scrape_data(d)
                if not explanation:
                    explanation = "<ERROR, Explanation not found>"
                st.session_state["URL_CACHE"][url] = f"* {l}.{f}: {explanation} \n(source: {url})"

            st.markdown(st.session_state["URL_CACHE"][url])
        
    # st.button("Calculate!", key="run")