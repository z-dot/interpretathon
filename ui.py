import streamlit as st
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import torch as t
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from interpretathon import get_cosine_similarity, get_dot_product, run, str_to_feature_pair


def plot(data, title="", xlabels=None, ylabels=None, minv=None, maxv=None):
    fig, ax = plt.subplots()

    ax.imshow(data, vmin=minv, vmax=maxv, cmap='RdBu')
    ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
    ax.set_xlabel("Layers")
    ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
            
    ax.set_title(title)
    fig.tight_layout()
    
    return fig, ax    


if __name__ == "__main__":

    st.title("Testing 123")

    if "MODEL_LOADED" not in st.session_state:
        with st.spinner("Model Loading in progress..."):
            #device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
            saes, _ = get_gpt2_res_jb_saes()
            gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
            
            st.session_state["SAES"] = saes
            st.session_state["MODEL"] = gpt2_small
            st.session_state["MODEL_LOADED"] = True
            #st.session_state["DEVICE"] = device
    
    prompt = st.text_input("please enter your prompt", value="This movie is amazing! The best I have ever seen")
    feature = st.text_input("please enter a feature", value="8.23510")
    
    
    use_dot = st.toggle("Use Dot Product or Cosine Similarity", value=True)
    st.write(f"{'Dot product' if use_dot else 'Cosine Similarity'} selected")
    func = get_dot_product if use_dot else get_cosine_similarity
    
    st.button("Calculate!", key="run")
    
    if st.session_state["run"]:
        
        #x = st.session_state["x"]
        
        #if use_dot:
            #todo
        #    minv, maxv = None, None
        #else:
        #    minv, maxv = -1, 1
        
        with st.spinner("Creating Plot..."):
            
            f = str_to_feature_pair(feature)
            fig = run(st.session_state["MODEL"], st.session_state["SAES"], prompt, f, func)  
            st.plotly_chart(fig)
            
            #layers = range(x.model.cfg.n_layers)
            #prompt_tok = x.model.to_str_tokens(prompt)
        
            #data = x.run(x.model.to_tokens(prompt), func, x.get_feature_from_name(feature).to(st.session_state["device"])).detach().cpu()
            
            #plot_title = f"Activations of the prompt for feature '{feature}', measured with {'dot product' if use_dot else 'cosine similarity'}"
            #fig, ax = plot(data, title=plot_title, xlabels=layers, ylabels=prompt_tok, minv=-1, maxv=1)
            #st.pyplot(fig)