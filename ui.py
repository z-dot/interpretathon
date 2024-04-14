import streamlit as st
from typing import List
import matplotlib.pyplot as plt
import numpy as np

import torch as t
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from interpretathon import X, get_cosine_similarity, get_dot_product


def generate_fake_data(num_rows, num_cols) -> np.array:
    return np.round(np.random.rand(num_rows, num_cols), 2)

def get_layer_list() -> List[str]:
    """
    Returns:
        List: names of layers
    """
    return list(range(0, 4))

def tokenize_prompt(prompt:str) -> List[str]:
    """
    Args:
        prompt (str): 

    Returns:
        List[str]: list of tokens
    """
    return prompt.split(" ")


if __name__ == "__main__":
    device = t.device("cuda") if t.cuda.is_available() else t.device("mps")
    t.set_default_device(device)
    saes, sparsities = get_gpt2_res_jb_saes()
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

    st.title("Testing 123")
    
    prompt = st.text_input("please enter your prompt")

    x = X(gpt2_small, saes)
    
    #layers = st.selectbox("Choose a Layer", get_layer_list())
    use_dot = st.toggle("Use Dot Product or Cosine Similarity", value=True)
    st.write(f"{'Dot product' if use_dot else 'Cosine Similarity'} selected")
    
    func = get_dot_product if use_dot else get_cosine_similarity
    
    
    features = st.selectbox("choose feature", ['0.4']) # st.selectbox("Choose a Feature", x.get_feature_names())
    
    layers = range(x.model.cfg.n_layers)
    prompt_tok = x.model.to_str_tokens(prompt)
    
    # data = generate_fake_data(len(prompt_tok), len(layers))
    data = x.run(x.model.to_tokens(prompt), func, x.get_feature_from_name(features).to(device)).detach().cpu()
    
    #st.write(data)

    assert tuple(data.shape) == (len(prompt_tok), len(layers))

    fig, ax = plt.subplots()
    im = ax.imshow(data)    
    ax.set_xticks(np.arange(len(layers)), labels=layers)
    ax.set_xlabel("Layers")
    ax.set_yticks(np.arange(len(prompt_tok)), labels=prompt_tok)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(prompt_tok)):
        for j in range(len(layers)):
            text = ax.text(j, i, data[i, j],
                        ha="center", va="center", color="w")
            
    ax.set_title("Gimme a proper title")
    fig.tight_layout()
    st.pyplot(fig)
    
    #toggle dot/cosine