import streamlit as st
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def generate_fake_data(num_rows, num_cols) -> np.array:
    return np.round(np.random.rand(num_rows, num_cols), 2)

def get_feature_list() -> List[str]:
    """
    Returns:
        List: name of features
    """
    return ["a", "b", "c"]

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
    st.title("Testing 123")
    
    prompt = st.text_input("please enter your prompt")
    
    #layers = st.selectbox("Choose a Layer", get_layer_list())
    features = st.selectbox("Choose a Feature", get_feature_list())
    
    layers = get_layer_list()
    prompt_tok = tokenize_prompt(prompt)
    
    data = generate_fake_data(len(prompt_tok), len(layers))
    
    #st.write(data)

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