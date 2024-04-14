from cgitb import Hook
from scipy import sparse
import torch as t
from torch import Tensor
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Sequence, Tuple, Dict, Literal, Set, Union
import einops
import streamlit as st

from huggingface_hub import hf_hub_download

if t.cuda.is_available():
    device = t.device("cuda")
elif t.backends.mps.is_available():
    device = t.device("mps")
else:
    device = t.device("cpu")

def get_all_activations(model: HookedTransformer, cache: ActivationCache) -> Float[Tensor, "layer seq_pos d_model"]:
    return t.stack([cache["resid_pre", i] for i in range(model.cfg.n_layers)])

def get_dot_product(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layer _seq_pos"]]:
    return [einops.einsum(activations, features, "l t d, f d -> f l t") for activations in activations_list]

def get_cosine_similarity(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layer _seq_pos"]]:
    return [t.nn.functional.cosine_similarity(einops.rearrange(activation, "l s d -> 1 d l s"), features[:, :, None, None], dim=1) for activation in activations_list]


def get_feature(sae: dict[str, SparseAutoencoder], layer: int, feature: int, decoder: bool = True) -> Float[Tensor, "d_model"]:
    if decoder:
        return sae[f"blocks.{layer}.hook_resid_pre"].W_dec[feature]
    return sae[f"blocks.{layer}.hook_resid_pre"].W_enc[feature]

def get_feature_movement(
        model: HookedTransformer,
        sae: dict[str, SparseAutoencoder],
        prompts: List[str],
        func: Callable[
            [List[Float[Tensor, "layer _seq_pos d_model"]], Float[Tensor, "features d_model"]],
            List[Float[Tensor, "features layers _seq_pos"]]
        ],
        features: Union[List[Tuple[int, int]], Float[Tensor, 'features d_model']],
        decoder: bool = True,
) -> List[Float[Tensor, "features layer _seq_pos"]]:
    if isinstance(features, list) and all(isinstance(item, tuple) for item in features):
        feats: Float[Tensor, "features d_model"] = t.stack([get_feature(sae, layer, feature, decoder) for (layer, feature) in features])
    elif isinstance(features, Tensor):
        feats = features
    with t.no_grad():
        all_activations = [
            get_all_activations(
                model,
                model.run_with_cache(
                    model.to_tokens(prompt),
                    remove_batch_dim=True
                )[1]
            ).to(device)
            for prompt
            in prompts
        ]
        
        return func(all_activations, feats.to(device))



def plot_stacked_heatmaps_flipped(tensor, normalized=True, x_axis_names=None, y_axis_ticks=None):
    z_data = tensor.numpy()  # Convert tensor to numpy array
    
    if y_axis_ticks is None:
        y_axis_ticks = [f'Y{i+1}' for i in range(z_data.shape[1])]
    y_axis_ticks = y_axis_ticks[::-1]
    
    if x_axis_names is None:
        x_axis_names = [f'X{i+1}' for i in range(z_data.shape[0])]
    
    # Normalize the data to be between -1 and 1
    # z_data = (z_data - np.min(z_data)) / (np.max(z_data) - np.min(z_data)) * 2 - 1
    
    # Create subplots
    fig = make_subplots(cols=z_data.shape[0], rows=1, horizontal_spacing=0.01)
    
    # Create heatmap traces for each layer
    for i in range(z_data.shape[0]):
        trace = go.Heatmap(
            z=np.flip(z_data[i].T, 0),  # Transpose the data to swap axes
            y=y_axis_ticks,
            x=[f'{j}' for j in range(z_data.shape[1])],
            hovertemplate="Layer: %{x}<br>Token: '%{y}'<br>%{z}<extra></extra>",
            colorscale=[[0.0, 'darkblue'], [0.5, 'white'], [0.75, 'red'], [1, 'black']] if normalized else 'viridis',
            showscale=False,
            zmin=-1 if normalized else None,
            zmax=1 if normalized else None
        )
        fig.add_trace(trace, col=i+1, row=1)
        fig.update_xaxes(showticklabels=False)
    
    # Update figure layout
    fig.update_layout(
        width=200 * z_data.shape[0],  # Adjust the width based on the number of layers
        showlegend=False,
        font_family="Courier",
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    
    # Update y-axis properties for the leftmost subplot
    fig.update_yaxes(col=1, row=1)
    
    # Update x-axis properties for each subplot with the corresponding name from x_axis_names
    for i in range(z_data.shape[0]):
        fig.update_xaxes(title_text=x_axis_names[i], col=i+1, row=1, title_font=dict(size=12))
    
    # Hide y-axis labels for subplots except the leftmost one
    for i in range(2, z_data.shape[0]+1):
        fig.update_yaxes(showticklabels=False, col=i, row=1)
    
    fig.update_yaxes(side="right", showticklabels=True, col=z_data.shape[0], row=1)

    # Add thin borders between cells
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Display the plot
    # fig.show()
    return fig

def str_to_feature_pair(s: str) -> List[Tuple[int, int]]:
    # Example input: '8.123, 123.1222 , 32.123'
    out = []
    for x in s.split(','):
        y = x.strip().split('.')
        out.append((int(y[0]), int(y[1])))
    return out


def run(model, saes, prompt, features, func, dec=True, normalized=True):
    tensor = get_feature_movement(model, saes, [prompt], func, features, dec)
    data = tensor[0].cpu()
    return plot_stacked_heatmaps_flipped(data, normalized=normalized, x_axis_names=[f"{layer}.{feature}" for layer, feature in features], y_axis_ticks=model.to_str_tokens(prompt))

if __name__ == "__main__":

    st.title("Track to the Feature")

    if "MODEL_LOADED" not in st.session_state:
        with st.spinner("Model Loading in progress..."):
            saes, _ = get_gpt2_res_jb_saes()
            gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
            st.session_state["SAES"] = saes
            st.session_state["MODEL"] = gpt2_small
            st.session_state["MODEL_LOADED"] = True
    
    prompt = st.text_input("please enter your prompt", value="This movie is amazing! The best I have ever seen", key='prompt')
    feature = st.text_input("please enter a feature", value="8.23510", key='feature')
    label = "Dot product" if st.session_state.get("dot_prod", False) else "Cosine similarity"
    use_dot = st.toggle(label, value=st.session_state.get("dot_prod", False), key="toggle")
    func = get_dot_product if use_dot else get_cosine_similarity
    
    st.button("Calculate!", key="run")
    
    if st.session_state["run"]:
        
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