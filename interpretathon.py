# %%
from cgitb import Hook
from scipy import sparse
import torch as t
from torch import Tensor

from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import einops

from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    device = t.device("cuda") if t.cuda.is_available() else t.device("mps")
    t.set_default_device(device)
    saes, sparsities = get_gpt2_res_jb_saes()
    gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%


# saes is a dictionary with keys like 'blocks.{layer}.hook_resid_pre' 
# saes['blocks.{layer}.hook_resid_pre'].W_dec gets a tensor of shape (features, d_model)
# So to get the vector that corresponds to a particular feature (number n), we'd do
# saes['blocks.{layer}.hook_resid_pre'].W_dec[n]

# saes['blocks.8.hook_resid_pre'].W_dec.shape

# def run(model: HookedTransformer, features: Float[Tensor, "features d_model"], feature_id: int, tokens: Int[Tensor, "str_pos"]) -> Float[Tensor, "layer str_pos"]:
#     assert 0 <= feature_id < features.shape[0]
#     assert model.cfg.d_model == features.shape[1]
#     assert tokens.shape[1] < model.cfg.n_ctx
#     feature = features[feature_id].to(device)
#     out = t.empty((model.cfg.n_layers, tokens.shape[1])).to(device)
#     logits_out, cache = model.run_with_cache(tokens, remove_batch_dim=True)
#     for layer in range(model.cfg.n_layers):
#         out[layer] = einops.einsum(cache["resid_pre", layer], feature, "... s d, ... d -> ... s")
#     return out

def get_all_activations(model: HookedTransformer, cache: ActivationCache) -> Float[Tensor, "layer seq_pos d_model"]:
    return t.stack([cache["resid_pre", i] for i in range(model.cfg.n_layers)])

def get_dot_product(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layer _seq_pos"]]:
    return [einops.einsum(activations, features, "l t d, f d -> f l t") for activations in activations_list]

def get_cosine_similarity(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layer _seq_pos"]]:
    return [t.nn.functional.cosine_similarity(einops.rearrange(activation, "l s d -> 1 d l s"), features[:, :, None, None], dim=1) for activation in activations_list]


# class X:
#     def __init__(self, model: HookedTransformer, sae: dict[str, SparseAutoencoder]) -> None:
#         self.model = model
#         self.sae = sae
    

    
#     def run(self, tokens, func, feature) -> Float[Tensor, "layer seq_pos"]:
#         with t.no_grad():
#             logits_out, cache = self.model.run_with_cache(tokens, remove_batch_dim=True)
#             return func([get_all_activations(self.model, cache)], feature.unsqueeze(0))[0][0].T

#     def get_feature_names(self) -> List[str]:
#         return [
#             f"{layer}.{feature}"
#             for feature in range(self.sae['blocks.0.hook_resid_pre'].W_dec.shape[0])
#             for layer in range(self.model.cfg.n_layers)
#             ]
    
#     def get_feature_from_name(self, name) -> Float[Tensor, "d_model"]:
#         layer, feature = map(int, name[0].split('.'))

#         return self.get_feature(layer, feature)

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


# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stacked_heatmaps(tensor, y_axis_names=None, x_axis_ticks=None):
    z_data = tensor.numpy()  # Convert tensor to numpy array
    
    if x_axis_ticks is None:
        x_axis_ticks = [f'X{i+1}' for i in range(z_data.shape[1])]
    
    if y_axis_names is None:
        y_axis_names = [f'Y{i+1}' for i in range(z_data.shape[0])]
    
    # Create subplots
    fig = make_subplots(rows=z_data.shape[0], cols=1, vertical_spacing=0.05)
    
    # Create heatmap traces for each layer
    for i in range(z_data.shape[0]):
        trace = go.Heatmap(
            z=z_data[i],
            x=x_axis_ticks,
            y=[f'Y{j+1}' for j in range(z_data.shape[1])],
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
            colorscale='Viridis',
            showscale=False,
        )
        fig.add_trace(trace, row=i+1, col=1)
    
    # Update figure layout
    fig.update_layout(
        title='Stacked Interactive Heatmaps',
        height=200 * z_data.shape[0],  # Adjust the height based on the number of layers
        showlegend=False,
    )
    
    # Update x-axis properties for the bottom subplot
    fig.update_xaxes(title_text='X Axis', row=z_data.shape[0], col=1)
    
    # Update y-axis properties for each subplot with the corresponding name from y_axis_names
    for i in range(z_data.shape[0]):
        fig.update_yaxes(title_text=y_axis_names[i], row=i+1, col=1)
    
    # Hide x-axis labels for subplots except the bottom one
    for i in range(1, z_data.shape[0]):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    
    # Display the plot
    fig.show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stacked_heatmaps_flipped(tensor, x_axis_names=None, y_axis_ticks=None):
    z_data = tensor.numpy()  # Convert tensor to numpy array
    
    if y_axis_ticks is None:
        y_axis_ticks = [f'Y{i+1}' for i in range(z_data.shape[1])]
    
    if x_axis_names is None:
        x_axis_names = [f'X{i+1}' for i in range(z_data.shape[0])]
    
    # Create subplots
    fig = make_subplots(cols=z_data.shape[0], rows=1, horizontal_spacing=0.05)
    
    # Create heatmap traces for each layer
    for i in range(z_data.shape[0]):
        trace = go.Heatmap(
            z=z_data[i].T,  # Transpose the data to swap axes
            y=y_axis_ticks,
            x=[f'X{j+1}' for j in range(z_data.shape[1])],
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>',
            colorscale='Viridis',
            showscale=False,
        )
        fig.add_trace(trace, col=i+1, row=1)
        fig.update_xaxes(showticklabels=False)
    
    # Update figure layout
    fig.update_layout(
        width=200 * z_data.shape[0],  # Adjust the width based on the number of layers
        showlegend=False,
    )
    
    # Update y-axis properties for the leftmost subplot
    fig.update_yaxes(title_text='Y Axis', col=1, row=1)
    
    # Update x-axis properties for each subplot with the corresponding name from x_axis_names
    for i in range(z_data.shape[0]):
        fig.update_xaxes(title_text=x_axis_names[i], col=i+1, row=1)
    
    # Hide y-axis labels for subplots except the leftmost one
    for i in range(2, z_data.shape[0]+1):
        fig.update_yaxes(showticklabels=False, col=i, row=1)
    
    # Display the plot
    fig.show()

def visualise(prompt, features):
    tensor = get_feature_movement(gpt2_small, saes, [prompt], get_cosine_similarity, features)
    data = tensor[0].cpu()
    plot_stacked_heatmaps(data, [str(feature) for feature in features], gpt2_small.to_str_tokens(prompt))

def visualise_flipped(prompt, features):
    tensor = get_feature_movement(gpt2_small, saes, [prompt], get_cosine_similarity, features)
    data = tensor[0].cpu()
    plot_stacked_heatmaps_flipped(data, x_axis_names=[str(feature) for feature in features], y_axis_ticks=gpt2_small.to_str_tokens(prompt))

# %%
