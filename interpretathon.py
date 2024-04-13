# %%
from scipy import sparse
import torch as t
from torch import Tensor
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union
import einops

from huggingface_hub import hf_hub_download
device = t.device("cuda") if t.cuda.is_available() else t.device("mps")
t.set_default_device(device)
saes, sparsities = get_gpt2_res_jb_saes()
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")

# %%


# saes is a dictionary with keys like 'blocks.{layer}.hook_resid_pre' 
# saes['blocks.{layer}.hook_resid_pre'].W_dec gets a tensor of shape (features, d_model)
# So to get the vector that corresponds to a particular feature (number n), we'd do
# saes['blocks.{layer}.hook_resid_pre'].W_dec[n]

saes['blocks.8.hook_resid_pre'].W_dec.shape

def run(model: HookedTransformer, features: Float[Tensor, "features d_model"], feature_id: int, tokens: Int[Tensor, "str_pos"]) -> Float[Tensor, "layer str_pos"]:
    assert 0 <= feature_id < features.shape[0]
    assert model.cfg.d_model == features.shape[1]
    assert tokens.shape[1] < model.cfg.n_ctx
    feature = features[feature_id].to(device)
    out = t.empty((model.cfg.n_layers, tokens.shape[1])).to(device)
    logits_out, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    for layer in range(model.cfg.n_layers):
        out[layer] = einops.einsum(cache["resid_pre", layer], feature, "... s d, ... d -> ... s")
    return out

def get_dot_product(activations_list: List[Float[Tensor, "_tokens d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features _tokens"]]:
    return [einops.einsum(activations, features, "... t d, ... f d -> ... f t") for activations in activations_list]

def get_cosine_similarity(activations_list: List[Float[Tensor, "_tokens d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features _tokens"]]:
    pass

def get_feature(all_features: Float[Tensor, "features d_model"], layers_features: ) -> Float[Tensor, "d_model"]:
    return all_features[layers_features]

def generate_feature_visualisation(model, features):
    pass


# %%
