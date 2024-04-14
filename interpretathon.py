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

def get_dot_product(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layers _seq_pos"]]:
    return [einops.einsum(activations, features, "l t d, f d -> f l t") for activations in activations_list]

def get_cosine_similarity(activations_list: List[Float[Tensor, "layer _seq_pos d_model"]], features: Float[Tensor, "features d_model"]) -> List[Float[Tensor, "features layer _seq_pos"]]:
    return [t.nn.functional.cosine_similarity(einops.rearrange(activation, "l s d -> 1 d l s"), features[:, :, None, None], dim=1) for activation in activations_list]


class X:
    def __init__(self, model: HookedTransformer, sae: dict[str, SparseAutoencoder]) -> None:
        self.model = model
        self.sae = sae
    
    def get_feature(self, layer: int, feature: int) -> Float[Tensor, "d_model"]:
        return self.sae[f"blocks.{layer}.hook_resid_pre"].W_dec[feature]
    
    def run(self, tokens, func, feature) -> Float[Tensor, "layer seq_pos"]:
        with t.no_grad():
            logits_out, cache = self.model.run_with_cache(tokens, remove_batch_dim=True)
            return func([get_all_activations(self.model, cache)], feature.unsqueeze(0))[0][0].T

    def get_feature_names(self) -> List[str]:
        return [
            f"{layer}.{feature}"
            for feature in range(self.sae['blocks.0.hook_resid_pre'].W_dec.shape[0])
            for layer in range(self.model.cfg.n_layers)
            ]
    
    def get_feature_from_name(self, name) -> Float[Tensor, "d_model"]:
        layer, feature = map(int, name[0].split('.'))

        return self.get_feature(layer, feature)



# %%
