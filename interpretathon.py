from scipy import sparse
import torch as t
from torch import Tensor
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
from jaxtyping import Float, Int, Bool
from typing import List, Optional, Callable, Tuple, Dict, Literal, Set, Union

from huggingface_hub import hf_hub_download
saes, sparsities = get_gpt2_res_jb_saes()
device = t.device("cuda") if t.cuda.is_available() else t.device("mps")




# saes is a dictionary with keys like 'blocks.{layer}.hook_resid_pre' 
# saes['blocks.{layer}.hook_resid_pre'].W_dec gets a tensor of shape (features, d_model)
# So to get the vector that corresponds to a particular feature (number n), we'd do
# saes['blocks.{layer}.hook_resid_pre'].W_dec[n]

saes['blocks.8.hook_resid_pre'].W_dec.shape

def run(model: HookedTransformer, features: Float[Tensor, "features d_model"], feature_id: int, prompt: str) -> Float[Tensor, "layer str_pos"]:
    assert 0 <= feature_id < features.shape[0]
    assert model.cfg.d_model == features.shape[1]
    assert len(prompt) < model.cfg.n_ctx

    logits_out, cache = model.run_with_cache(prompt)
    


def generate_feature_visualisation(model, features):
    pass

