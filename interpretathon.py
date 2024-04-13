from scipy import sparse
import torch 
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader

from huggingface_hub import hf_hub_download
saes, sparsities = get_gpt2_res_jb_saes()

# saes is a dictionary with keys like 'blocks.{layer}.hook_resid_pre' 
# saes['blocks.{layer}.hook_resid_pre'].W_dec gets a tensor of shape (features, d_model)
# So to get the vector that corresponds to a particular feature (number n), we'd do
# saes['blocks.{layer}.hook_resid_pre'].W_dec[n]

saes['blocks.8.hook_resid_pre'].W_dec.shape



def generate_feature_visualisation(model, features):
    pass