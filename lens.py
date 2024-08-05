# %%
import sys, os
import torch as t
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from my_utils import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
p = 113
cfg = HookedTransformerConfig(
    n_layers = 1,
    d_vocab = p+1,
    d_model = 128,
    d_mlp = 4 * 128,
    n_heads = 4,
    d_head = 128 // 4,
    n_ctx = 3,
    act_fn = "relu",
    normalization_type = None,
    device = device
)

hooked_model = HookedTransformer(cfg)
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
# %%
keys = state_dict.keys()
for key in keys:
    print(key, (state_dict[key].shape))

# %%
hooked_model = load_in_state_dict(hooked_model, state_dict)
full_run_data = checkpoint['run_data']
train_loss = full_run_data['train_loss']
val_loss = full_run_data['val_loss']

epochs = list(range(0, len(train_loss)*100, 100))

# Create the plot
fig = go.Figure()

# Add traces for train loss and validation loss
fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Train Loss'))
fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))

# Update layout
fig.update_layout(
    title='Grokking Training Curve',
    xaxis_title='Epoch',
    yaxis_title='Loss',
    template='plotly_white',
    #yaxis_type="log"
)

# Show the plot
fig.show()
"""
lines(
    lines_list=[
        full_run_data['train_loss'][::10], 
        full_run_data['val_loss']
    ], 
    labels=['train loss', 'test loss'], 
    title='Grokking Training Curve', 
    x=np.arange(5000)*10,
    xaxis='Epoch',
    yaxis='Loss',
    log_y=True
)
"""
# %%
W_O = hooked_model.W_O[0]
W_K = hooked_model.W_K[0]
W_Q = hooked_model.W_Q[0]
W_V = hooked_model.W_V[0]
W_in = hooked_model.W_in[0]
W_out = hooked_model.W_out[0]
W_pos = hooked_model.W_pos
W_E = hooked_model.W_E[:-1]
final_pos_resid_initial = hooked_model.W_E[-1] + W_pos[2]
W_U = hooked_model.W_U[:, :-1]

print('W_O  ', tuple(W_O.shape))
print('W_K  ', tuple(W_K.shape))
print('W_Q  ', tuple(W_Q.shape))
print('W_V  ', tuple(W_V.shape))
print('W_in ', tuple(W_in.shape))
print('W_out', tuple(W_out.shape))
print('W_pos', tuple(W_pos.shape))
print('W_E  ', tuple(W_E.shape))
print('W_U  ', tuple(W_U.shape))

# %%
def fn(a, b):
    return (a + b) % p

all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([fn(i, j) for i, j, _ in all_data]).to(device)
original_logits, cache = hooked_model.run_with_cache(all_data)
# Final position only, also remove the logits for `=`
original_logits = original_logits[:, -1, :-1]
original_loss = cross_entropy_high_precision(original_logits, labels)
print(f"Original loss: {original_loss.item()}")
# %%
attn_mat = cache['pattern', 0][:, :, 2]
neuron_acts_post = cache['post', 0][:, -1]
neuron_acts_pre = cache['pre', 0][:, -1]

W_logit = W_out @ W_U

W_OV = W_V @ W_O
W_neur = W_E @ W_OV @ W_in

W_QK = W_Q @ W_K.transpose(-1, -2)
W_attn = final_pos_resid_initial @ W_QK @ W_E.T / (cfg.d_head ** 0.5)

attn_mat = attn_mat[:, :, :2]
# Note, we ignore attn from 2 -> 2

attn_mat_sq = einops.rearrange(attn_mat, "(x y) head seq -> x y head seq", x=p)
# We rearranged attn_mat, so the first two dims represent (x, y) in modular arithmetic equation

inputs_heatmap(
    attn_mat_sq[..., 0],
    title=f'Attention score for heads at position 0',
    animation_frame=2,
    animation_name='head'
)
# %%
neuron_acts_post_sq = einops.rearrange(neuron_acts_post, "(x y) d_mlp -> x y d_mlp", x=p)
neuron_acts_pre_sq = einops.rearrange(neuron_acts_pre, "(x y) d_mlp -> x y d_mlp", x=p)
# We rearranged activations, so the first two dims represent (x, y) in modular arithmetic equation

top_k = 3
inputs_heatmap(
    neuron_acts_post_sq[..., :top_k],
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
# %%
attn_mat_fourier_basis = fft2d(attn_mat_sq)

# Plot results
imshow_fourier(
    attn_mat_fourier_basis[..., 0],
    title=f'Attention score for heads at position 0, in Fourier basis',
    animation_frame=2,
    animation_name='head'
)
# %%
neuron_acts_post_fourier_basis = fft2d(neuron_acts_post_sq)

top_k = 3
imshow_fourier(
    neuron_acts_post_fourier_basis[..., :top_k],
    title=f'Activations for first {top_k} neurons',
    animation_frame=2,
    animation_name='Neuron'
)
# %%
lines(
    fft1d(W_attn),
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis = 'Contribution to attn score',
    title=f'Contribution to attn score (pre-softmax) for each head, in Fourier Basis',
    hover=fourier_basis_names
)
# %%
line(
    (fourier_basis @ W_E).pow(2).sum(1),
    hover=fourier_basis_names,
    title='Norm of embedding of each Fourier Component',
    xaxis='Fourier Component',
    yaxis='Norm'
)
# %%
