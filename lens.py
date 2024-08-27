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
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Union, Optional
from fancy_einsum import einsum
import einops
from tqdm import tqdm
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from my_utils import *
from model import GPTConfig, GPT

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
out_dir = 'my_runs' # ignored if init_from is not 'resume'
title = "+113_0_2e-4-fixed.pt"
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed_offset = 0
seed = 1337 + seed_offset
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
# %%
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
    ckpt_path = os.path.join(out_dir, title)
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

# %%
hooked_model = modified_load_in_state_dict(hooked_model, state_dict)

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
W_ln = state_dict['transformer.ln_f.weight']

print('W_O  ', tuple(W_O.shape))
print('W_K  ', tuple(W_K.shape))
print('W_Q  ', tuple(W_Q.shape))
print('W_V  ', tuple(W_V.shape))
print('W_in ', tuple(W_in.shape))
print('W_out', tuple(W_out.shape))
print('W_pos', tuple(W_pos.shape))
print('W_E  ', tuple(W_E.shape))
print('W_U  ', tuple(W_U.shape))
print('W_ln ', tuple(W_ln.shape))
# %%
def fn(a, b):
    return (a + b) % p
def subfn(a, b):
    return (a - b) % p
all_data = t.tensor([(i, j, p) for i in range(p) for j in range(p)]).to(device)
sub_data = t.tensor([(i, j, p+1) for i in range(p) for j in range(p)]).to(device)
labels = t.tensor([fn(i, j) for i, j, _ in all_data]).to(device)
sub_labels = t.tensor([subfn(i, j) for i, j, _ in sub_data]).to(device)

#all_data = t.cat((all_data, sub_data)).to(device)
#labels = t.cat((labels, sub_labels)).to(device)

original_logits, cache = hooked_model.run_with_cache(all_data)
# Final position only, also remove the logits for `=`
original_logits = original_logits[:, -1, :-1]

original_loss = cross_entropy_high_precision(original_logits, labels)
with t.no_grad():
    logits, original_loss, accuracy = model(all_data, labels)

print(f"Original loss: {original_loss.item()}")
# %%
attn_mat = cache['pattern', 0][:, :, 2]
neuron_acts_post = cache['post', 0][:, -1]
neuron_acts_pre = cache['pre', 0][:, -1]

W_logit = (W_out * W_ln) @ W_U

W_OV = W_V @ W_O
W_neur = W_E @ W_OV @ W_in

W_QK = W_Q @ W_K.transpose(-1, -2)
W_attn = final_pos_resid_initial @ W_QK @ W_E.T / (cfg.d_head ** 0.5)

average_attn = attn_mat.mean(dim=0)
seq_len = 3
num_heads = 4  # assuming 4 heads

# Initialize a matrix to store the results
average_attn_matrix = torch.zeros(num_heads, seq_len)

# Populate the average_attn_matrix
for head in range(num_heads):
    for pos in range(seq_len):
        # Average attention score for head at each position
        average_attn_matrix[head, pos] = average_attn[head, pos]
scaled = average_attn_matrix * 1000.0
scaled = t.round(scaled)
average_attn_matrix = scaled / 1000.0
inputs_heatmap(
    average_attn_matrix,
    title=f'Attention score for heads at each position',
    xaxis=f'Position',
    yaxis=f'Head',
    text_auto=True
)
# %%
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

top_k = 3
animate_multi_lines(
    W_neur[..., :top_k],
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Input token', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attention)'
)

# %%
lines(
    W_attn,
    labels = [f'head {hi}' for hi in range(4)],
    xaxis='Input token',
    yaxis='Contribution to attn score',
    title=f'Contribution to attention score (pre-softmax) for each head'
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
def prepare_lines_list(run_data, fourier_basis):
    lines_list = []
    for step in range(len(run_data['embedding'])):
        W_E = run_data['embedding'][step][:-1]
        norm_of_embedding = (fourier_basis @ W_E).pow(2).sum(1)
        lines_list.append(norm_of_embedding)
    return lines_list

lines_list = prepare_lines_list(full_run_data, fourier_basis)
animate_lines(
    lines_list,
    snapshot_index=np.arange(len(lines_list)),
    snapshot='Time Step',
    hover=fourier_basis_names,
    xaxis='Fourier Component',
    yaxis='Norm',
    title='Evolution of Embedding Norm Over Time'
)
# %%
W_neur_fourier = fft1d_given_dim(W_neur, dim=1)

top_k = 5
animate_multi_lines(
    W_neur_fourier[..., :top_k],
    y_index = [f'head {hi}' for hi in range(4)],
    labels = {'x':'Fourier component', 'value':'Contribution to neuron'},
    snapshot='Neuron',
    hover=fourier_basis_names,
    title=f'Contribution to first {top_k} neurons via OV-circuit of heads (not weighted by attn), in Fourier basis'
)

# %%
neuron_acts_centered = neuron_acts_post_sq - neuron_acts_post_sq.mean((0, 1), keepdim=True)

# Take 2D Fourier transform
neuron_acts_centered_fourier = fft2d(neuron_acts_centered)


imshow_fourier(
    neuron_acts_centered_fourier.pow(2).mean(-1),
    title=f"Norms of 2D Fourier components of centered neuron activations",
)
# %%
neuron_freqs, neuron_frac_explained = find_neuron_freqs(neuron_acts_centered_fourier)
key_freqs, neuron_freq_counts = t.unique(neuron_freqs, return_counts=True)

print(key_freqs.tolist())

# %%
fraction_of_activations_positive_at_posn2 = (cache['pre', 0][:, -1] > 0).float().mean(0)

scatter(
    x=neuron_freqs,
    y=neuron_frac_explained,
    xaxis="Neuron frequency",
    yaxis="Frac explained",
    colorbar_title="Frac positive",
    title="Fraction of neuron activations explained by key freq",
    color=utils.to_numpy(fraction_of_activations_positive_at_posn2)
)
# %%
# To represent that they are in a special sixth cluster, we set the frequency of these neurons to -1
neuron_freqs[neuron_frac_explained < 0.85] = -1.
key_freqs_plus = t.concatenate([key_freqs, -key_freqs.new_ones((1,))])

for i, k in enumerate(key_freqs_plus):
    print(f'Cluster {i}: freq k={k}, {(neuron_freqs==k).sum()} neurons')
# %%
print(neuron_freqs)
# %%
logits_in_freqs = []

for freq in key_freqs:
    filtered_neuron_acts = neuron_acts_post[:, neuron_freqs==freq]
    filtered_neuron_acts_in_freq = project_onto_frequency(filtered_neuron_acts, freq)
    logits_in_freq = filtered_neuron_acts_in_freq @ W_logit[neuron_freqs==freq]
    logits_in_freqs.append(logits_in_freq)

print('Loss with neuron activations ONLY in key freq (inclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Loss with neuron activations ONLY in key freq (exclusing always firing cluster)\n{:.6e}\n'.format( 
    test_logits(
        sum(logits_in_freqs[:-1]), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Loss with zeroed logits\n{:.6e}\n'.format( 
    test_logits(
        t.zeros(12769,113, device='cuda'), 
        bias_correction=True, 
        original_logits=original_logits
    )
))
print('Original loss\n{:.6e}'.format(original_loss))

# %%
print('Loss with neuron activations excluding none:     {:.9f}'.format(original_loss.item()))
for c, freq in enumerate(key_freqs_plus):
    print('Loss with neuron activations excluding freq={}:  {:.9f}'.format(
        freq, 
        test_logits(
            sum(logits_in_freqs) - logits_in_freqs[c], 
            bias_correction=True, 
            original_logits=original_logits
        )
    ))
# %%
US = W_logit @ fourier_basis.T

imshow_div(
    US,
    x=fourier_basis_names,
    yaxis='Neuron index',
    title='W_logit in the Fourier Basis',
    height=800,
    width=600
)
# %%
imshow_fourier(
    einops.reduce(neuron_acts_centered_fourier.pow(2), 'y x neuron -> y x', 'mean'), 
    title='Norm of Fourier Components of Neuron Acts'
)

# Rearrange logits, so the first two dims represent (x, y) in modular arithmetic equation
original_logits_sq = einops.rearrange(original_logits, "(x y) z -> x y z", x=p)
original_logits_fourier = fft2d(original_logits_sq)

imshow_fourier(
    einops.reduce(original_logits_fourier.pow(2), 'y x z -> y x', 'mean'), 
    title='Norm of Fourier Components of Logits'
)
# %%
