# %%
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPT, GPTConfig
from dataclasses import dataclass
# %%
p = 113
model_args = dict(block_size=3, vocab_size=p+1, n_layer=1, n_head=4, n_embd=128, dropout=0.0, bias=False)
"""
@dataclass
class Config:
    block_size: int = 128 // 4
    vocab_size: int = p+1 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 1
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
"""
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

# %%
seed_offset = 0
torch.manual_seed(1337 + seed_offset)
r = 0.3
def fn(a, b):
    return (a + b) % p
all_data = torch.tensor([(i, j, p) for i in range(p) for j in range(p)])
labels = torch.tensor([fn(i, j) for i, j, _ in all_data])


indices = torch.randperm(all_data.size()[0])
all_data = all_data[indices]
labels = labels[indices]
split = math.ceil(p*p*r) 
train_data=all_data[0:split]
train_labels=labels[0:split]
val_data = all_data[split:]
val_labels = labels[split:]

# %%
#more config
out_dir = 'out'
eval_interval = 100
log_interval = 100
#eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'grokking'
wandb_run_name = '+p=113seed0' # 'run' + str(time.time())
# data
#dataset = 'openwebtext'
batch_size = split # if gradient_accumulation_steps > 1, this is the micro-batch size
gradient_accumulation_steps = math.ceil(p*p*r/batch_size) # used to simulate larger batch sizes
block_size = 3
# adamw optimizer
learning_rate = 0.001
max_iters = 10000 # total number of training iterations
weight_decay = 1
beta1 = 0.9
beta2 = 0.98
grad_clip = 0.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = False # whether to decay the learning rate
#warmup_iters = 0 # how many steps to warm up for
#lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
#min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
#backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
#exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
master_process = True

ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# %%
iter_num = 0
best_val_loss = 1e9
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)


# %%
@torch.no_grad()
def get_loss():
    out = {}
    model.eval()
    #train
    index = 0
    losses = torch.zeros(gradient_accumulation_steps)
    accuracies = torch.zeros(gradient_accumulation_steps)
    for k in range(gradient_accumulation_steps):
        X, Y = get_batch('train', index)
        index += 1
        with ctx:
            logits, loss, accuracy = model(X, Y)
        losses[k] = loss.item()
        accuracies[k] = accuracy
    out['train'] = losses.mean()
    out['train_accuracy'] = accuracies.mean()
    #test
    index = 0
    """
    Removed for full batch test loss
    test_steps = math.ceil(p*p*(1-r)/batch_size)
    losses = torch.zeros(test_steps)
    accuracies = torch.zeros(test_steps)
    for k in range(test_steps):
    """
    losses = torch.zeros(1)
    accuracies = torch.zeros(1)
    for k in range(1):
        X, Y = get_batch('val', index)
        index += 1
        with ctx:
            logits, loss, accuracy = model(X, Y)
        losses[k] = loss.item()
        accuracies[k] = accuracy
    out['val'] = losses.mean()
    out['val_accuracy'] = accuracies.mean()
    model.train()
    return out
# %%
def get_batch(split, batch_idx):
    if batch_idx == gradient_accumulation_steps-1:
        if split == 'train':
            x = train_data[batch_idx*batch_size:]
            y = train_labels[batch_idx*batch_size:]
        else:
            x = val_data[batch_idx*batch_size:]
            y = val_labels[batch_idx*batch_size:]
    else:
        if split == 'train':
            x = train_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = train_labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            x = val_data[batch_idx*batch_size:(batch_idx+1)*batch_size]
            y = val_labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
    x = x.contiguous()
    y = y.contiguous()
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# %%
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train', 0) # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0
run_data = {
    'train_loss': [],
    'val_loss': [],
    'train_accuracy': [],
    'val_accuracy': [],
}
while True:
    batch_index = 0
    # determine and set the learning rate for this iteration
    #lr = get_lr(iter_num) if decay_lr else learning_rate
    lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = get_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train accuracy {losses['train_accuracy']:.4f}, val accuracy {losses['val_accuracy']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "train/accuracy": losses['train_accuracy'],
                "val/accuracy": losses["val_accuracy"],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            run_data["train_loss"].append(losses['train'].item())
            run_data["val_loss"].append(losses['val'].item())
            run_data["train_accuracy"].append(losses['train_accuracy'].item())
            run_data["val_accuracy"].append(losses['val_accuracy'].item())
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'run_data': run_data,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, accuracy = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        X, Y = get_batch('train', batch_index)
        batch_index += 1
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == -1 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break