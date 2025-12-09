"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import pickle
import json
import inspect
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
import optimizers

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer parameters
learning_rate = 1e-3 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
l2_lambda = 0.0
optim_mode = 'adamw' # options: adamw (decoupled weight decay), adam_l2 (coupled L2)
l2_target = 'all' # options: all, weights_only, no_embeddings
beta1 = 0.9
beta2 = 0.999
eps = 1e-6
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
lr_schedule = 'cosine' # options: constant, step, cosine
step_lr_milestones = '30,60,80' # comma-separated milestones for step schedule
eta_min = 1e-4 # min lr for cosine variants
warm_restarts = False # only valid with cosine
T0 = 1000 # initial period for warm restarts
T_mult = 2 # multiplicative factor for next restart cycle
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
run_name = '' # optional name; if empty a name is auto-created from optimizer/schedule hyperparameters
# -----------------------------------------------------------------------------
# flags for the algorithmic extensions
attention_variant = 'mha' # options: mha (multi-head attention implemented in nanoGPT), gqa (grouped-query attention)
n_headgroup = 1 # number of heads per group if using grouped-query attention
optimizer_variant = 'adamw' # options: adamw (optimizer used in nanoGPT), adam (implemented in optimizers.py), adafactor, adamsn
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# parse CLI strings and derive run naming
base_out_dir = out_dir
if isinstance(step_lr_milestones, str):
    step_lr_milestones = [int(m) for m in step_lr_milestones.split(',') if m.strip()]
if warm_restarts and lr_schedule != 'cosine':
    raise ValueError("warm_restarts=True requires lr_schedule='cosine'")
if run_name == '':
    schedule_tag = lr_schedule + ('_wr' if warm_restarts else '')
    decay_tag = f"wd{weight_decay}" if optim_mode == 'adamw' else f"l2{l2_lambda}_{l2_target}"
    run_name = f"{optim_mode}_{schedule_tag}_lr{learning_rate}_{decay_tag}"
out_dir = os.path.join(base_out_dir, run_name)
config.update({
    'run_name': run_name,
    'out_dir': out_dir,
    'step_lr_milestones': step_lr_milestones,
    'l2_target': l2_target,
})

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
log_file = os.path.join(out_dir, 'logs.txt')

def append_log(record):
    if not master_process:
        return
    with open(log_file, 'a') as f:
        f.write(json.dumps(record) + "\n")

# learning rate scheduler builder
def create_lr_scheduler(optimizer):
    schedulers = []
    if warmup_iters > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(1.0 / max(1, warmup_iters), 1e-5),
            total_iters=warmup_iters,
        )
        schedulers.append(warmup)

    main_sched = None
    if decay_lr and lr_schedule != 'constant':
        if lr_schedule == 'step':
            main_sched = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_lr_milestones or [], gamma=0.1
            )
        elif lr_schedule == 'cosine':
            if warm_restarts:
                main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T0, T_mult=T_mult, eta_min=eta_min
                )
            else:
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=lr_decay_iters, eta_min=eta_min
                )
        else:
            raise ValueError(f"Unknown lr_schedule {lr_schedule}")

    if main_sched and schedulers:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[schedulers[0], main_sched], milestones=[warmup_iters]
        )
    if main_sched:
        return main_sched
    if schedulers:
        return schedulers[0]
    return None


def parameter_l2_norm(parameters):
    total = None
    for p in parameters:
        if not p.requires_grad:
            continue
        term = (p.detach().float() ** 2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def grad_l2_norm(parameters):
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.sqrt(sum((g ** 2).sum() for g in grads))


def l2_param_iterator(raw_model):
    """Return parameters to include in manual L2 depending on l2_target."""
    params = []
    for name, p in raw_model.named_parameters():
        if not p.requires_grad:
            continue
        if l2_target == 'weights_only' and p.dim() < 2:
            continue
        if l2_target == 'no_embeddings' and ('wte' in name or 'wpe' in name):
            continue
        params.append(p)
    return params


torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_headgroup=n_headgroup, n_embd=n_embd, block_size=block_size,
                    bias=False, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_headgroup', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
print(f"optimizer_variant = {optimizer_variant}, optim_mode = {optim_mode}, lr_schedule = {lr_schedule}, warm_restarts = {warm_restarts}")

optimizer = None
if optim_mode == 'adam_l2':
    fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps, **extra_args
    )
elif optim_mode == 'adamw':
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# legacy fallback to keep previous optimizer_variant options working
if optimizer is None:
    if optimizer_variant == 'adamw':
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    elif optimizer_variant == 'adam':
        optimizer = optimizers.Adam(model.parameters(), learning_rate, betas=(beta1, beta2))
    elif optimizer_variant == 'adafactor':
        linear_modules = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
        regular_params = [p for p in model.parameters() if id(p) not in [id(p) for p in linear_modules]]
        param_groups = [
            {'params': regular_params, 'param_type': 'regular'},
            {'params': linear_modules, 'param_type': 'linear'}
        ]
        optimizer = optimizers.Adafactor(param_groups, learning_rate, betas=(beta1, beta2))
    elif optimizer_variant == 'adamsn':
        linear_modules = [module.weight for module in model.modules() if isinstance(module, nn.Linear)]
        regular_params = [p for p in model.parameters() if id(p) not in [id(p) for p in linear_modules]]
        param_groups = [
            {'params': regular_params, 'param_type': 'regular'},
            {'params': linear_modules, 'param_type': 'linear'}
        ]
        optimizer = optimizers.AdamSN(param_groups, learning_rate, betas=(beta1, beta2))

if optimizer is None:
    raise ValueError(f"Unknown optimizer/optim_mode combination (optim_mode={optim_mode}, optimizer_variant={optimizer_variant})")


if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

lr_scheduler = create_lr_scheduler(optimizer)
if init_from == 'resume' and checkpoint is not None and 'lr_scheduler' in checkpoint and lr_scheduler is not None:
    try:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    except Exception as e:
        print(f"warning: could not load lr_scheduler state: {e}")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate scheduler builder
def create_lr_scheduler(optimizer):
    schedulers = []
    if warmup_iters > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=max(1.0 / max(1, warmup_iters), 1e-5),
            total_iters=warmup_iters,
        )
        schedulers.append(warmup)

    main_sched = None
    if decay_lr and lr_schedule != 'constant':
        if lr_schedule == 'step':
            main_sched = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_lr_milestones or [], gamma=0.1
            )
        elif lr_schedule == 'cosine':
            if warm_restarts:
                main_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T0, T_mult=T_mult, eta_min=eta_min
                )
            else:
                main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=lr_decay_iters, eta_min=eta_min
                )
        else:
            raise ValueError(f"Unknown lr_schedule {lr_schedule}")

    if main_sched and schedulers:
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[schedulers[0], main_sched], milestones=[warmup_iters]
        )
    if main_sched:
        return main_sched
    if schedulers:
        return schedulers[0]
    return None


def parameter_l2_norm(parameters):
    total = None
    for p in parameters:
        if not p.requires_grad:
            continue
        term = (p.detach().float() ** 2).sum()
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def grad_l2_norm(parameters):
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.sqrt(sum((g ** 2).sum() for g in grads))

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    lr = optimizer.param_groups[0]['lr']

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        append_log({
            "iter": iter_num,
            "split": "val",
            "train_loss": float(losses['train']),
            "val_loss": float(losses['val']),
            "lr": lr,
            "optim_mode": optim_mode,
            "lr_schedule": lr_schedule,
            "warm_restarts": warm_restarts,
            "weight_decay": weight_decay,
            "l2_lambda": l2_lambda,
            "l2_target": l2_target,
        })
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    l2_term_value = 0.0
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            l2_term = 0.0
            if optim_mode == 'adam_l2' and l2_lambda > 0.0:
                params_for_l2 = l2_param_iterator(raw_model)
                l2_term = 0.5 * l2_lambda * sum((p * p).sum() for p in params_for_l2)
                loss = loss + l2_term
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            if isinstance(l2_term, torch.Tensor):
                l2_term_value = l2_term.item()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # unscale and optionally clip gradients, track grad norm for logging
    scaler.unscale_(optimizer)
    grad_norm_value = grad_l2_norm(raw_model.parameters()).item()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    if lr_scheduler is not None:
        lr_scheduler.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        lr = optimizer.param_groups[0]['lr']
        param_norm_value = parameter_l2_norm(raw_model.parameters()).item()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {lr:.6f}, ||p|| {param_norm_value:.4f}, ||g|| {grad_norm_value:.4f}")
        append_log({
            "iter": iter_num,
            "split": "train",
            "loss": lossf,
            "lr": lr,
            "param_norm": param_norm_value,
            "grad_norm": grad_norm_value,
            "l2_term": l2_term_value,
            "optim_mode": optim_mode,
            "lr_schedule": lr_schedule,
            "warm_restarts": warm_restarts,
            "weight_decay": weight_decay,
            "l2_lambda": l2_lambda,
            "l2_target": l2_target,
            "mfu": running_mfu,
            "grad_clip": grad_clip,
        })
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
