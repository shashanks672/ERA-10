# -*- coding: utf-8 -*-
"""
ERA V5 Session 10 — Gradients, Accumulation, MFU, Precision
Small model + a real training loop that reports the truth about itself.
"""

import math
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class ToyConfig:
    vocab_size = 65
    seq_len = 32
    d_model = 64
    n_layer = 2
    n_head = 4
    d_ff = 256


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.w_q(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.w_o(out)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)
        return hidden, logits


def causal_loss(logits, tokens):
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )


def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 0. Device + model
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ToyConfig()
model = ToyGPT(config).to(device)
n_params = count_params(model)
print("Device:", device)
print(f"Parameters: {n_params:,} ({n_params/1e6:.3f} M)")


# ---------------------------------------------------------------------------
# 1. Print every tensor in one training step
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("1. TENSOR SHAPES IN ONE TRAINING STEP")
print("=" * 72)

B, L, V, D = 4, config.seq_len, config.vocab_size, config.d_model
tokens = torch.randint(1, V, (B, L), device=device)
hidden, logits = model(tokens)
loss = causal_loss(logits, tokens)
loss.backward()

def shp(t):
    return str(list(t.shape))

print(f"tokens          {shp(tokens):20s}  [B, L] batch x context")
print(f"tok_emb.weight  {shp(model.tok_emb.weight):20s}  [V, D] vocab x hidden")
print(f"pos_emb.weight  {shp(model.pos_emb.weight):20s}  [Lmax, D] positions x hidden")
print(f"hidden          {shp(hidden):20s}  [B, L, D] residual stream")
print(f"logits          {shp(logits):20s}  [B, L, V] unnormalized next-token scores")
print(f"loss            {shp(loss):20s}  [] scalar mean CE over B*(L-1) tokens")
w = model.lm_head.weight
print(f"lm_head.weight  {shp(w):20s}  [V, D] output projection")
print(f"lm_head.grad    {shp(w.grad):20s}  [V, D] dL/dW for that matrix")
print(f"one attn W_q    {shp(model.blocks[0].attn.w_q.weight):20s}  [D, D] Q projection")
print("Shift used: logits[:, :-1] predicts tokens[:, 1:]")


# ---------------------------------------------------------------------------
# 2. Verify one gradient by hand (central finite difference)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("2. HAND-CHECK ONE GRADIENT")
print("=" * 72)

# float64 clone: CE on a 0.1M model is insensitive to one weight in fp32
fd_model = ToyGPT(config).to(device=device, dtype=torch.float64)
fd_model.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
tokens_fd = torch.randint(1, V, (2, L), device=device)
param = fd_model.lm_head.weight
i0, j0 = 3, 7

_, logits0 = fd_model(tokens_fd)
loss0 = causal_loss(logits0, tokens_fd)
fd_model.zero_grad(set_to_none=True)
loss0.backward()
analytic = param.grad[i0, j0].item()

eps = 1e-6
with torch.no_grad():
    param[i0, j0] += eps
    _, lp = fd_model(tokens_fd)
    Lp = causal_loss(lp, tokens_fd).item()
    param[i0, j0] -= 2 * eps
    _, lm = fd_model(tokens_fd)
    Lm = causal_loss(lm, tokens_fd).item()
    param[i0, j0] += eps

numeric = (Lp - Lm) / (2 * eps)
rel = abs(numeric - analytic) / (abs(analytic) + 1e-18)

print(f"Parameter probed : lm_head.weight[{i0}, {j0}]  (float64)")
print(f"epsilon          : {eps}")
print(f"L(w+eps)         : {Lp:.12f}")
print(f"L(w-eps)         : {Lm:.12f}")
print(f"numeric dL/dw    : {numeric:.10e}")
print(f"backward() dL/dw : {analytic:.10e}")
print(f"abs diff         : {abs(numeric-analytic):.3e}")
print(f"rel diff         : {rel:.3e}")
ok = rel < 1e-5 or abs(numeric - analytic) < 1e-8
print("PASS — finite difference matches backward()" if ok else "CHECK — they should agree to several decimals")
model.zero_grad(set_to_none=True)


# ---------------------------------------------------------------------------
# 3. Break gradient accumulation on purpose
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("3. BROKEN vs CORRECT GRADIENT ACCUMULATION")
print("=" * 72)
print("Wrong : mean of per-microbatch mean losses  (average of averages)")
print("Right : token-weighted  sum(loss_i * n_i) / sum(n_i)")

torch.manual_seed(0)
# three micro-batches with DIFFERENT lengths
micro_lengths = [8, 16, 32]
n_rounds = 40

# two clones so both stories start from the same weights
m_wrong = ToyGPT(config).to(device)
m_right = ToyGPT(config).to(device)
m_right.load_state_dict(m_wrong.state_dict())
opt_w = torch.optim.AdamW(m_wrong.parameters(), lr=3e-3)
opt_r = torch.optim.AdamW(m_right.parameters(), lr=3e-3)

wrong_curve, right_curve = [], []

for rnd in range(n_rounds):
    micros = [
        torch.randint(1, V, (2, length), device=device) for length in micro_lengths
    ]

    # --- WRONG: average of averages ---
    opt_w.zero_grad(set_to_none=True)
    acc_wrong = 0.0
    for xb in micros:
        _, lg = m_wrong(xb)
        li = causal_loss(lg, xb)
        (li / len(micros)).backward()  # equal vote per microbatch, ignore length
        acc_wrong += li.item()
    opt_w.step()
    wrong_curve.append(acc_wrong / len(micros))

    # --- RIGHT: token-weighted ---
    opt_r.zero_grad(set_to_none=True)
    weighted_sum = 0.0
    token_count = 0
    for xb in micros:
        _, lg = m_right(xb)
        n_tok = xb.size(0) * (xb.size(1) - 1)
        li = causal_loss(lg, xb)  # mean over that microbatch
        # scale so total = sum(n_i * mean_i) / sum(n_i)
        (li * n_tok).backward()
        weighted_sum += li.item() * n_tok
        token_count += n_tok
    # grads are sum(n_i * d mean_i); divide by total tokens
    for p in m_right.parameters():
        if p.grad is not None:
            p.grad.div_(token_count)
    opt_r.step()
    right_curve.append(weighted_sum / token_count)

print(f"Wrong final mean-of-means : {wrong_curve[-1]:.4f}")
print(f"Right final token-weighted: {right_curve[-1]:.4f}")
print(f"Gap at last step          : {abs(wrong_curve[-1]-right_curve[-1]):.4f}")

os.makedirs("/home/workdir/artifacts", exist_ok=True)
plt.figure(figsize=(8, 4.2))
plt.plot(wrong_curve, label="broken: average of averages")
plt.plot(right_curve, label="correct: token-weighted")
plt.xlabel("optimizer step")
plt.ylabel("reported loss")
plt.title("Micro-batches of lengths 8, 16, 32 — equal vote vs token vote")
plt.legend()
plt.tight_layout()
plot_path = "/home/workdir/artifacts/accum_gap.png"
plt.savefig(plot_path, dpi=140)
plt.close()
print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------------------------
# 4. Log grad norm every step; find a step where norm moved before loss
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("4. GRAD NORM vs LOSS")
print("=" * 72)

m = ToyGPT(config).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
steps, losses, norms = [], [], []

for step in range(80):
    xb = torch.randint(1, V, (8, L), device=device)
    opt.zero_grad(set_to_none=True)
    _, lg = m(xb)
    li = causal_loss(lg, xb)
    li.backward()
    gn = global_grad_norm(m)
    opt.step()
    steps.append(step)
    losses.append(li.item())
    norms.append(gn)
    if step % 10 == 0:
        print(f"step {step:3d}  loss={li.item():.4f}  grad_norm={gn:.4f}")

# first step where |Δnorm| is large while |Δloss| is small (norm moved first)
found = None
for t in range(1, len(steps)):
    d_loss = abs(losses[t] - losses[t - 1])
    d_norm = abs(norms[t] - norms[t - 1])
    # relative: norm jump larger than its running scale, loss almost flat
    if d_norm > 0.15 * (sum(norms[:t]) / t) and d_loss < 0.02:
        found = t
        break
if found is None:
    # fallback: largest d_norm / (d_loss + eps)
    best, best_score = 1, -1
    for t in range(1, len(steps)):
        score = abs(norms[t] - norms[t - 1]) / (abs(losses[t] - losses[t - 1]) + 1e-8)
        if score > best_score:
            best, best_score = t, score
    found = best

print(f"\nStep where grad-norm moved before the loss: {found}")
print(f"  loss[{found-1}]={losses[found-1]:.4f} -> loss[{found}]={losses[found]:.4f}   Δ={losses[found]-losses[found-1]:+.4f}")
print(f"  norm[{found-1}]={norms[found-1]:.4f} -> norm[{found}]={norms[found]:.4f}   Δ={norms[found]-norms[found-1]:+.4f}")
print("Reading: the update direction changed size before the scalar loss showed it.")

plt.figure(figsize=(8, 4.2))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(steps, losses, color="C0", label="loss")
ax2.plot(steps, norms, color="C1", label="grad norm")
ax1.axvline(found, color="gray", ls="--", lw=1)
ax1.set_xlabel("step")
ax1.set_ylabel("loss")
ax2.set_ylabel("grad norm")
ax1.set_title(f"Loss vs grad-norm (marker at step {found})")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
plt.tight_layout()
norm_plot = "/home/workdir/artifacts/gradnorm_vs_loss.png"
plt.savefig(norm_plot, dpi=140)
plt.close()
print(f"Plot saved: {norm_plot}")


# ---------------------------------------------------------------------------
# 5. MFU
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. MODEL FLOPS UTILIZATION (MFU)")
print("=" * 72)

# PaLM/Kaplan-style: forward+backward ≈ 6 * N * T  (matmul-dominated)
# T = tokens processed per step
meas_B, meas_L = 8, config.seq_len
T = meas_B * (meas_L - 1)
flops_per_step = 6.0 * n_params * T

m_mfu = ToyGPT(config).to(device)
opt_mfu = torch.optim.SGD(m_mfu.parameters(), lr=1e-3)
# warmup
for _ in range(5):
    xb = torch.randint(1, V, (meas_B, meas_L), device=device)
    _, lg = m_mfu(xb)
    causal_loss(lg, xb).backward()
    opt_mfu.step()
    opt_mfu.zero_grad(set_to_none=True)

if device.type == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
n_meas = 30
for _ in range(n_meas):
    xb = torch.randint(1, V, (meas_B, meas_L), device=device)
    _, lg = m_mfu(xb)
    causal_loss(lg, xb).backward()
    opt_mfu.step()
    opt_mfu.zero_grad(set_to_none=True)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
steps_per_s = n_meas / elapsed
achieved_flops = steps_per_s * flops_per_step

# peak: honest device peak
if device.type == "cuda":
    # T4 ~ 65 TFLOP/s fp32 Tensor? report as 8.1 TFLOP/s fp32 (T4 official fp32 ~8.1)
    peak = 8.1e12
    peak_name = "T4 fp32 ~8.1 TFLOP/s (conservative)"
else:
    # Mac/CPU: do not pretend 40% of an H100
    peak = 0.5e12
    peak_name = "CPU/MPS ballpark 0.5 TFLOP/s (order-of-magnitude, not a datasheet)"

mfu = 100.0 * achieved_flops / peak
print(f"N parameters              : {n_params:,}")
print(f"Tokens / step T           : {T}")
print(f"6 N T FLOPs / step        : {flops_per_step:.3e}")
print(f"Measured steps / s        : {steps_per_s:.2f}")
print(f"Achieved FLOP/s           : {achieved_flops:.3e}")
print(f"Peak used for MFU         : {peak:.3e}   ({peak_name})")
print(f"MFU                       : {mfu:.3f} %")
print("Distance to 40%:")
print("  This is a tiny model, tiny batch, Python eager loop.")
print("  Cost of the gap: host overhead, no fused kernels, no FlashAttention,")
print("  memory-bound tiny GEMMs, and (on CPU) no tensor cores at all.")
print("  40% MFU is an H100/large-matmul regime, not this toy.")


# ---------------------------------------------------------------------------
# 6. 0.1 in fp32 / bf16 / fp8 E4M3
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("6. THE NUMBER 0.1 IN THREE FORMATS")
print("=" * 72)

def bits_fp32(x: float) -> str:
    u = torch.tensor([x], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
    b = f"{u:032b}"
    return b[0], b[1:9], b[9:]

def bits_bf16(x: float) -> str:
    # bf16 is top 16 bits of fp32
    u = torch.tensor([x], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
    b = f"{u:032b}"
    return b[0], b[1:9], b[9:16]

def bits_fp8_e4m3(x: float):
    """
    FP8 E4M3 (OCP): 1 sign, 4 exp (bias 7), 3 mantissa, implicit leading 1 for normals.
    No inf; exponent 15 with nonzero mantissa is NaN.
    We round 0.1 to nearest representable E4M3.
    """
    if x == 0:
        return "0", "0000", "000", 0.0
    sign = 0 if x >= 0 else 1
    ax = abs(x)
    # unbiased exponent
    e = int(math.floor(math.log2(ax)))
    exp_field = e + 7
    # clamp to normal range exp_field 1..14 (0 is subnormal, 15 is NaN)
    if exp_field <= 0:
        # subnormals: significand = ax / 2^(-6) / 2^-3 wait
        # value = 2^(1-bias) * (m/8) = 2^-6 * m/8
        m = int(round(ax / (2 ** -6) * 8))
        m = max(0, min(7, m))
        recon = (2 ** -6) * (m / 8.0)
        return str(sign), "0000", f"{m:03b}", (-recon if sign else recon)
    if exp_field >= 15:
        # max finite E4M3 is 1.111 * 2^(14-7) = 1.875 * 2^7 = 240
        return str(sign), "1110", "111", (-240.0 if sign else 240.0)
    frac = ax / (2 ** e) - 1.0  # in [0,1)
    m = int(round(frac * 8))
    if m == 8:
        m = 0
        exp_field += 1
        e += 1
        if exp_field >= 15:
            return str(sign), "1110", "111", (-240.0 if sign else 240.0)
    recon = (1.0 + m / 8.0) * (2 ** (exp_field - 7))
    return str(sign), f"{exp_field:04b}", f"{m:03b}", (-recon if sign else recon)


s32, e32, m32 = bits_fp32(0.1)
s16, e16, m16 = bits_bf16(0.1)
s8, e8, m8, r8 = bits_fp8_e4m3(0.1)

x32 = torch.tensor(0.1, dtype=torch.float32)
x16 = torch.tensor(0.1, dtype=torch.bfloat16)
print("Target value: 0.1")
print()
print("fp32   (1 + 8 + 23)")
print(f"  bits     : {s32} {e32} {m32}")
print(f"  stored   : {x32.item():.10f}")
print(f"  error    : {abs(x32.item()-0.1):.3e}")
print()
print("bf16   (1 + 8 + 7)  — same exponent field as fp32, short mantissa")
print(f"  bits     : {s16} {e16} {m16}")
print(f"  stored   : {x16.float().item():.10f}")
print(f"  error    : {abs(x16.float().item()-0.1):.3e}")
print()
print("fp8 E4M3 (1 + 4 + 3), bias 7")
print(f"  bits     : {s8} {e8} {m8}")
print(f"  stored   : {r8:.10f}")
print(f"  error    : {abs(r8-0.1):.3e}   ({100*abs(r8-0.1)/0.1:.1f}%)")
print()
print("Which I would train in: bf16.")
print("  fp32 wastes memory/bandwidth; tiny grads survive but the machine is slow.")
print("  fp8 E4M3 cannot even hold 0.1 well (~nearest 0.1094) and needs per-tensor")
print("  scaling so small gradients do not flush to zero. Fine for mature stacks")
print("  (Transformer Engine), not for this assignment loop.")
print("  bf16 keeps the fp32 exponent (range) and halves the width. That is the")
print("  default I would use on a real GPU.")

print("\nDone.")
