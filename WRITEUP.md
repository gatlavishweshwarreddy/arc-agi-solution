# ARC-AGI-2 & AGI-3 Solution Writeup
**Author:** Vishweshwar Reddy Gatla
**GitHub:** https://github.com/gatlavishweshwarreddy/arc-agi-solution
**Date:** November 2026

---

## Abstract
We present two solutions for ARC Prize 2026. For AGI-3, we achieve 93.07% using Qwen3.8-Flash-Next with speculative decoding on Tufa Labs' Duck harness. For AGI-2, we implement a high-throughput Test-Time Training (TTT) pipeline using RSLoRA on 4x Nvidia L4 GPUs with dihedral augmentation and GPU-optimized beam search decoding.

---

## 1. AGI-3 Solution (93.07%)

### 1.1 Base Infrastructure
Built on Tufa Labs' Duck harness (Jeroen Cottaar). The attached source bundle is an unchanged copy of Tufa Labs' original. All modifications applied via patch at notebook start based on commit 7652836.

### 1.2 Model
- Qwen3.8-Flash-Next (switched from original Duck model)
- Intel AutoRound 4-bit quantization
- Albucino quantized MTP draft checkpoint for speculative decoding
- Inference via Pennyroyal (John Pezzulli SGLang fork v2.5.3, commit d00d88e)
- Patches from Mamy Ratsimbazafy and Gabriel Olympie

### 1.3 Why Speculative Decoding?
Standard autoregressive decoding generates one token per forward pass. Speculative decoding uses a small draft model to propose multiple tokens verified in parallel by the main model. For ARC grids (long repetitive digit sequences) this yields 3-5x throughput improvement enabling deeper beam search within time budget.

### 1.4 Why 93.07%?
- Qwen3.8-Flash-Next superior reasoning on abstract grid tasks
- Speculative decoding = more beam candidates per second
- Duck harness robust puzzle handling and augmentation scoring
- RTX PRO 6000 GPU sufficient VRAM for full precision inference

---

## 2. AGI-2 Solution

### 2.1 Model
- Qwen 4B loaded via Unsloth (2x faster inference)
- BFloat16 precision
- 4x Nvidia L4 GPUs (22GB each = 88GB total)
- Custom ARC tokenizer: digits 0-9, newline, EOS
- Max sequence length: 8192 tokens
### 2.2 Test-Time Training (TTT)
For each puzzle at inference time:Step 1: Generate 32 augmented views of training pairs
Step 2: Fine-tune RSLoRA adapters for 1 epoch
Step 3: Run DFS beam search with fine-tuned model
Step 4: Score candidates across 4 augmented views
Step 5: Select best candidate via score_kgmon
Step 6: Reset LoRA adapters for next puzzle
### 2.3 RSLoRA Configurationr = 256
lora_alpha = 32
use_rslora = True
target_modules = [q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj,
embed_tokens, lm_head]
optimizer = AdamW
learning_rate = 1e-4
warmup_ratio = 0.2
scheduler = cosine
num_train_epochs = 1
### 2.4 Data Augmentation
16 dihedral symmetry views per puzzle:
- 4 rotations (0, 90, 180, 270 degrees)
- Transpose of each rotation
- Random color permutations (n=32)
- Example order shuffling
Total: 32 augmented training sequences per puzzle

### 2.5 GPU Optimizations

#### turbo_log_softmax
Instead of transferring full vocabulary logits (150k floats) to CPU:
- Compute log_softmax on GPU
- Extract only 12 ARC token log-probs
- Transfer 12 values to CPU
- Result: 12,500x less PCIe data transfer

#### calc_scores (Teacher-Forced Scoring)
- KV cache disabled: zero redundant memory writes
- Gather target token log-probs directly on GPU
- Batch scoring across multiple candidates simultaneously

#### Multi-GPU Parallelism
- 4 independent worker processes (one per GPU)
- Sequential model loading (rank>0 waits for rank-1)
- Prevents Unsloth patching conflicts
- CPU-pinned memory pools reduce host-device latency

### 2.6 Beam Search / DFS Decoding
- Depth-first search over ARC token vocabulary (0-9, newline, EOS)
- Beam threshold: -log(0.1) = 2.303 NLL per token
- Augmented re-ranking across 8 views per candidate
- Known scores cached to avoid redundant GPU passes
- Timeout: 720s DFS, 1200s total per puzzle

---

## 3. Why AGI-2 Score Is Lower — Limitations

### 3.1 Model Size Ceiling
Qwen 4B has a hard reasoning ceiling on ARC-AGI-2 complexity. Tasks requiring multi-step logical inference benefit significantly from 7B-14B+ models. With optimal hyperparameters the 4B ceiling on AGI-2 is approximately 50-65%.

### 3.2 Training vs Inference Time Trade-offSimple puzzles:  ~400s training + ~800s inference = solved ✅
Complex puzzles: ~3000s+ training = timeout, 0s inference ❌
Example: puzzle 981571dc took 3377s training aloneepochs=2 caused timeouts on 3/4 eval puzzles.
Fixed with epochs=1 but time pressure remains on hardest puzzles.

### 3.3 No Ensemble
Top leaderboard solutions use 3-5 model ensembles with majority voting. Single model has higher per-puzzle variance. Ensemble alone adds estimated 10-15% score.

### 3.4 Hardware Constraint
4x L4 (88GB total) vs competitors using A100/H100 clusters:
- Larger VRAM enables bigger batch sizes during TTT
- Higher LoRA rank possible
- Longer sequence lengths
- More beam candidates per second

### 3.5 Beam Search Pruning
Threshold -log(0.1) prunes low-confidence tokens aggressively. Some correct solutions pruned early when model is uncertain. Wider beam would recover these at cost of compute.

### 3.6 TRL/Unsloth Compatibility Issues
Multiple crashes due to Unsloth patching conflicts with TRL trainers. Required patching 12 trainer classes before import. Cost significant development time that could have been spent on hyperparameter tuning.

---

## 4. What Would Push AGI-2 To 90%+

| Change | Expected Gain |
|--------|--------------|
| Qwen 7B backbone | +10-15% |
| 3-model ensemble | +10-15% |
| Learned search policy | +5-10% |
| n=64 augmentations | +3-5% |
| Better selection algorithm | +3-5% |
| A100 GPU (more VRAM) | +5-10% |

---

## 5. Conclusion
TTT with RSLoRA enables rapid puzzle-specific specialization at inference time. GPU-side log_softmax and teacher-forced scoring eliminate key bottlenecks. The primary limitation is model capacity (4B) and hardware (L4 vs A100). AGI-3 demonstrates that combining speculative decoding with a strong reasoning model achieves 93.07%, validating the overall approach.

---

## 6. References
- Tufa Labs Duck harness: Jeroen Cottaar
- Unsloth: github.com/unslothai/unsloth
- RSLoRA: Kalajdzievski (2023)
- ARC-AGI: Francois Chollet (2019)
- Qwen3: Alibaba Cloud (2024)
- SGLang: Zheng et al. (2023)
- Pennyroyal: John Pezzulli

---

## 7. Code & Reproducibility
Full notebooks available at:
https://github.com/gatlavishweshwarreddy/arc-agi-solution

AGI-3: Copy notebook on Kaggle, select RTX PRO 6000, run all cells
AGI-2: Copy notebook on Kaggle, select 4x L4 GPU, run all cells
