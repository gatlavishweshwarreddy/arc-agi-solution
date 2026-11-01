# ARC-AGI Solution — AGI-2 & AGI-3
**Author:** Vishweshwar Reddy Gatla  
**GitHub:** https://github.com/gatlavishweshwarreddy/arc-agi-solution

---

---

## AGI-3 Solution (93.07%)

### Architecture
- **Base:** Tufa Labs Duck harness (credit: Jeroen Cottaar)
- **Model:** Qwen3.8-Flash-Next (switched from original)
- **Quantization:** Intel AutoRound
- **Draft Model:** Albucino quantized MTP checkpoint
- **Inference:** Pennyroyal (John Pezzulli's SGLang fork v2.5.3)
- **GPU:** RTX PRO 6000

### Key Modifications (patch applied at notebook start)
- Switched base model to Qwen3.8-Flash-Next
- Applied speculative decoding via quantized MTP draft
- Custom patches on top of SGLang fork (commit d00d88e)
- Contributions integrated from Mamy Ratsimbazafy and Gabriel Olympie

### Why 93.07%?
- Speculative decoding dramatically increased inference throughput
- Qwen3.8-Flash-Next superior reasoning on grid tasks
- Duck harness provides robust beam search + augmentation scoring

---

## AGI-2 Solution

### Architecture
- **Model:** Qwen 4B fine-tuned on ARC grid token representations
- **Method:** Test-Time Training (TTT) with RSLoRA
- **GPUs:** 4x Nvidia L4 (22GB each)
- **Framework:** Unsloth + HuggingFace Transformers
