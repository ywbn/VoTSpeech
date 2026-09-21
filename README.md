# VoTSpeech

**VoTSpeech: Decoupling Voice Design from Speech Generation in Instruction-Following TTS**

[Online demo](https://ywbn.github.io/VoTSpeech/)

VoTSpeech (Voice-of-Thought Speech) separates voice design from speech generation through an explicit continuous voice representation. Given a natural-language voice instruction, a shared causal language model first guides a flow-based Voice DiT to design the voice. The resulting voice latent then conditions both language modeling and acoustic generation.

![VoTSpeech architecture](assets/architecture.png)

## Highlights

- Explicitly separates *what voice to create* from *what that voice says*.
- Combines speaker identity features and pooled audio VAE latents to supervise the voice representation.
- Uses dual-path conditioning to connect semantic planning with acoustic rendering.
- Trained with 1,543.52 hours of Chinese task-specific adaptation data.
- Achieves 84.8% APS, 77.3% DSD, and 66.5% RP instruction-following accuracy on InstructTTSEval-ZH, with a 2.58% character error rate.

## Paper results

| Model | APS ↑ | DSD ↑ | RP ↑ | CER ↓ | Naturalness ↑ | Expressiveness ↑ | Adherence ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **VoTSpeech (Ours)** | 84.8 | **77.3** | **66.5** | 2.58 | **4.12 ± 0.22** | **4.30 ± 0.18** | **4.26 ± 0.19** |
| Ming-Omni-TTS-0.5B | 84.9 | 72.2 | 53.9 | **2.48** | 3.71 ± 0.27 | 3.71 ± 0.24 | 3.75 ± 0.25 |
| Qwen3-TTS-12Hz-1.7B-VD | **87.1** | 76.0 | 55.2 | 2.99 | 4.08 ± 0.21 | 4.10 ± 0.19 | 4.06 ± 0.20 |
| MOSS-VoiceGenerator | 73.1 | 70.2 | 58.9 | 5.03 | 3.86 ± 0.22 | 3.72 ± 0.18 | 3.70 ± 0.19 |
| VoxCPM2 | 84.7 | 71.8 | 56.8 | 2.58 | 3.80 ± 0.25 | 3.82 ± 0.18 | 3.82 ± 0.17 |

Instruction-following performance is evaluated on **InstructTTSEval-ZH**. Following its evaluation protocol, **Gemini 2.5 Pro** assesses compliance for Acoustic-Parameter Specification (APS), Descriptive-Style Directive (DSD), and Role-Play (RP), with accuracy reported as a percentage. Speech intelligibility is measured by the overall character error rate (CER) across all three categories using a pretrained **Paraformer-ZH** ASR model.

For subjective evaluation, **24 listeners** rate outputs from every system on the same **10 randomly selected examples** using a five-point scale. Naturalness measures perceived audio quality, Expressiveness measures how well the delivery fits the synthesis text, and Adherence measures compliance with the voice-design instruction. The table reports mean opinion scores with **95% confidence intervals**.

## Authors

Wenbing Yang¹, Qihang Lu², Zihan Sun², Peilei Jia², Yingming Gao¹, Ya Li¹, and Jun Gao²

¹ Beijing University of Posts and Telecommunications  
² Hello Group Inc.
