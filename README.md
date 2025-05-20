# FSE Vision Prototypes — From CNNs to Continuous Fields
**Author: Pirassena Sabaratnam**

## Overview
This repository archives the iterative research process of developing **continuous field dynamics** for computer vision, motivated by the limitations of standard CNNs in real-time AR.

The core observation: traditional CNNs treat image features as discrete, rigid tensors. For real-time AR, we need spatio-temporal coherence — features that evolve smoothly across frames, not features that are recomputed independently per frame.

## Architectural Evolution

### 1. TensorFlow Prototype (`01-TF-Prototype`)
First implementation of continuous field layers as custom Keras layers.
- **Goal**: Prove that neural activations can be modeled as continuous signals with learnable evolution rates, rather than static feature maps.
- **Result**: The math worked, but TensorFlow/Keras overhead was too high for the low-latency field updates needed for real-time AR.

### 2. CNN + Continuous Field Hybrid (`02-CNN-Hybrid`)
Transitional model augmenting standard CNN backbones (MobileNet/EfficientNet) with custom field-evolution layers.
- **Goal**: Determine if adding continuous field layers to standard CNNs improves multi-task coherence (keypoints + segmentation + normals predicted from shared evolving features).
- **Result**: Improved stability in surface normal estimation. Confirmed that a native engine (not a framework plugin) was necessary for real-time performance.

## Conclusion
This research established that continuous field dynamics require a purpose-built engine — standard deep learning frameworks add too much overhead for the fine-grained field updates the approach requires. This led directly to the development of the **FlowField engine** (see [FSENativeFLUXAFF](https://github.com/ps2604/FSENativeFLUXAFF)).

## License
Apache License 2.0.

---
*Developed May 2025 — Auralith Inc.*
