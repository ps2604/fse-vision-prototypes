# FSE Vision Prototypes: The Architectural Thesis
**Author: Pirassena Sabaratnam**

## Research Thesis: From Tensors to Fields
The development of the **Field Signal Engine (FSE)** was driven by a core observation during the creation of the **NAIMME AR app**: traditional Convolutional Neural Networks (CNNs) treat image features as discrete, rigid tensors. While effective for classification, this approach often lacks the "spatio-temporal fluidity" required for seamless, real-time Augmented Reality.

This repository archives the iterative research process of moving away from standard deep learning frameworks toward a custom **Flow Field** engine.

## Architectural Evolution

### 1. TensorFlow Prototype (`01-TF-Prototype`)
The first full-scale implementation of the FSE philosophy. 
- **Goal**: To prove that neural activations could be modeled as continuous signals (Fields) using custom Keras layers (`FLIT`, `CSE`, `FIL`).
- **Discovery**: While the math was sound, the TensorFlow/Keras overhead was too high to achieve the low-latency "field evolution" required for AR.

### 2. CNN-Hybrid Iteration (`02-CNN-Hybrid`)
A transitional model that integrated FSE blocks into industry-standard backbones (MobileNet/EfficientNet).
- **Goal**: To determine if augmenting standard CNNs with field-evolution logic could improve multi-task coherence (Keypoints + Segmentation + Normals).
- **Findings**: Documented in the May 2025 metrics, this version showed improved stability in surface normal estimation but confirmed that a native engine was necessary for true real-time performance.

## Core Philosophical Units
The project utilizes specialized nomenclature for its internal field-processing units, rooted in standard ML concepts:
- **FLIT (Floating Information Unit)**: Conceptually analogous to **Adaptive Stochastic Neurons** or **Dynamic State Vectors** with learnable evolution rates.
- **CSE (Continuous State Element)**: Functionally similar to **Spatio-Temporal Memory Cells** or **Global Context Latents** with momentum-based state updates.
- **FIL (Field Interaction Layer)**: A specialized **Feature Fusion/Modality-Mixing Block** that manages continuous, wave, or quantum field transformations.

## Conclusion
This research journey established that true "Continuous Field" dynamics are best executed outside of traditional high-level frameworks. This realization led directly to the development of the **FSENativeFLUXAFF** (Flowfield) engine, which utilizes custom vectorized kernels for optimized field computation.

## License
This project is licensed under the Apache License 2.0.

---
*Developed in 2025 as part of the Auralith Inc. Research.*
