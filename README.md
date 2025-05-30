# FSE Vision Prototypes: Architectural Evolution
**Author: Pirassena Sabaratnam**

## Overview
This repository archives the architectural evolution of the **Field Signal Engine (FSE)** for computer vision, developed between early and mid-2025. It documents the transition from standard deep learning frameworks (TensorFlow/Keras) to custom-built, high-performance neural engines (Native Flow Field).

These prototypes served as the research foundation for the **FLUXA AR Perception Engine**, exploring the intersection of continuous field dynamics and real-time multi-modal perception.

## Architectural Journey

### 1. TensorFlow Prototype (`FSEFLUXATF`)
The first implementation of the FSE philosophy within a high-level framework:
- **Core Components**: Initial development of **FLIT (Floating Information Unit)** and **CSE (Continuous State Element)** as custom Keras layers.
- **Field Dynamics**: Early exploration of continuous state updates using momentum-based memory mixing.
- **Training Infrastructure**: GCS-integrated data pipelines for Vertex AI, featuring automated checkpointing and TensorBoard logging.

### 2. CNN-Hybrid Iteration (`FSEFLUXACNN`)
A transitional architecture that combined traditional spatial feature extraction with FSE field-evolution logic:
- **Anatomy**: Integrated **FIL (Field Interaction Layer)** blocks into CNN backbones (MobileNet/EfficientNet).
- **Multi-Modal Synthesis**: Refined the interaction between spatial fields (Keypoints/Segmentation) and bottleneck fields (Env Lighting).
- **Metrics (May 2025)**: Evaluated architectural stability and task-coherence during the transition to native computation.

## Core Terminology (Research Context)
The project utilizes specialized nomenclature for its internal field-processing units, mapped to standard Machine Learning concepts:
- **FLIT (Floating Information Unit)**: Conceptually analogous to **Adaptive Stochastic Neurons** or **Dynamic State Vectors** with learnable evolution rates.
- **CSE (Continuous State Element)**: Functionally similar to **Spatio-Temporal Memory Cells** or **Global Context Latents** with momentum-based state updates.
- **FIL (Field Interaction Layer)**: A specialized **Feature Fusion/Modality-Mixing Block** that manages continuous, wave, or quantum field transformations.
- **SYNTHA**: A **Global Orchestrator** or **Context-Conditioned Gating Module** for cross-task synchronization.

## Research Findings & Metrics
The results archived in this repository (see `.xlsx` and `.csv` files) document the stability and convergence rates of FSE-based vision tasks during the May 2025 testing phase. These findings directly informed the development of the vectorized, low-latency **FSENativeFLUXAFF** engine.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---
*Developed in 2025 as part of the Auralith Inc. Research.*
