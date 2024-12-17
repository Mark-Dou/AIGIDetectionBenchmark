# **AIGIDetectionBenchmark**

**The official code of** [*"Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection"*](#)! 

This repository provides a comprehensive **benchmark** for **AI-generated image detection**, integrating advanced methods to ensure generalization and adaptability in continually evolving tasks.


## **Overview**

The AIGIDetectionBenchmark integrates the following components:  

- **Existing AI-generated image detection models** : A comprehensive suite of mainstream detection methods for benchmarking across various datasets and architectures.  
- **Continual learning methods** : Ensures detection models stay adaptive to new generative methods while retaining knowledge of previous tasks.  
- **Training and evaluation workflows** : Ready-to-use code and pre-trained model weights for seamless implementation.  

---
##  **Supported Models for AI-generated Image Detection**

The following state-of-the-art AI-generated image detection models are integrated into the benchmark:

| **Method**            | **Paper**                                                                 | **Training Code** | **Testing Code** | **Model Weights** |
|-----------------------|--------------------------------------------------------------------------|------------------|-----------------|------------------|
| **CNNSpot**           | Cnn-generated images are surprisingly easy to spot... for now           | ✅                | ✅               | ✅                |
| **FreDect**           | Leveraging frequency analysis for deep fake image recognition           | ✅                | ✅               | ✅                |
| **GramNet**           | Global texture enhancement for fake face detection in the wild          | ✅                | ✅               | ✅                |
| **LGrad**             | Learning on gradients: Generalized artifacts representation             | ✅                | ✅               | ✅                |
| **LNP**               | Detecting generated images by real images                               | ✅                | ✅               | ✅                |
| **UnivFD**            | Towards universal fake image detectors that generalize across models    | ✅                | ✅               | ✅                |
| **FatFormer**         | Forgery-aware adaptive transformer for generalizable synthetic detection| ✅                | ✅               | ✅                |
| **NPR**               | Rethinking up-sampling operations in generative networks                | ✅                | ✅               | ✅                |
| **SAFE**              | Improving synthetic image detection towards generalization              | ✅                | ✅               | ✅                |
| **SimE (Our Method)** | Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection| ✅                | ✅               | ✅                |

These models allow for thorough benchmarking across diverse datasets and generative models.

---

## **Supported Continual Learning Methods**
The following prominent continual learning techniques are integrated to ensure adaptability over time:
- **Elastic Weight Consolidation (EWC)** 🔒  
- **Learning Without Forgetting (LwF)** 🔄  
- **Replay-based Methods** (e.g., Experience Replay) 🔁  
- **Knowledge Distillation** 🌟  
- **Online Parameter Updates** 📈  

These methods effectively combat **catastrophic forgetting** and ensure detection models retain previous performance while adapting to new AI-generated image data.

---
