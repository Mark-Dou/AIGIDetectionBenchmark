# **AIGIDetectionBenchmark**

**The official code of** [*"Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection"*](#)! 

This repository provides a comprehensive **benchmark** for **AI-generated image detection**, integrating advanced methods to ensure generalization and adaptability in continually evolving tasks.


## **Overview**

The AIGIDetectionBenchmark integrates the following components:  

- **Existing AI-generated image detection models** : A comprehensive suite of mainstream detection methods for benchmarking across various datasets and architectures.  
- **Continual learning methods** : Ensures detection models stay adaptive to new generative methods while retaining knowledge of previous tasks.  
- **Training and evaluation workflows** : Ready-to-use code and pre-trained model weights for seamless implementation.  

##  **Supported Models for AI-generated Image Detection**

The following state-of-the-art AI-generated image detection models are integrated into the benchmark:

| **Method**            | **Paper**                                                                 | **Training Code** | **Testing Code** | **Model Weights** |
|:-----------------------:|:-----------------------------------------------------------------------------------:|:------------------:|:-----------------:|:------------------:|
| **CNNSpot**           | Cnn-generated images are surprisingly easy to spot... for now                     | ✅                | ✅               | ✅                |
| **FreDect**           | Leveraging frequency analysis for deep fake image recognition                     | ✅                | ✅               | ✅                |
| **GramNet**           | Global texture enhancement for fake face detection in the wild                    | ✅                | ✅               | ✅                |
| **LGrad**             | Learning on gradients: Generalized artifacts representation                       | ✅                | ✅               | ✅                |
| **LNP**               | Detecting generated images by real images                                         | ✅                | ✅               | ✅                |
| **UnivFD**            | Towards universal fake image detectors that generalize across models              | ✅                | ✅               | ✅                |
| **FatFormer**         | Forgery-aware adaptive transformer for generalizable synthetic detection          | ✅                | ✅               | ✅                |
| **NPR**               | Rethinking up-sampling operations in generative networks                          | ✅                | ✅               | ✅                |
| **SAFE**              | Improving synthetic image detection towards generalization                        | ✅                | ✅               | ✅                |
| **SimE (Ours)** | Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection| ✅                | ✅               | ✅                |


## **Supported Continual Learning Methods**

The following prominent continual learning techniques are integrated to ensure adaptability over time:

| **Method** | **Paper** | **Training Code** | **Testing Code** |
|:------------:|:-----------:|:------------------:|:-----------------:|
| **Seq**    |     /      | ✅                | ✅               |
| **Joint**  |     /      | ✅                | ✅               |
| **ER**     |On tiny episodic memories in continual learning           | ✅                | ✅               |
| **EWC**    |Overcoming catastrophic forgetting in neural networks           | ✅                | ✅               |
| **OSLA**   |Online structured laplace approx- imations for overcoming catastrophic forgetting           | ✅                | ✅               |
| **A-GEM**  |Efficient lifelong learning with a-gem           | ✅                | ✅               |
| **SI**     |Continual learning through synaptic intelligence           | ✅                | ✅               |
| **iCaRL**  |icarl: Incremental classifier and representation learning           | ✅                | ✅               |
| **Linear (Ours)** |Generalizable and Adaptive Continual Learning Framework for AI-generated Image Detection           | ✅                | ✅               |

These methods effectively combat catastrophic forgetting and ensure detection models retain previous performance while adapting to new AI-generated image data.


## **Datasets**

### **Training Datasets**
To train the detection model for AI-generated images, we follow the widely recognized protocol adopted by most methods. Specifically, we train on the **ProGAN dataset**, which is provided by [CNNSpot](https://github.com/peterwang512/CNNDetection). 

### **Evaluation Datasets**
The evaluation datasets are organized into two main categories:
- **GAN-generated images**:
  For GAN-based methods, we employ 8 types of GANs for testing, including:

  ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, and DeepFake.

  These datasets are sourced from benchmarks:[CNNSpot](https://github.com/peterwang512/CNNDetection)
  
- **Diffusion model-generated images**:
  For the DMs dataset, we collect 19 types of SOTA DMs, including:

   DDPM, iDDPM, ADM, DALLE, GLIDE, LDM, PNDM, Wukong, SDv1.4, Midjourney, SDv1.5, VQDM, SDV2.1, SDXL1.0, SD-Turbo, SDXL-Turbo, SDv3.0, PixArt-sigma, FLUX.1 from UniversalFakeDetect, GenImage, DRCT-2M, DiTFake.

  These datasets are sourced from benchmarks: [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect), [GenImage](https://github.com/GenImage-Dataset/GenImage), [DRCT-2M](https://github.com/beibuwandeluori/DRCT), and [DiTFake](https://github.com/ouxiang-li/safe).

### **Continual Learning Datasets**
For continual learning experiments, We meticulously organize these datasets to align with the chronological emergence of that our approach accurately reflects real-world scenarios in the continual learning context. 

![Continual_Learning_Datasets](https://github.com/Mark-Dou/AIGIDetectionBenchmark/blob/main/Assets/CL_Datasets.png)


