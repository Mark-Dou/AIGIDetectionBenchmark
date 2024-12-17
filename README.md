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
To train the detection model for AI-generated images, we follow the widely recognized protocol adopted by most methods. Specifically, we train on the **ProGAN dataset**, which is provided by **[Cnn-generated images are surprisingly easy to spot... for now](https://github.com/peterwang512/CNNDetection)**. 

### **Evaluation Datasets**
The evaluation datasets are categorized into two main types:
- **GAN-generated images**:
  For the GANs dataset, we utilize the 8 types of GANs for testing, including ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN and DeepFake. The original download link can be found here.
- **Diffusion model-generated images**:
  For the DMs dataset, we collect 19 types of SOTA DMs, including DDPM, iDDPM, ADM, DALLE, GLIDE, LDM, PNDM, Wukong, SDv1.4, Midjourney, SDv1.5, VQDM, SDV2.1, SDXL1.0, SD-Turbo, SDXL-Turbo, SDv3.0, PixArt-sigma, FLUX.1 from UniversalFakeDetect, GenImage, DRCT-2M, DiTFake.

### **Continual Learning Datasets**
For continual learning experiments, We meticulously organize these datasets to align with the chronological emergence of that our approach accurately reflects real-world scenarios in the continual learning context. 

**Table: Continual learning dataset.**

| Family       | Method           | Date    |
|--------------|------------------|--------|
| **GAN**      | ProGAN           | Oct 17 |
|              | CycleGAN         | Nov 17 |
|              | StarGAN          | Nov 17 |
|              | BigGAN           | Sep 18 |
|              | StyleGAN         | Dec 18 |
|              | GauGAN           | Mar 19 |
|              | StyleGAN2        | Dec 19 |
| **DeepFake** | Deepfake         | Nov 17 |
| **Diffusion**| DDPM             | Jun 20 |
|              | ADM              | May 21 |
|              | iDDPM            | Dec 21 |
|              | DALL·E           | Dec 21 |
|              | GLIDE            | Dec 21 |
|              | LDM              | Dec 21 |
|              | PNDM             | Feb 22 |
|              | Wukong           | Feb 22 |
|              | SD v1.4          | Apr 22 |
|              | Midjourney       | Jul 22 |
|              | SD v1.5          | Aug 22 |
|              | VQDM             | Aug 22 |
|              | SD v2.1          | Nov 22 |
|              | SDXL v1.0        | Jul 23 |
|              | SD-Turbo         | Nov 23 |
|              | SDXL-Turbo       | Nov 23 |
|              | SD v3.0          | Feb 24 |
|              | PixArt-Σ         | Mar 24 |
|              | FLUX.1           | Aug 24 |


