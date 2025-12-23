# LUGAI - Latent Uncertainty Guided Adversary Intervention

## 1. Introduction

Deep learning models have achieved state-of-the-art performance across numerous computer vision tasks. However, their susceptibility to adversarial examples, inputs that are minimally perturbed yet cause incorrect predictions—remains a fundamental challenge. These adversarial examples exploit the high-dimensional and non-linear nature of neural networks, leading to unreliable behavior under adversarial conditions.

Most existing defense mechanisms rely on adversarial training, which requires retraining models on adversarial data and significantly increases training cost. Moreover, such defenses often fail against unseen or adaptive attacks. This project explores an alternative paradigm: **post-hoc defense through detection and correction**, rather than retraining.


## 2. Motivation

The motivation behind LUGAI is based on two key observations:

1. Adversarial inputs often produce abnormal behavior in the latent feature representations of neural networks.
2. While adversarial perturbations mislead classifiers, they typically preserve the semantic structure of the original input.

By leveraging these properties, LUGAI aims to:

* Identify adversarial inputs through latent-space irregularities.
* Restore corrupted inputs using learned denoising transformations.
* Maintain high classification accuracy without modifying the original classifier.


## 3. System Overview

LUGAI operates as a two-stage defense pipeline:

1. **Detection Stage**
   Latent uncertainty and reconstruction deviation are used to determine whether an input is adversarial.

2. **Purification Stage**
   A denoising autoencoder is applied to detected adversarial inputs to remove perturbations before final classification.

This design ensures that clean inputs are processed efficiently, while adversarial inputs undergo corrective intervention.


## 4. Adversarial Threat Model

This project considers white-box adversarial attacks, where the attacker has access to the model architecture and parameters. The following attacks are implemented:

* **Fast Gradient Sign Method (FGSM)**
  Generates adversarial samples using a single gradient step in the direction of maximum loss increase.

* **Projected Gradient Descent (PGD)**
  An iterative extension of FGSM that produces stronger adversarial examples.

* **DeepFool**
  An optimization-based attack that finds minimal perturbations required to change the model’s prediction.

These attacks serve as benchmarks to evaluate the robustness of the proposed defense.


## 5. Latent Uncertainty-Based Detection

Traditional confidence-based detection methods are insufficient, as neural networks often assign high confidence to adversarial inputs. LUGAI instead relies on latent uncertainty, which captures instability in internal feature representations.

Detection is performed using:

* Reconstruction error from an autoencoder trained on clean data.
* Statistical deviation in latent representations compared to clean input distributions.

Inputs exceeding predefined uncertainty thresholds are flagged as adversarial.


## 6. Denoising Autoencoder for Purification

The purification stage employs a denoising autoencoder trained to map adversarial inputs back to their clean counterparts. The model is trained using adversarial-clean image pairs, minimizing mean squared reconstruction error.

Formally, the training objective is:

$$[
\min_{\theta} \mathbb{E}*{(x*{adv}, x)} \left[ | x - DAE_\theta(x_{adv}) |_2^2 \right]
]$$

This enables the system to remove adversarial noise while preserving semantic content.


## 7. Methodology

The experimental workflow is as follows:

1. Train a baseline convolutional neural network classifier on clean MNIST data.
2. Generate adversarial examples using FGSM, PGD, and DeepFool.
3. Train a denoising autoencoder on adversarial-clean image pairs.
4. Detect adversarial inputs using latent uncertainty metrics.
5. Purify detected inputs prior to final classification.
6. Evaluate performance in terms of accuracy, detection rate, and recovery rate.


## 8. Experimental Results

Experiments were conducted on the MNIST dataset. The following results summarize the effectiveness of LUGAI:

| Scenario                       | Classification Accuracy |
| ------------------------------ | ----------------------- |
| Clean Inputs                   | 99.28%                  |
| FGSM Attacked Inputs (ε = 0.3) | 22.97%                  |
| After LUGAI Purification       | 93.02%                  |
| Accuracy Recovery Rate         | 91.80%                  |

These results demonstrate that LUGAI substantially mitigates the impact of adversarial attacks without retraining the classifier.


## 9. Installation and Execution

### Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset Preparation

```bash
python data_utils.py
```

### Training

```bash
python train_baseline.py
python train_denoising_autoencoder.py
```

### Evaluation

```bash
python evaluate_attacks.py
python evaluate_detection.py
python evaluate_purification.py
```

### Interactive Demonstration

```bash
streamlit run app.py
```


## 10. Limitations

* Evaluation is limited to the MNIST dataset.
* Fixed detection thresholds may be vulnerable to adaptive adversaries.
* Autoencoder capacity limits scalability to higher-resolution datasets.


## 11. Future Scope

* Extension to CIFAR-10 and ImageNet
* Adaptive thresholding mechanisms
* Defense against adaptive adversarial attacks
* Latent-space explainability and interpretability analysis
* Deployment-oriented optimization for real-time systems


## 12. Project Information

**Supervisor:** Dr. Sanjay B. Sonar

**Contributors:**
- Jayal Shah
- Sakshi Makwana
- Mayank Jangid


## 13.Demo

Below is a screenshot of the **LUGAI Streamlit application**, which demonstrates detection and purification of adversarial inputs in real time.

![LUGAI Demo Screenshot](results/figures/image.png)

