# Deep Learning Projects

A collection of deep learning projects: implementations, experiments and Kaggle-style competition pipelines built while studying and applying core architectures (CNNs, U-Nets, residual networks, and more to come).

Every project is a single Jupyter notebook. The notebooks are written to be read from top to bottom: each step has a short explanation, the code uses descriptive names, and the settings are collected at the top.

---

## Projects

| # | Project | Task | Model | Metric |
|---|---|---|---|---|
| 1 | [Hardware Parts Semantic Segmentation](./hardware_parts_segmentation.ipynb) | Label every pixel of an image as one of 6 hardware parts or background | Attention Residual U-Net | Mean Dice |
| 2 | [ECG Heartbeat Arrhythmia Classification](./heartbeat_arrhythmia_classification.ipynb) | Classify single heartbeats into 4 arrhythmia categories | 1D residual CNN + timing-feature network | Macro F1 |

---

### 1. Hardware Parts Semantic Segmentation

[Open notebook](./hardware_parts_segmentation.ipynb)

Finds and outlines hex nuts, washers, bolts, ball bearings, springs and o-rings in photos.

- **Model:** a U-Net with residual blocks in the encoder and decoder, and attention gates on the skip connections. The gates learn to ignore background regions before the features are passed to the decoder.
- **Loss:** Cross-Entropy plus Dice loss, so small parts count as much as the large background.
- **Training:** 80/20 train/validation split, data augmentation (flips, rotations, shifts, colour changes), mixed-precision training, gradient clipping and a cosine-annealing learning-rate schedule. The best model is chosen by validation Dice score.
- **Prediction:** Test-Time Augmentation. Each test image is predicted three times (original, flipped left-right, flipped up-down) and the probabilities are averaged.
- **Output:** masks converted to run-length encoding (RLE) in `submission.csv`. Very small predicted regions (under 30 pixels) are removed.

**Tech stack:** PyTorch, Albumentations, NumPy, Pandas, Pillow

---

### 2. ECG Heartbeat Arrhythmia Classification

[Open notebook](./heartbeat_arrhythmia_classification.ipynb)

Classifies each heartbeat (250 ECG signal values plus 3 timing features) into one of four arrhythmia categories, labelled `0` to `3`. The competition ran from Aug 21 to Aug 25, 2026 and is scored with **Macro F1**, where all four classes count equally.

- **Data:** 71,748 training beats and 68,914 test beats. The classes are very unbalanced: about 45,000 beats in class 0, but only about 600 in class 3.
- **Model:** two branches joined before the final decision.
  - A 1D convolutional network with three residual blocks reads the ECG signal.
  - A small dense network reads the three RR-interval features (`pre_rr`, `post_rr`, `rr_ratio`).
- **Handling class imbalance:** a weighted sampler and a weighted loss, both using `1 / sqrt(class size)`, plus label smoothing.
- **Augmentation:** random noise, amplitude scaling and small time shifts on the training signals.
- **Training:** 5-fold stratified cross-validation. In each fold, the best weights are chosen using a smoothed validation F1 (average of the last 3 epochs), with early stopping and a learning rate that is reduced when progress stalls.
- **Prediction:** the class probabilities of the 5 fold models are averaged (an ensemble) and the most likely class is chosen.

**Results from the saved run** (5-fold cross-validation on the training data):

| Measure | Value |
|---|---|
| Overall out-of-fold Macro F1 | 0.8933 |
| Mean Macro F1 across folds | 0.8945 +/- 0.0125 |
| Accuracy | 0.9835 |

| Class | Precision | Recall | F1 | Beats |
|---|---|---|---|---|
| 0 | 0.9949 | 0.9881 | 0.9915 | 45,000 |
| 1 | 0.9249 | 0.9719 | 0.9478 | 3,599 |
| 2 | 0.9928 | 0.9788 | 0.9858 | 22,552 |
| 3 | 0.5096 | 0.8894 | 0.6480 | 597 |

Class 3 is the weak point: the model finds most of its beats (high recall) but also wrongly gives the label to beats from other classes (low precision). It lowers the Macro F1 the most. Better handling of this class is the clearest way to improve the score.

**Tech stack:** PyTorch, scikit-learn, NumPy, Pandas

---

## Getting Started

### Requirements

- Python 3.10 or newer
- A GPU is strongly recommended. The ECG notebook takes about 30 minutes per fold on a GPU, and the segmentation notebook trains for 100 epochs.

```
pip install torch numpy pandas scikit-learn albumentations pillow tqdm jupyter
```

### Running a notebook

1. Download the competition data and put it somewhere you can reach.
2. Open the notebook and change the path settings in its first settings cell (`DATA_DIR` for the ECG notebook, `DATASET_DIR` for the segmentation notebook). The paths in the notebooks point to the Kaggle input folders they were run on.
3. Run all cells from top to bottom. Each notebook writes a `submission.csv` in the current folder.

To get a quick first result, lower `NUMBER_OF_FOLDS` and `MAX_EPOCHS` (ECG notebook) or `NUMBER_OF_EPOCHS` (segmentation notebook) in the settings cell.

## Tech Stack

| Category | Tools |
|---|---|
| Frameworks | PyTorch |
| Data and augmentation | Albumentations, NumPy, Pandas, Pillow |
| Evaluation | scikit-learn |
| Environment | Jupyter Notebook, Kaggle |

## Repository Structure

```
DL/
├── hardware_parts_segmentation.ipynb
├── heartbeat_arrhythmia_classification.ipynb
└── README.md
```

Running the notebooks also creates `submission.csv` and, depending on the notebook, a model file (`best_unet.pth` or `model_kfold.pt`).

## About

Maintained by [Anmol Kumar Shaharwal](https://github.com/anmolkumar875511) as a running log of deep learning work. Each project is a self-contained notebook.

## License

This repository is open for reference and learning purposes. Feel free to explore and adapt the code.
