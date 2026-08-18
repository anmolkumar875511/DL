# Deep Learning Projects

A collection of deep learning projects — implementations, experiments, and Kaggle-style competition pipelines built while studying and applying core DL architectures (CNNs, RNNs, GANs, Transformers, and beyond).

More projects will be added here over time as I continue building.

---

## Projects

### 1. [Hardware Parts Semantic Segmentation](./hardware_parts_segmentation.ipynb)

Multi-class semantic segmentation of hardware parts (hex nuts, washers, bolts, ball bearings, springs, o-rings) using an **Attention Residual U-Net**.

- **Architecture:** Custom U-Net with residual blocks in the encoder/decoder and attention gates on the skip connections
- **Loss:** Combined Cross-Entropy + Dice loss to handle class imbalance
- **Training:** Mixed-precision (AMP), gradient clipping, cosine-annealing LR schedule
- **Inference:** Test-Time Augmentation (horizontal + vertical flip averaging)
- **Output:** RLE-encoded submission in a Kaggle-style competition format

**Tech stack:** PyTorch, Albumentations, NumPy, Pandas

[View notebook](./hardware_parts_segmentation.ipynb)

---

## Tech Stack

| Category | Tools |
|---|---|
| Frameworks | PyTorch |
| Data & Augmentation | Albumentations, NumPy, Pandas, PIL |
| Environment | Jupyter Notebook, Kaggle |

## Repository Structure

```
dl/
├── hardware_parts_segmentation.ipynb
└── README.md
```

## About

Maintained by [Anmol Kumar Shaharwal](https://github.com/anmolkumar875511) as a running log of deep learning work — spanning CNN/U-Net architectures, RNNs, GANs, and Transformer-based models. Each project folder contains its own notebook (and, where applicable, a project-specific README with more detail).

## License

This repository is open for reference and learning purposes. Feel free to explore and adapt the code.
