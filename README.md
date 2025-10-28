# CycleGAN: photos to Monet-style paintings

A course project for NI-MVI (Computational Intelligence Methods) at FIT CTU, winter semester 2022/23. It is a CycleGAN written from scratch in PyTorch for the Kaggle competition [I'm Something of a Painter Myself](https://www.kaggle.com/competitions/gan-getting-started). The competition asks for a model that turns photos into Monet-style paintings.

| | | | | |
|---|---|---|---|---|
| **Photo** | ![](images/photo-1.jpg) | ![](images/photo-100.jpg) | ![](images/photo-500.jpg) | ![](images/photo-900.jpg) |
| **Generated** | ![](images/monet-1.jpg) | ![](images/monet-100.jpg) | ![](images/monet-500.jpg) | ![](images/monet-900.jpg) |

These are four of the 1,001 images generated with the trained photo → Monet generator. The outputs keep the layout of the scene and soften it toward a painting. They are still blurry, though, with visible checkerboard artefacts, and they don't look much like Monet.

## Approach

- **Data:** 300 Monet paintings and about 7,000 photos, all 256×256 and unpaired. I used the JPG copies, dropped the duplicate TFRecords, and loaded them with torchvision's `ImageFolder`.
- **Why CycleGAN:** the two sets aren't paired, so a plain GAN isn't enough. A CycleGAN ([Zhu et al., 2017](https://arxiv.org/abs/1703.10593)) trains two generators (photo → Monet and Monet → photo) and two discriminators. A cycle-consistency loss requires that an image translated to the other domain and back matches the original.
- **Generators:** ResNet-style:
  - a 7×7 convolution
  - two stride-2 downsampling convolutions
  - 9 residual blocks
  - upsampling with sub-pixel convolution (`PixelShuffle`)
  - a `tanh` output

  They use instance normalisation and reflection padding.
- **Discriminators:** PatchGAN. Four stride-2 convolution blocks (64 → 512 channels) followed by a one-channel output map.
- **Losses:**
  - least-squares (MSE) adversarial loss
  - L1 cycle loss (weight 10)
  - L1 identity loss (weight 5)
- **Training:** Adam (learning rate 2e-4, betas 0.5 and 0.99), batch size 1.
  - The report used a 100-epoch run.
  - A later run went to 200 epochs, about 7 hours at roughly 2 minutes per epoch on a CUDA GPU.

Most architecture choices and hyperparameters follow the CycleGAN paper and the public implementations cited in the report, mainly [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and Amy Jang's [Monet CycleGAN tutorial](https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial).

![Training losses over 200 epochs](images/losses-200-epochs.png)

The chart shows the seven training losses over the 200-epoch run: identity, adversarial and cycle losses for the generators, plus one loss per discriminator. They flatten after about 50 epochs. The original plot has no legend.

## Known issues

- **Most photos are never used.** Each epoch zips the photo loader with the much smaller Monet loader, so it only sees about 300 of the ~7,000 photos.
- **The learning rate is far too small.** `custom_lr` returns an absolute learning rate, but `LambdaLR` multiplies the base rate by whatever the function returns. As committed, the effective rate is 2e-4 × 2e-4 = 4e-8. The function should return a factor instead: 1.0, then decreasing linearly to 0.
- **The notebooks are rough.**
  - Both have leftover TODOs and commented-out code.
  - The saved outputs in `notebook.ipynb` come from a short 10-epoch run.
  - `notebook_200_epoch.ipynb` was last used to load the saved models on the CPU. Its per-epoch sample images were removed to keep the file small.

## Files

| File | Contents |
|---|---|
| `notebook.ipynb` | Data loading, models, training loop, loss plot, saving the models |
| `notebook_200_epoch.ipynb` | The same code with the output of the 200-epoch run |
| `report.pdf` | The final two-page report (on the 100-epoch run) |
| `checkpoint.pdf` | The milestone report (December 2022) |
| `report.zip` | LaTeX source of the report |
| `images/` | The sample outputs and the loss chart used in this README |

## Running it

Download the data from [Kaggle](https://www.kaggle.com/competitions/gan-getting-started/data), which needs an account. There is also a [Google Drive copy](https://drive.google.com/file/d/1-XE75v923yUErA51WDinQCtctNFTk-iW/view?usp=sharing) with both the original and the restructured data. The notebook expects this layout:

```
data-monet/
├── monet/monet/*.jpg
└── photo/photo/*.jpg
```

The extra nested folder is there because `ImageFolder` labels images by subfolder.

```bash
pip install torch torchvision matplotlib numpy jupyter
jupyter notebook notebook.ipynb
```

The notebook picks CUDA, Apple MPS or the CPU automatically. `EPOCH_COUNT` is set to 200. Create `models/04/` before running the cell that saves the models.
