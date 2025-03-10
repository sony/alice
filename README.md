
# **ALICE** - **Al**ignment by **I**nterventional **C**onsist**e**ncy


This is the code associated with the paper [Partial Alignment of Representations via Interventional Consistency](https://openreview.net/forum?id=eimAJqoIWt) presented at the Re-Align workshop at ICLR 2025.


## Installation

Assuming you have `pip`, you can install the dependencies listed in `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

## Usage

The project uses [`omnifig`](https://github.com/felixludos/omni-fig) to organize the scripts and config files.

Two scripts are provided:
- `collect` - organize and preprocess the data to save time during training
- `train` - trains the model

These can be run using the `fig` command (see `fig -h` for more information). For example, to run the `train` script with the settings to train a model on the ViT features of COCO images using a non-linear intervention model and all the same hyperparameters as the corresponding model in the paper:

```bash
fig train h/ws2 a/wide norm intv/module m/ced-man d/coco-img --classifier.dropout 0.1 --latent-dim 512 --small-width 1024
```

Equivalently, you can use the more familiar: `python main.py <script-name> <configs> <args>`.

Note: The current version of the code is still geared towards our specific in-house setup (e.g. settings in `config/h/ws2.yml`), but we are working on making it more user-friendly. We hope to provide a generally representative demo script as well as better documentation soon.

Until then, you can browse our main contributions in the code and configs directly (which are relatively readable).

For the code, the scripts `train` and `collect` entry points are in `src/op.py`. You will the most central components to our method in `src/interventions.py` and `src/models.py`. Other important components such as the datasets and the baselines can be found in `src/dataset.py` and `src/baselines.py` respectively.

For the configs in `config/`, generally, information is organized by:
- `a/` - architecture specific settings
- `d/` - dataset specific settings
- `h/` - host/hardware specific settings (probably not of interest to you)
- `intv/` - settings for the intervention module
- `m/` - method settings where `ced-man.yml` corresponds to our proposed method discussed in the paper with baselines included in `clip.yml` and `cyclip.yml`.

Additionally, there are several jupyter notebooks which show how to use the code in a more interactive manner. However, these are generally less documented and organized, so we recommend starting with the scripts for now.

