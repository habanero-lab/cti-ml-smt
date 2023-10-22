# Concrete Type Inference for Code Optimization using Machine Learning with SMT Solving

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8332121.svg)](https://doi.org/10.5281/zenodo.8332121)

The code for the paper [Concrete Type Inference for Code Optimization using Machine Learning with SMT Solving](https://dl.acm.org/doi/10.1145/3622825).

```
@article{10.1145/3622825,
    author     = {Ye, Fangke and Zhao, Jisheng and Shirako, Jun and Sarkar, Vivek},
    title      = {Concrete Type Inference for Code Optimization Using Machine Learning with SMT Solving},
    year       = {2023},
    issue_date = {October 2023},
    publisher  = {Association for Computing Machinery},
    address    = {New York, NY, USA},
    volume     = {7},
    number     = {OOPSLA2},
    url        = {https://doi.org/10.1145/3622825},
    doi        = {10.1145/3622825},
    journal    = {Proc. ACM Program. Lang.},
    month      = {oct},
    articleno  = {249},
    numpages   = {28}
}
```

## Environment Setup

A C++ compiler that supports C++14 is required.

It is recommended to use a conda environment to install the Python dependencies:
```bash
conda create -n cti python=3.9
conda activate cti
pip install -r requirements.txt
```

Note that the cvc5 package specified in [requirements.txt](requirements.txt) may not be available for all platforms. If the installation fails, please refer to [cvc5's official website](https://cvc5.github.io/) for installation instructions.

The PyTorch and torch-scatter packages included in [requirements.txt](requirements.txt) are for CPU only and are used for model inference in the experiments. To train our deep learning models using a GPU, please refer to the websites of [PyTorch](https://pytorch.org/) and [torch-scatter](https://github.com/rusty1s/pytorch_scatter) for instructions on how to install the GPU version.

## Benchmarks

Please see the [benchmarks](benchmarks) directory.

## The SciPy Dataset and Machine Learning Type Inference Models

For the SciPy dataset and the Freq, DeepTyper-FS, and CodeT5-FT models, please refer to the [training](training) directory.

For the GPT-4 model, please refer to the [gpt4](gpt4) directory.

## Intrepydd Compiler

We provide a modified version of the Intrepydd compiler introduced in [the Intrepydd paper](https://dl.acm.org/doi/10.1145/3426428.3426915). Please refer to the [intrepydd](intrepydd) directory for more details.
