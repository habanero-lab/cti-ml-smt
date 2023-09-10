# Concrete Type Inference for Code Optimization using Machine Learning with SMT Solving

The code for the paper "Concrete Type Inference for Code Optimization using Machine Learning with SMT Solving" by Fangke Ye, Jisheng Zhao, Jun Shirako, and Vivek Sarkar.

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
