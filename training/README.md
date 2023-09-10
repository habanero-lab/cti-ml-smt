# The SciPy Dataset and Machine Learning Type Inference Models

This directory contains the code for creating the SciPy dataset and training/evaluating the Freq, DeepTyper-FS, and CodeT5-FT models.

The dataset and model files are available at Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8330329.svg)](https://doi.org/10.5281/zenodo.8330329)

## The SciPy Dataset

The SciPy dataset is created using a modified version of [MonkeyType](https://github.com/Instagram/MonkeyType), which is included in the [MonkeyType](MonkeyType) directory.
To create the SciPy dataset by yourself, please follow the instructions in [scipy_dataset/README.md](scipy_dataset/README.md).
Alteratively, you can download the prebuilt dataset from Zenodo: [dataset_scipy_raw.pkl](https://zenodo.org/record/8330329/files/dataset_scipy_raw.pkl?download=1).

## Preprocessing the SciPy Dataset for Training

Run the following commands to preprocess the SciPy dataset for training the models:

```bash
# For Freq and DeepTyper-FS
python run.py dataset -rd dataset_scipy_raw.pkl -d dataset_scipy_dt.pkl -v vocab_scipy_dt.pkl

# For CodeT5-FT
python run.py dataset-tf -rd dataset_scipy_raw.pkl -d dataset_scipy_t5.pkl
```

Alternatively, you can download the preprocessed dataset files from Zenodo:
[vocab_scipy_dt.pkl](https://zenodo.org/record/8330329/files/vocab_scipy_dt.pkl?download=1),
[dataset_scipy_dt.pkl](https://zenodo.org/record/8330329/files/dataset_scipy_dt.pkl?download=1), and
[dataset_scipy_t5.pkl](https://zenodo.org/record/8330329/files/dataset_scipy_t5.pkl?download=1).

## Training the Models

Run the following commands to train the models:

```bash
# Freq does not require extra training

# Train DeepTyper-FS with seed 42
python run.py train -dt -v vocab_scipy_dt.pkl -d dataset_scipy_dt.pkl -s models_dt_s42 -sd 42

# Train CodeT5-FT with seed 42
python run.py train-tf -d dataset_scipy_t5.pkl -s models_t5_s42 -sd 42
```

After training, the model with the best validation accuracy will be saved in the `models_*` directory as `model.pt`.

In our experiments, we trained each model with 3 different random seeds (42, 43, and 44). The models with the best validation accuracy are available at Zenodo:
[model_dt_s42.pt](https://zenodo.org/record/8330329/files/model_dt_s42.pt?download=1),
[model_dt_s43.pt](https://zenodo.org/record/8330329/files/model_dt_s43.pt?download=1),
[model_dt_s44.pt](https://zenodo.org/record/8330329/files/model_dt_s44.pt?download=1),
[model_t5_s42.pt](https://zenodo.org/record/8330329/files/model_t5_s42.pt?download=1),
[model_t5_s43.pt](https://zenodo.org/record/8330329/files/model_t5_s43.pt?download=1), and
[model_t5_s44.pt](https://zenodo.org/record/8330329/files/model_t5_s44.pt?download=1).

## Evaluating the Models

To get the accuracy of the models on the test set, use the following commands:

```bash
# Top-k accuracy for Freq
python run.py test-freq -d dataset_scipy_dt.pkl -v vocab_scipy_dt.pkl -beam k

# Top-k accuracy for DeepTyper-FS with beam size k
python run.py test -d dataset_scipy_dt.pkl -l model_dt_s42.pt -beam k

# Top-k accuracy for CodeT5-FT with beam size k
python run.py test-tf -d dataset_scipy_t5.pkl -l model_t5_s42.pt -beam k
```

To get the accuracy of the models on the benchmarks, use the following commands:

```bash
# Top-k accuracy for Freq
python run.py infer-freq -i ../benchmarks -v vocab_scipy_dt.pkl -k k

# Top-k accuracy for DeepTyper-FS
python run.py infer -i ../benchmarks -v vocab_scipy_dt.pkl -l model_dt_s42.pt -k k

# Top-k accuracy for CodeT5-FT
python run.py infer-tf -i ../benchmarks -l model_t5_s42.pt -k k

# Top-k accuracy for GPT-4
python run.py infer-gpt -i ../benchmarks -v vocab_scipy_dt.pkl -l gpt4_0.pkl -k k
```

Note that the `infer-gpt` command requires the parsed responses from GPT-4. See [../gpt4/README.md](../gpt4/README.md) for more details.
