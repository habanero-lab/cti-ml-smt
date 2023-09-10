# Zero-Shot Prompt-Based Type Inference With GPT-4

## Running the Inference Script

[infer.py](infer.py) is a script that generates prompts from the given input code and the prompt template in [prompt_template.txt](prompt_template.txt), and then uses GPT-4's API to generate the inferred types.

To use the script, an OpenAI API key is required. Run the script with the following command to infer types for all the benchmarks in the [benchmarks](../benchmarks) directory:

```bash
python infer.py -k API_KEY -i ../benchmarks -o ./out
```

By default, the script will generate 3 files for each benchmark to the `out` directory. Each file contains one sample of the inferred types for the benchmark.

Note that for the `ipnsw` benchmark, which is the longest one, the script may encounter a timeout error. To walk around this issue, run the following command to print the generated prompt:

```bash
python infer.py -i ../benchmarks/intrepydd/ipnsw/kernel.pydd --prompt-only
```

And then copy the generated prompt to [OpenAI Playground](https://platform.openai.com/playground) to get the response. Do not forget to set the model version, temperature, and max length appropriately.

The responses we obtained in our experiments can be found in the [responses](responses) directory.

## Parsing the Responses

To ensure the reproducibility of experiments using the predictions from GPT-4, we used the results in the [responses](responses) directory for all related experiments instead of calling GPT-4's API every time. The script [parse.py](parse.py) parses the responses and into a format that can be used by the compiler and other evaluation scripts. The script can be run with the following commands:

```bash
python parse.py -i ./responses -o ./gpt4_0.pkl -rd 0
python parse.py -i ./responses -o ./gpt4_1.pkl -rd 1
python parse.py -i ./responses -o ./gpt4_2.pkl -rd 2
```

Alternatively, the parsed responses can be downloaded from Zenodo:
[gpt4_0.pkl](https://zenodo.org/record/8330329/files/gpt4_0.pkl?download=1),
[gpt4_1.pkl](https://zenodo.org/record/8330329/files/gpt4_1.pkl?download=1),
[gpt4_2.pkl](https://zenodo.org/record/8330329/files/gpt4_2.pkl?download=1).

## Evaluating the Prediction Accuracy on the Benchmarks

See [../training/README.md#evaluating-the-models](../training/README.md#evaluating-the-models).
