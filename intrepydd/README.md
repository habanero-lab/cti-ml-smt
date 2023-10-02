# Intrepydd Compiler

This directory contains a modified version of the Intrepydd compiler introduced in [the Intrepydd paper](https://dl.acm.org/doi/10.1145/3426428.3426915). This modified version supports SMT-based and machine learning type inference, multi-version code generation, and an additional Numba-AOT backend.

The code relevant to the type inference approaches introduced in our work is mostly in the [compiler/smttypeinfer](compiler/smttypeinfer) directory.

## Usage

Add the `compiler` directory to the `PATH` environment variable:

```bash
export PATH=$PATH:$PWD/compiler
```

The compiler can then be invoked with the `pyddc` command. The following command compiles the `kernel.pydd` file in the current directory:

```bash
pyddc -verbose -smtti -smtti-use-ml -smtti-ml-model chatgpt -smtti-model /path/to/gpt4_1.pkl -smtti-ml-auto ./kernel.pydd
```

* The `-smtti` option enables SMT-based type inference.
* The `-smtti-use-ml` option enables machine learning type inference.
* The `-smtti-ml-model` option specifies the machine learning model to use (`chatgpt` corresponds to the GPT-4 model).
* The `-smtti-model` option specifies the path to the machine learning model file.
* The `-smtti-ml-auto` option enables the progressive relaxation of machine learning constraints.

Note that the modification to the Intrepydd compiler may have broken its original type inference and code generation functionalities.
So please make sure to use the `-smtti` option when invoking the compiler.

For more command line options, please see [compiler/glb.py](compiler/glb.py).
Options starting with `-smtti-` are related to the type inference approaches in this work.
