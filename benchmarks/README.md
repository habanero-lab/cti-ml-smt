# Benchmarks

This directory contains the 10 benchmarks used in the paper.

Each benchmark program consists of two parts:
* The Python main program  (`main.py`)
* The Intrepydd kernel code (`kernel.pydd`)

Some benchmarks also come with a script for preparing the input data (`prep.sh`). Make sure to run the script before starting the benchmark.

To run a benchmark, first compile the kernel code using the Intrepydd compiler (see the [intrepydd](../intrepydd) directory for more instructions). If the Intrepydd C++ backend is used for compilation, run the main program with the `KERNEL_TYPE` environment variable set to `pydd`:

```bash
KERNEL_TYPE=pydd python main.py
```

If the Numba-AOT backend is used for compilation, run the main program with the `KERNEL_TYPE` environment variable set to `numba`:

```bash
KERNEL_TYPE=numba python main.py
```
