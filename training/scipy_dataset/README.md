# How to Create the SciPy Dataset

1. Clone the code repository of SciPy.

```bash
git clone https://github.com/scipy/scipy.git
```

2. Run the tests once to make sure everything works normally. This will also build the code.

```bash
cd scipy
python runtests.py -v
```

3. Collect test paths.

```bash
python runtests.py -- --collect-only -q > tests.txt
```

4. Extract the dataset.

```bash
python ../extract_dataset.py -i test.txt -o ../dataset_raw
```

5. Merge and deduplicate the dataset.

```bash
cd ..
python merge_dataset.py -i dataset_raw -o dataset_scipy_raw.pkl
```
