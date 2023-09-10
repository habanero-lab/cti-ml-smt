import argparse
from pathlib import Path
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=Path, required=True)
    args = parser.parse_args()

    total = 0
    functions = {}
    num_f_has_array = 0
    num_f_has_dict = 0

    with open(args.input, 'rb') as f:
        while True:
            try:
                src, arg_types = pickle.load(f)
            except EOFError:
                break
            except Exception as e:
                print(repr(e))
                break

            total += 1
            functions[src] = functions.get(src, 0) + 1

            f_has_array = False
            f_has_dict = False
            for typ in arg_types.values():
                if has(typ, 'array'):
                    f_has_array = True
                if has(typ, 'dict'):
                    f_has_dict = True
            if f_has_array:
                num_f_has_array += 1
            if f_has_dict:
                num_f_has_dict += 1

    print(f'Num functions: {len(functions)} unique, {total} in total')
    print(f'Max occurrence: {max(functions.values())}, '
          f'min occurrence: {min(functions.values())}')
    print(f'Has array: {num_f_has_array}, has dict: {num_f_has_dict}')
    # fs = sorted(functions.items(), key=lambda x: -x[1])
    # print(fs[0][0])

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # sns.set_style("whitegrid")
    # # fig, ax = plt.subplots(figsize=(8, 6))
    # sns.boxplot(data=list(functions.values()), showfliers=False)
    # plt.savefig("dataset_dist.pdf", bbox_inches="tight")


def has(typ, type_name):
    if typ is None:
        return False
    if not isinstance(typ, tuple):
        return typ == type_name
    if typ[0] == type_name:
        return True
    for x in typ[1:]:
        if has(x, type_name):
            return True
    return False


if __name__ == '__main__':
    main()
