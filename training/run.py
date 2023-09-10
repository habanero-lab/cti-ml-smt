import argparse
from pathlib import Path
import numpy as np
import logging
import mlti.rnn
import mlti.tf


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)

    dataset_parser = subparsers.add_parser('dataset')
    dataset_parser.add_argument('--raw-dataset',
                                '-rd',
                                type=Path,
                                required=True)
    dataset_parser.add_argument('--bpe-max-merges',
                                '-bmm',
                                type=int,
                                default=10000)
    dataset_parser.add_argument('--bpe-min-freq', '-bmf', type=int, default=10)
    dataset_parser.add_argument('--nonname-min-freq',
                                '-nmf',
                                type=int,
                                default=10)
    dataset_parser.add_argument('--split-ratios',
                                '-sr',
                                type=float,
                                nargs=3,
                                default=[0.9, 0.05, 0.05])
    dataset_parser.add_argument('--vocab', '-v', type=Path, required=True)
    dataset_parser.add_argument('--dataset', '-d', type=Path, required=True)
    dataset_parser.add_argument('--seed', '-s', type=int, default=42)
    dataset_parser.add_argument('--num-processes',
                                '-np',
                                type=int,
                                default=None)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--vocab', '-v', type=Path, required=True)
    train_parser.add_argument('--dataset', '-d', type=Path, required=True)
    train_parser.add_argument('--save', '-s', type=Path, required=True)
    train_parser.add_argument('--model-dim', '-md', type=int, default=128)
    train_parser.add_argument('--num-layers', '-nl', type=int, default=3)
    train_parser.add_argument('--dropout', '-do', type=float, default=0.2)
    train_parser.add_argument('--batch-size', '-bs', type=int, default=32)
    train_parser.add_argument('--lr', '-lr', type=float, default=1e-3)
    train_parser.add_argument('--num-epochs', '-ne', type=int, default=50)
    train_parser.add_argument('--disable-gpu',
                              '-dg',
                              action='store_false',
                              dest='use_gpu')
    train_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    train_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)
    train_parser.add_argument('--deep-typer', '-dt', action='store_true')
    train_parser.add_argument('--seed', '-sd', type=int, default=42)

    test_parser = subparsers.add_parser('test')
    test_parser.add_argument('--dataset', '-d', type=Path, required=True)
    test_parser.add_argument('--load', '-l', type=Path, required=True)
    test_parser.add_argument('--batch-size', '-bs', type=int, default=32)
    test_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    test_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)
    test_parser.add_argument('--disable-gpu',
                             '-dg',
                             action='store_false',
                             dest='use_gpu')

    infer_parser = subparsers.add_parser('infer')
    infer_parser.add_argument('--input', '-i', type=Path, required=True)
    infer_parser.add_argument('--vocab', '-v', type=Path, required=True)
    infer_parser.add_argument('--load', '-l', type=Path, required=True)
    infer_parser.add_argument('--top-k', '-k', type=int, required=True)
    infer_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    infer_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)

    test_parser = subparsers.add_parser('test-freq')
    test_parser.add_argument('--dataset', '-d', type=Path, required=True)
    test_parser.add_argument('--vocab', '-v', type=Path, required=True)
    test_parser.add_argument('--beam-size', '-beam', type=int, default=32)

    infer_parser = subparsers.add_parser('infer-freq')
    infer_parser.add_argument('--input', '-i', type=Path, required=True)
    infer_parser.add_argument('--vocab', '-v', type=Path, required=True)
    infer_parser.add_argument('--top-k', '-k', type=int, required=True)

    infer_parser = subparsers.add_parser('infer-gpt')
    infer_parser.add_argument('--input', '-i', type=Path, required=True)
    infer_parser.add_argument('--vocab', '-v', type=Path, required=True)
    infer_parser.add_argument('--load', '-l', type=Path, required=True)
    infer_parser.add_argument('--top-k', '-k', type=int, required=True)

    dataset_parser = subparsers.add_parser('dataset-tf')
    dataset_parser.add_argument('--raw-dataset',
                                '-rd',
                                type=Path,
                                required=True)
    dataset_parser.add_argument('--encoder',
                                '-enc',
                                type=str,
                                default='Salesforce/codet5-large-ntp-py')
    dataset_parser.add_argument('--keep-hints-and-comments',
                                '-khc',
                                action='store_false',
                                dest='strip_hints_and_comments')
    dataset_parser.add_argument('--split-ratios',
                                '-sr',
                                type=float,
                                nargs=3,
                                default=[0.9, 0.05, 0.05])
    dataset_parser.add_argument('--dataset', '-d', type=Path, required=True)
    dataset_parser.add_argument('--seed', '-s', type=int, default=42)
    dataset_parser.add_argument('--num-processes',
                                '-np',
                                type=int,
                                default=None)

    train_parser = subparsers.add_parser('train-tf')
    train_parser.add_argument('--dataset', '-d', type=Path, required=True)
    train_parser.add_argument('--save', '-s', type=Path, required=True)
    train_parser.add_argument('--encoder',
                              '-enc',
                              type=str,
                              default='Salesforce/codet5-large-ntp-py')
    train_parser.add_argument('--decoder-dim', '-dd', type=int, default=128)
    train_parser.add_argument('--batch-size', '-bs', type=int, default=16)
    train_parser.add_argument('--subbatch-size', '-sbs', type=int, default=2)
    train_parser.add_argument('--infer-batch-size',
                              '-ibs', type=int, default=16)
    train_parser.add_argument('--lr', '-lr', type=float, default=5e-5)
    train_parser.add_argument('--num-epochs', '-ne', type=int, default=10)
    train_parser.add_argument('--lr-scheduler', '-lrs',
                              type=str, default='constant')
    train_parser.add_argument('--disable-gpu',
                              '-dg',
                              action='store_false',
                              dest='use_gpu')
    train_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    train_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)
    train_parser.add_argument('--seed', '-sd', type=int, default=42)

    test_parser = subparsers.add_parser('test-tf')
    test_parser.add_argument('--dataset', '-d', type=Path, required=True)
    test_parser.add_argument('--load', '-l', type=Path, required=True)
    test_parser.add_argument('--encoder',
                             '-enc',
                             type=str,
                             default='Salesforce/codet5-large-ntp-py')
    test_parser.add_argument('--batch-size', '-bs', type=int, default=16)
    test_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    test_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)
    test_parser.add_argument('--disable-gpu',
                             '-dg',
                             action='store_false',
                             dest='use_gpu')

    infer_parser = subparsers.add_parser('infer-tf')
    infer_parser.add_argument('--input', '-i', type=Path, required=True)
    infer_parser.add_argument('--load', '-l', type=Path, required=True)
    infer_parser.add_argument('--encoder',
                              '-enc',
                              type=str,
                              default='Salesforce/codet5-large-ntp-py')
    infer_parser.add_argument('--top-k', '-k', type=int, required=True)
    infer_parser.add_argument('--beam-size', '-beam', type=int, default=32)
    infer_parser.add_argument('--max-type-length', '-mtl', type=int, default=8)

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.command == 'dataset':
        mlti.rnn.preprocess.preprocess_dataset(
            args.raw_dataset,
            args.bpe_max_merges,
            args.bpe_min_freq,
            args.nonname_min_freq,
            args.split_ratios,
            args.vocab,
            args.dataset,
            np.random.SeedSequence(args.seed),
            args.num_processes
        )

    elif args.command == 'train':
        mlti.rnn.train.train(vocab_path=args.vocab,
                             dataset_path=args.dataset,
                             model_dir=args.save,
                             model_dim=args.model_dim,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             batch_size=args.batch_size,
                             lr=args.lr,
                             num_epochs=args.num_epochs,
                             use_gpu=args.use_gpu,
                             beam_size=args.beam_size,
                             max_type_length=args.max_type_length,
                             deep_typer=args.deep_typer,
                             seed=args.seed)

    elif args.command == 'test':
        mlti.rnn.train.test(dataset_path=args.dataset,
                            model_path=args.load,
                            batch_size=args.batch_size,
                            use_gpu=args.use_gpu,
                            beam_size=args.beam_size,
                            max_type_length=args.max_type_length)

    elif args.command == 'infer':
        input_path = args.input
        if input_path.is_dir():
            paths = list(input_path.glob('**/*.pydd'))
        else:
            paths = [input_path]
        mlti.rnn.infer.infer(paths, args.vocab, args.load, args.top_k,
                             args.beam_size, args.max_type_length)

    elif args.command == 'test-freq':
        mlti.rnn.train.test_freq(dataset_path=args.dataset,
                                 vocab_path=args.vocab,
                                 beam_size=args.beam_size)

    elif args.command == 'infer-freq':
        input_path = args.input
        if input_path.is_dir():
            paths = list(input_path.glob('**/*.pydd'))
        else:
            paths = [input_path]
        mlti.rnn.infer.infer_freq(paths, args.vocab, args.top_k)

    elif args.command == 'infer-gpt':
        input_path = args.input
        if input_path.is_dir():
            paths = list(input_path.glob('**/*.pydd'))
        else:
            paths = [input_path]
        mlti.rnn.infer.infer_gpt(paths, args.vocab, args.load, args.top_k)

    elif args.command == 'dataset-tf':
        mlti.tf.preprocess.preprocess_dataset(
            args.raw_dataset,
            args.encoder,
            args.strip_hints_and_comments,
            args.split_ratios,
            args.dataset,
            np.random.SeedSequence(args.seed),
            args.num_processes
        )

    elif args.command == 'train-tf':
        mlti.tf.train.train(
            dataset_path=args.dataset,
            model_dir=args.save,
            encoder_name=args.encoder,
            decoder_dim=args.decoder_dim,
            batch_size=args.batch_size,
            subbatch_size=args.subbatch_size,
            infer_batch_size=args.infer_batch_size,
            lr=args.lr,
            num_epochs=args.num_epochs,
            lr_scheduler_name=args.lr_scheduler,
            use_gpu=args.use_gpu,
            beam_size=args.beam_size,
            max_type_length=args.max_type_length,
            seed=args.seed)

    elif args.command == 'test-tf':
        mlti.tf.train.test(
            dataset_path=args.dataset,
            model_path=args.load,
            encoder_name=args.encoder,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
            beam_size=args.beam_size,
            max_type_length=args.max_type_length)

    elif args.command == 'infer-tf':
        input_path = args.input
        if input_path.is_dir():
            paths = list(input_path.glob('**/*.pydd'))
        else:
            paths = [input_path]
        mlti.tf.infer.infer(
            paths=paths,
            model_path=args.load,
            encoder_name=args.encoder,
            top_k=args.top_k,
            beam_size=args.beam_size,
            max_type_length=args.max_type_length)


if __name__ == '__main__':
    main()
