# encoding: utf-8

import argparse
import json
import os.path
import dynet as dy
import numpy as np
from nltk.translate import bleu_score
import six
from time import time
import chainer
from chainer.dataset import convert
import preprocess
import net


class AIAYNAdamTrainer(object):
    def __init__(self, param_col, learning_rate=1.0, dim=512, warmup_steps=4000, beta_1=0.9, beta_2=0.98, eps=1e-9):
        self.optimizer = dy.AdamTrainer(param_col, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2, eps=eps)
        self.dim = dim
        self.warmup_steps = warmup_steps
        self.steps = 0

    def update(self):
        self.steps += 1
        decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
        self.optimizer.learning_rate = 1 * decay
        self.optimizer.update()


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0, bos_id=2):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = np

    # The paper did not mention eos
    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=bos_id)
    return x_block, y_in_block, y_out_block


def source_pad_concat_convert(x_seqs, device, eos_id=0, bos_id=2):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = np

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    x_block = xp.pad(x_block, ((0, 0), (1, 0)),
                     'constant', constant_values=bos_id)
    return x_block


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, device=-1, max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self):
        print('## Calculate BLEU')
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend([[t.tolist()] for t in targets])

            # sources = [chainer.dataset.to_device(self.device, x) for x in sources]
            sources = [x for x in sources]

            ys = [y.tolist() for y in self.model.translate(sources, self.max_length, beam=False)]
            # greedy generation for efficiency
            hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(references, hypotheses,
                                      smoothing_function=bleu_score.SmoothingFunction().method1) * 100
        print('BLEU:', bleu)


def main():
    parser = argparse.ArgumentParser(
        description='Dynet example: Attention is all you need')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--head', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--input', '-i', type=str, default='data',
                        help='Input directory')
    parser.add_argument('--source', '-s', type=str,
                        default='train.ja',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='train.en',
                        help='Filename of train data for target language')
    parser.add_argument('--source-valid', '-svalid', type=str,
                        default='dev.ja',
                        help='Filename of validation data for source language')
    parser.add_argument('--target-valid', '-tvalid', type=str,
                        default='dev.en',
                        help='Filename of validation data for target language')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--no-bleu', '-no-bleu', action='store_true',
                        help='Skip BLEU calculation')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--use-fixed-lr', action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')

    parser.add_argument('--dynet-gpus', default=0)
    parser.add_argument('--dynet-devices', default=0)
    parser.add_argument('--dynet-gpu', default=0)
    parser.add_argument('--dynet-autobatch', default=0)
    parser.add_argument('--dynet-mem', default=10000)

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))

    # Check file
    src_path = os.path.join(args.input, args.source)
    source_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(src_path, args.source_vocab)
    source_data = preprocess.make_dataset(src_path, source_vocab)

    tar_path = os.path.join(args.input, args.target)
    target_vocab = ['<eos>', '<unk>', '<bos>'] + preprocess.count_words(tar_path, args.target_vocab)
    target_data = preprocess.make_dataset(tar_path, target_vocab)

    assert len(source_data) == len(target_data)

    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t)
                  for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < 50 and 0 < len(t) < 50]
    print('Filtered training data size: %d' % len(train_data))

    src_path = os.path.join(args.input, args.source_valid)
    source_data = preprocess.make_dataset(src_path, source_vocab)
    tar_path = os.path.join(args.input, args.target_valid)
    target_data = preprocess.make_dataset(tar_path, target_vocab)
    assert len(source_data) == len(target_data)

    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    dy_model = dy.Model()

    # Define Model
    model = net.Transformer(dy_model,
                            args.layer,
                            min(len(source_ids), len(source_words)),
                            min(len(target_ids), len(target_words)),
                            args.unit,
                            h=args.head,
                            max_length=500,
                            use_label_smoothing=args.use_label_smoothing,
                            embed_position=args.embed_position)

    # Setup Optimizer
    # optimizer = dy.AdamTrainer(dy_model, alpha=0.001)
    optimizer = AIAYNAdamTrainer(dy_model)

    # Setup Trainer
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize, repeat=False, shuffle=False)

    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)

    num_steps = 0
    time_s = time()
    while train_iter.epoch < args.epoch:
        # dy.renew_cg()
        num_steps += 1
        # ---------- One iteration of the training loop ----------
        # Dynet Training Code:
        train_batch = train_iter.next()
        in_arrays = seq2seq_pad_concat_convert(train_batch, args.gpu)
        model.set_dropout(args.dropout)
        loss = model(*in_arrays)

        loss.backward()
        optimizer.update()

        if num_steps % 200 == 0:
            print("epoch: {:.03f}/{:02d}\tsteps: {}\ttrain_loss: {:.04f}\tlr: {:.05f}\t time: {:.01f} sec".format(train_iter.epoch_detail,
                                                                                             train_iter.epoch + 1,
                                                                                             num_steps,
                                                                                             loss.value(),
                                                                                             optimizer.optimizer.learning_rate,
                                                                                             time() - time_s))

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
            test_losses = []
            while True:
                # dy.renew_cg()
                test_batch = test_iter.next()
                in_arrays = seq2seq_pad_concat_convert(test_batch, args.gpu)

                # Forward the test data
                model.set_dropout(0.0)
                loss_test = model(*in_arrays)

                # Calculate the accuracy
                test_losses.append(loss_test.value())

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f}'.format(np.mean(test_losses)))

            CalculateBleu(model, test_data, 'val/main/bleu', device=args.gpu, batch=args.batchsize // 4)()

            ############################################################


if __name__ == '__main__':
    main()
