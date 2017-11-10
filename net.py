# encoding: utf-8

import numpy as np
import dynet as dy
import chainer.functions as F
from train import source_pad_concat_convert

MIN_VALUE = -10000


class TimeDistributed(object):
    def __call__(self, input):
        (model_dim, seq_len), batch_size = input.dim()
        total_words = seq_len * batch_size
        return dy.reshape(input, (model_dim,), batch_size=total_words)


class ReverseTimeDistributed(object):
    def __call__(self, input, seq_len, batch_size):
        (model_dim,), total_words = input.dim()
        assert (seq_len * batch_size == total_words)
        return dy.reshape(input, (model_dim, seq_len), batch_size=batch_size)


class Linear(object):
    def __init__(self, dy_model, input_dim, output_dim):
        self.W1 = dy_model.add_parameters((output_dim, input_dim))
        self.b1 = dy_model.add_parameters(output_dim)

    def __call__(self, input_expr, reconstruct_shape=True, timedistributed=False):
        W1 = dy.parameter(self.W1)
        b1 = dy.parameter(self.b1)

        if not timedistributed:
            input = TimeDistributed()(input_expr)
        else:
            input = input_expr

        output = dy.affine_transform([b1, W1, input])

        if not reconstruct_shape:
            return output
        (_, seq_len), batch_size = input_expr.dim()
        return ReverseTimeDistributed()(output, seq_len, batch_size)


class Linear_nobias(object):
    def __init__(self, dy_model, input_dim, output_dim):
        self.W1 = dy_model.add_parameters((output_dim, input_dim))
        self.output_dim = output_dim
        self.b1 = dy_model.add_parameters(output_dim)

    def __call__(self, input_expr):
        W1 = dy.parameter(self.W1)
        # b1 = dy.parameter(self.b1)
        b1 = dy.zeros(self.output_dim)
        (_, seq_len), batch_size = input_expr.dim()

        output = dy.affine_transform([b1, W1, input_expr])

        if seq_len == 1: # This is helpful when sequence length is 1 especially during decoding
            output = ReverseTimeDistributed()(output, seq_len, batch_size)

        return output


class LayerNorm(object):
    def __init__(self, dy_model, d_hid):
        self.p_g = dy_model.add_parameters(dim=d_hid, init=dy.ConstInitializer(1.0))
        self.p_b = dy_model.add_parameters(dim=d_hid, init=dy.ConstInitializer(0.0))

    def __call__(self, input_expr):
        # g = dy.parameter(self.p_g)
        # b = dy.parameter(self.p_b)
        #
        # (_, seq_len), batch_size = input_expr.dim()
        # input = TimeDistributed()(input_expr)
        # output = dy.layer_norm(input, g, b)
        # return ReverseTimeDistributed()(output, seq_len, batch_size)
        return input_expr


def sentence_block_embed(embed, x):
    """ Change implicitly embed_id function's target to ndim=2

    Apply embed_id for array of ndim 2,
    shape (batchsize, sentence_length),
    instead for array of ndim 1.

    """

    batch, length = x.shape
    # units, _ = embed.shape()
    _, units = embed.shape()  # According to updated Dynet

    y = np.copy(x)
    y[x < 0] = 0

    # Z = dy.zeros(units)
    e = dy.concatenate_cols([dy.zeros(units) if id_ == -1 else embed[id_] for id_ in x.reshape((batch * length,))])
    assert (e.dim() == ((units, batch * length), 1))

    # e = dy.lookup_batch(embed, y.reshape((batch * length,)))
    # assert (e.dim() == ((units,), batch * length))

    e = dy.reshape(e, (units, length), batch_size=batch)

    assert (e.dim() == ((units, length), batch))
    return e


def split_rows(X, h):
    (n_rows, _), batch = X.dim()
    l = range(n_rows)
    steps = n_rows // h
    output = []
    for i in range(0, n_rows, steps):
        # indexes = l[i:i + steps]
        # output.append(dy.select_rows(X, indexes))
        output.append(dy.pickrange(X, i, i + steps))
    return output


def split_batch(X, h):
    (n_rows, _), batch = X.dim()
    l = range(batch)
    steps = batch // h
    output = []
    for i in range(0, batch, steps):
        indexes = l[i:i + steps]
        output.append(dy.pick_batch_elems(X, indexes))
    return output


class MultiHeadAttention():
    """ Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, dy_model, n_units, h=8, self_attention=True):
        # TODO: keep bias = False
        # self.W_Q = Linear(dy_model, n_units, n_units)
        # self.W_K = Linear(dy_model, n_units, n_units)
        # self.W_V = Linear(dy_model, n_units, n_units)

        if self_attention:
            self.W_QKV = Linear_nobias(dy_model, n_units, n_units * 3)
        else:
            self.W_Q = Linear_nobias(dy_model, n_units, n_units)
            self.W_KV = Linear_nobias(dy_model, n_units, n_units * 2)

        self.finishing_linear_layer = Linear_nobias(dy_model, n_units, n_units)

        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        # self.dropout = dropout
        self.is_self_attention = self_attention

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, x, z=None, mask=None):
        h = self.h

        if self.is_self_attention:
            Q, K, V = split_rows(self.W_QKV(x), 3)
            # Q = self.W_Q(x)
            # K = self.W_K(x)
            # V = self.W_V(x)
        else:
            Q = self.W_Q(x)
            K, V = split_rows(self.W_KV(z), 2)
            # Q = self.W_Q(x)
            # K = self.W_K(z)
            # V = self.W_V(z)

        (n_units, n_querys), batch = Q.dim()
        (_, n_keys), _ = K.dim()

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency

        batch_Q = dy.concatenate_to_batch(split_rows(Q, h))
        batch_K = dy.concatenate_to_batch(split_rows(K, h))
        batch_V = dy.concatenate_to_batch(split_rows(V, h))

        assert(batch_Q.dim() == (n_units // h, n_querys), batch * h)
        assert(batch_K.dim() == (n_units // h, n_keys), batch * h)
        assert(batch_V.dim() == (n_units // h, n_keys), batch * h)

        mask = np.concatenate([mask] * h, axis=0)
        # mask = dy.inputTensor(np.moveaxis(mask, [0, 1, 2], [2, 1, 0]), batched=True)
        mask = dy.inputTensor(np.moveaxis(mask, [1, 0, 2], [0, 2, 1]), batched=True)

        batch_A = (dy.transpose(batch_Q) * batch_K) * self.scale_score
        batch_A = dy.cmult(batch_A, mask) + (1 - mask)*MIN_VALUE

        sent_len = batch_A.dim()[0][0]
        if sent_len == 1:
            batch_A = dy.softmax(batch_A)
        else:
            # batch_A = dy.transpose(dy.softmax(dy.transpose(batch_A)))
            batch_A = dy.softmax(batch_A, d=1)

        # TODO: Check whether this is correct after masking
        batch_A = dy.cmult(batch_A, mask)
        assert (batch_A.dim() == ((n_querys, n_keys), batch * h))

        # TODO: Check if attention dropout needs to be applied here
        batch_C = dy.transpose(batch_A * dy.transpose(batch_V))

        # batch_C = batch_V * dy.transpose(batch_A)  # TODO: Check the correctness of this step
        assert (batch_C.dim() == ((n_units // h, n_querys), batch * h))

        C = dy.concatenate(split_batch(batch_C, h), d=0)
        assert (C.dim() == ((n_units, n_querys), batch))
        C = self.finishing_linear_layer(C, reconstruct_shape=False, timedistributed=True)
        return C


class FeedForwardLayer():
    def __init__(self, dy_model, n_units):
        n_inner_units = n_units * 4
        self.W_1 = Linear(dy_model, n_units, n_inner_units)
        self.W_2 = Linear(dy_model, n_inner_units, n_units)

        # TODO: Put Leaky Relu here
        self.act = dy.rectify

    def __call__(self, e):
        e = self.W_1(e, reconstruct_shape=False, timedistributed=True)
        e = self.act(e)
        e = self.W_2(e, reconstruct_shape=False, timedistributed=True)
        return e


class EncoderLayer():
    def __init__(self, dy_model, n_units, h=8):
        self.self_attention = MultiHeadAttention(dy_model, n_units, h)
        self.feed_forward = FeedForwardLayer(dy_model, n_units)
        self.ln_1 = LayerNorm(dy_model, n_units)
        self.ln_2 = LayerNorm(dy_model, n_units)

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        self.self_attention.set_dropout(self.dropout)
        sub = self.self_attention(e, e, xx_mask)
        e = e + dy.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + dy.dropout(sub, self.dropout)
        e = self.ln_2(e)

        return e


class DecoderLayer():
    def __init__(self, dy_model, n_units, h=8):
        self.self_attention = MultiHeadAttention(dy_model, n_units, h)
        self.source_attention = MultiHeadAttention(dy_model, n_units, h, self_attention=False)
        self.feed_forward = FeedForwardLayer(dy_model, n_units)
        self.ln_1 = LayerNorm(dy_model, n_units)
        self.ln_2 = LayerNorm(dy_model, n_units)
        self.ln_3 = LayerNorm(dy_model, n_units)

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, e, s, xy_mask, yy_mask):
        self.self_attention.set_dropout(self.dropout)
        sub = self.self_attention(e, e, yy_mask)
        e = e + dy.dropout(sub, self.dropout)
        e = self.ln_1(e)

        self.source_attention.set_dropout(self.dropout)
        sub = self.source_attention(e, s, xy_mask)
        e = e + dy.dropout(sub, self.dropout)
        e = self.ln_2(e)

        sub = self.feed_forward(e)
        e = e + dy.dropout(sub, self.dropout)
        e = self.ln_3(e)
        return e


class Encoder():
    def __init__(self, dy_model, n_layers, n_units, h=8):
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(dy_model, n_units, h)
            self.layer_names.append((name, layer))

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        for name, layer in self.layer_names:
            layer.set_dropout(self.dropout)
            e = layer(e, xx_mask)
        return e


class Decoder():
    def __init__(self, dy_model, n_layers, n_units, h=8):
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = DecoderLayer(dy_model, n_units, h)
            self.layer_names.append((name, layer))

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, e, source, xy_mask, yy_mask):
        for name, layer in self.layer_names:
            layer.set_dropout(self.dropout)
            e = layer(e, source, xy_mask, yy_mask)
        return e


class Transformer(object):
    def __init__(self, dy_model, n_layers, n_source_vocab, n_target_vocab, n_units,
                 h=8, max_length=500,
                 use_label_smoothing=False,
                 embed_position=False):

        self.embed_x = dy_model.add_lookup_parameters((n_source_vocab, n_units))
        self.embed_y = dy_model.add_lookup_parameters((n_target_vocab, n_units))

        self.encoder = Encoder(dy_model, n_layers, n_units, h)
        self.decoder = Decoder(dy_model, n_layers, n_units, h)

        # TODO: Implement the feature of position embedding

        self.output_affine = Linear(dy_model, n_units, n_target_vocab)
        self.n_layers = n_layers
        self.xp = np
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.use_label_smoothing = use_label_smoothing
        self.initialize_position_encoding(max_length, n_units)
        self.scale_emb = self.n_units ** 0.5

    def initialize_position_encoding(self, length, n_units):
        xp = self.xp
        # Implementation in the Google tensor2tensor repo
        channels = n_units
        position = xp.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (xp.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * xp.exp(xp.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = xp.expand_dims(position, 1) * xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate([xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, channels])
        self.position_encoding_block = xp.transpose(signal, (0, 2, 1))

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += dy.inputTensor(self.position_encoding_block[0, :, :length])
        # TODO: If position embedding, incorporate it here.
        emb_block = dy.dropout(emb_block, self.dropout)
        return emb_block

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 0) * \
               (source_block[:, :, None] >= 0)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = self.xp.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[None,]
        history_mask = self.xp.broadcast_to(
            history_mask, (batch, length, length))
        return history_mask

    def output(self, h_block):
        concat_logit_block = self.output_affine(h_block, reconstruct_shape=False, timedistributed=True)
        return concat_logit_block

    def output_and_loss(self, h_block, t_block):
        (units, length), batch = h_block.dim()

        # Output (all together at once for efficiency)
        concat_logit_block = self.output_affine(h_block, reconstruct_shape=False)
        (_,), rebatch = concat_logit_block.dim()

        concat_t_block = t_block.reshape((rebatch))
        ignore_mask = (concat_t_block >= 0)
        n_token = ignore_mask.sum()
        normalizer = n_token  # n_token or batch or 1

        if not self.use_label_smoothing:
            bool_array = concat_t_block != -1
            indexes = np.argwhere(bool_array).ravel()
            concat_logit_block = dy.pick_batch_elems(concat_logit_block, indexes)
            concat_t_block = concat_t_block[bool_array]

            loss = dy.pickneglogsoftmax_batch(concat_logit_block, concat_t_block)
            loss = dy.mean_batches(loss)
        else:
            bool_array = concat_t_block != -1
            indexes = np.argwhere(bool_array).ravel()

            concat_logit_block_ls = dy.pick_batch_elems(concat_logit_block, indexes)
            concat_t_block_ls = concat_t_block[bool_array]

            log_prob = dy.log_softmax(concat_logit_block_ls)
            pre_loss = dy.pick_batch(log_prob, concat_t_block_ls)
            loss = - dy.mean_batches(pre_loss)

        # TODO: Can compute metrics like accuracy here

        if self.use_label_smoothing:
            label_smoothing = -1 * dy.mean_elems(log_prob)
            label_smoothing = dy.mean_batches(label_smoothing)
            loss = 0.9 * loss + 0.1 * label_smoothing

        return loss

    def set_dropout(self, dropout):
        self.dropout = dropout

    def __call__(self, x_block, y_in_block, y_out_block, get_prediction=False):
        dy.renew_cg()
        # print(self.dropout)
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        # Make Embedding
        ex_block = self.make_input_embedding(self.embed_x, x_block)
        ey_block = self.make_input_embedding(self.embed_y, y_in_block)

        # Make Masks
        xx_mask = self.make_attention_mask(x_block, x_block)
        xy_mask = self.make_attention_mask(y_in_block, x_block)
        yy_mask = self.make_attention_mask(y_in_block, y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Encode Sources
        self.encoder.set_dropout(self.dropout)
        z_blocks = self.encoder(ex_block, xx_mask)
        # [(batch, n_units, x_length), ...]

        # Encode Targets with Sources (Decode without Output)
        self.decoder.set_dropout(self.dropout)
        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)
        # (batch, n_units, y_length)

        if get_prediction:
            y_len = h_block.dim()[0][1]
            last_col = dy.pick(h_block, dim=1, index=y_len-1)
            return self.output(last_col)
        else:
            return self.output_and_loss(h_block, y_out_block)

    def translate(self, x_block, max_length=50, beam=5):
        # if beam:
        #     return self.translate_beam(x_block, max_length, beam)

        # TODO: efficient inference by re-using result
        x_block = source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        # y_block = self.xp.zeros((batch, 1), dtype=x_block.dtype)
        y_block = self.xp.full((batch, 1), 2, dtype=x_block.dtype)  # bos
        eos_flags = self.xp.zeros((batch,), dtype=x_block.dtype)
        result = []
        for i in range(max_length):
            # print(i)
            self.set_dropout(0.0)
            log_prob_tail = self(x_block, y_block, y_block, get_prediction=True)
            ys = self.xp.argmax(log_prob_tail.npvalue(), axis=0).astype('i')
            result.append(ys)
            y_block = F.concat([y_block, ys[:, None]], axis=1).data
            eos_flags += (ys == 0)
            if self.xp.all(eos_flags):
                break

        result = self.xp.stack(result).T

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y)
        return outs


#     def translate_beam(self, x_block, max_length=50, beam=5):
#         # TODO: efficient inference by re-using result
#         # TODO: batch processing
#         with chainer.no_backprop_mode():
#             with chainer.using_config('train', False):
#                 x_block = source_pad_concat_convert(
#                     x_block, device=None)
#                 batch, x_length = x_block.shape
#                 assert batch == 1, 'Batch processing is not supported now.'
#                 y_block = self.xp.full(
#                     (batch, 1), 2, dtype=x_block.dtype)  # bos
#                 eos_flags = self.xp.zeros(
#                     (batch * beam,), dtype=x_block.dtype)
#                 sum_scores = self.xp.zeros(1, 'f')
#                 result = [[2]] * batch * beam
#                 for i in range(max_length):
#                     log_prob_tail = self(x_block, y_block, y_block,
#                                          get_prediction=True)
#
#                     ys_list, ws_list = get_topk(
#                         log_prob_tail.data, beam, axis=1)
#                     ys_concat = self.xp.concatenate(ys_list, axis=0)
#                     sum_ws_list = [ws + sum_scores for ws in ws_list]
#                     sum_ws_concat = self.xp.concatenate(sum_ws_list, axis=0)
#
#                     # Get top-k from total candidates
#                     idx_list, sum_w_list = get_topk(
#                         sum_ws_concat, beam, axis=0)
#                     idx_concat = self.xp.stack(idx_list, axis=0)
#                     ys = ys_concat[idx_concat]
#                     sum_scores = self.xp.stack(sum_w_list, axis=0)
#
#                     if i != 0:
#                         old_idx_list = (idx_concat % beam).tolist()
#                     else:
#                         old_idx_list = [0] * beam
#
#                     result = [result[idx] + [y]
#                               for idx, y in zip(old_idx_list, ys.tolist())]
#
#                     y_block = self.xp.array(result).astype('i')
#                     if x_block.shape[0] != y_block.shape[0]:
#                         x_block = self.xp.broadcast_to(
#                             x_block, (y_block.shape[0], x_block.shape[1]))
#                     eos_flags += (ys == 0)
#                     if self.xp.all(eos_flags):
#                         break
#
#         outs = [[wi for wi in sent if wi not in [2, 0]] for sent in result]
#         outs = [sent if sent else [0] for sent in outs]
#         return outs
#
#
# def get_topk(x, k=5, axis=1):
#     ids_list = []
#     scores_list = []
#     xp = cuda.get_array_module(x)
#     for i in range(k):
#         ids = xp.argmax(x, axis=axis).astype('i')
#         if axis == 0:
#             scores = x[ids]
#             x[ids] = - float('inf')
#         else:
#             scores = x[xp.arange(ids.shape[0]), ids]
#             x[xp.arange(ids.shape[0]), ids] = - float('inf')
#         ids_list.append(ids)
#         scores_list.append(scores)
#     return ids_list, scores_list
