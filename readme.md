Requirements:
1. dynet >= 2.0
2. Chainer (for data batching)
3. progressbar 2.0
4. nltk for computing BLEU score


Implementaton of `Attention is all you need paper: (Transformer Model)`

features added:
1) Multi-Head Attention
2) Positional Encoding
3) Positional Embedding
4) Label Smoothing
5) Warm-up steps training of Adam Optimizer
6) Shared weights of target embedding and decoder softmax layer


Steps to run the model:
python train.py -s train-big.ja -t train-big.en --dynet-gpu --epoch 30 -b 128 --head 1

It reaches a maximum BLEU Score of around `25.2`. Also the current training speed is around 0.87 seconds per optimization step for batch size of 128. Overall 1 epoch takes ~ 10 minutes on TITAN X (Pascal) GPU.

Issues / Need for improvement:
i) Also, currently, `Layernorm is not working properly`. So that part of the code is commented out.
ii) If we keep mult-heads as 8, then the training speed decreases by a factor of 3. I am guessing, this is due to "dynet.pick_batch_elems()" function. If this can be converted to something like, dynet.pick_batch_range(), as is currently for the rows selection, then the code will speed up.




