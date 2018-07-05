# 4 - Signal 2

## Preface

The output of Task 3 consists of sequences of length 1, which is somewhat trivial. Task 4 is an extension to Task 3, where there are three signals instead of one, which leads to an output sequence of length 3, where each step corresponds to one signal.

With the increased complexity of this task, we introduce multihead-attention, which will demonstrate a far larger capacity compared to attention with a single head.

## Description

Consider a sequence, where each element is a randomly selected letter. We call the first three elements the *signals*. 

For example ('C', 'B', 'B' are the signals here):

```
Input:
['C'], ['B'], ['B'], ['B'], ['A'], ['B'], ['C']
Output:
[[1], [2], [2]]
```

The output is a sequence of three steps, corresponding to the count of the letter specified by each signal (first three input steps).

In the above case, the first input step is 'C', which means that we have to count the number of 'C's in the following sequence. The output is hence 1, since there is 1 'C' in the sequence (not counting the signal).

Likewise, for the second and third signals. Also, note that signals can be repeated, like the repeated 'B' signals in the example above.

I implement the entire encoder layer, with multihead-self-attention (see left half of Figure 1 and Section 3.1 in [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)). For decoding I just implement a single multihead-attention layer. There is no self-attention for the decoding. In all attention layers, the number of heads used is `--heads=4` by default. The `--pos_enc` flag is enabled by default as well.

There are two main models that can be trained with the script, with and without `--multihead`. More precisely, since `--multihead=True` by default here, we can pass `--multihead=False` when running the script to disable `--multihead`.

More on multihead-attention in the Details section.

### With `--multihead=False`

Without multihead-attention, the model here is the same as the model used in Task 3. 

Sample of losses when training with `--multihead=False`:

```
Iteration 1000 - Loss 1.430114507675171
Iteration 2000 - Loss 1.1829146146774292
Iteration 3000 - Loss 1.075669288635254
Iteration 4000 - Loss 0.8190537095069885
```

The model seems to have insufficient capacity to learn well.

### With `--multihead=True`

With multihead-attention, despite using the same number of attention layers, the model has a much larger capacity and learns much faster.

Sample of losses when training with `--multihead=True`:

```
Iteration 1000 - Loss 0.6175369024276733
Iteration 2000 - Loss 0.04598541930317879
Iteration 3000 - Loss 0.026927191764116287
Iteration 4000 - Loss 0.0017624727915972471
```

Sample output from testing the model trained with `--multihead=True`:

```
Input:
[[['B']
  ['C']
  ['C']
  ['C']
  ['B']
  ['A']
  ['A']
  ['B']
  ['A']
  ['C']
  ['C']
  ['C']
  ['A']]]

Prediction:
[[2 4 4]]

Encoder-Decoder Attention:
Attention of Output step 0 on Input steps
    Head #0
    [0.09, 0.091, 0.076, 0.065, 0.065, 0.088, 0.089, 0.065, 0.088, 0.065, 0.064, 0.064, 0.088]
    Head #1
    [0.459, 0.088, 0.095, 0.034, 0.052, 0.032, 0.029, 0.051, 0.03, 0.033, 0.033, 0.034, 0.03]
    Head #2
    [0.128, 0.003, 0.044, 0.003, 0.244, 0.084, 0.077, 0.239, 0.084, 0.003, 0.003, 0.003, 0.084]
    Head #3
    [0.023, 0.002, 0.062, 0.064, 0.029, 0.156, 0.149, 0.029, 0.15, 0.062, 0.062, 0.065, 0.149]
Attention of Output step 1 on Input steps
    Head #0
    [0.097, 0.189, 0.018, 0.076, 0.085, 0.057, 0.057, 0.085, 0.057, 0.074, 0.075, 0.074, 0.056]
    Head #1
    [0.023, 0.757, 0.001, 0.039, 0.016, 0.009, 0.01, 0.016, 0.01, 0.037, 0.038, 0.036, 0.009]
    Head #2
    [0.107, 0.037, 0.078, 0.035, 0.13, 0.095, 0.093, 0.129, 0.095, 0.035, 0.035, 0.035, 0.095]
    Head #3
    [0.055, 0.024, 0.076, 0.077, 0.06, 0.106, 0.104, 0.059, 0.105, 0.077, 0.076, 0.078, 0.104]
Attention of Output step 2 on Input steps
    Head #0
    [0.006, 0.001, 0.695, 0.009, 0.004, 0.063, 0.063, 0.004, 0.061, 0.01, 0.009, 0.009, 0.066]
    Head #1
    [0.023, 0.0, 0.782, 0.006, 0.018, 0.035, 0.031, 0.018, 0.033, 0.007, 0.006, 0.007, 0.034]
    Head #2
    [0.117, 0.0, 0.023, 0.0, 0.316, 0.061, 0.053, 0.305, 0.061, 0.0, 0.0, 0.0, 0.061]
    Head #3
    [0.008, 0.0, 0.042, 0.044, 0.012, 0.198, 0.184, 0.011, 0.187, 0.042, 0.042, 0.046, 0.184]

L2-Norm of Input Positional Encoding:
[1.47, 1.564, 1.911, 0.295, 0.287, 0.292, 0.299, 0.286, 0.291, 0.293, 0.289, 0.291, 0.29]
```

The output shows three sets of attention, one for each step of the output sequence. For each step, there are also four subsets of attention, each corresponding to a head. Each head shares the same weights, so we can look across the output steps to see what each head tries to focus on.

For example, Head #1 seems to be focusing on the relevant input signal for each output step (query), since the Head #1 attention is 45.9% for Output Step 0 on Input Step 0 (Signal 1), 75.7% for Output Step 1 on Input Step 1 (Signal 2) and 78.2% for Output Step 2 on Input Step 2 (Signal 3).

Also, Head #3 seems to be primarily fixated on the 'A's in the sequence, irregardless of which output step serves as the query.

Of course, these trends will differ from training run to training run and are actually rather inconsistent between training runs.

Again, we can check the L2-Norm for each vector in the positional encoding and as per the trend in Task 3, we also see here that the L2-Norms of the positional vectors for the signals are far larger than for the rest of the sequence.

## Commands

### Training With `--multihead`

```
$ python3 main.py --train
```

This trains a model with multihead-attention (enabled by default) and with default parameters:

- Training steps: `--steps=2000`
- Batchsize: `--batchsize=100`
- Learning rate: `--lr=1e-2`
- Savepath: `--savepath=models/`
- Encoding dimensions: `--hidden=64`
- Encoder layers: `--enc_layers=1`
- Number of heads: `--heads=4`
- Positional Encoding: `--pos_enc=True`

The model will be trained on the Difference Task with default parameters:

- Max sequence length: `--max_len=10`
- Vocabulary size: `--vocab_size=3`

### Training With `--multihead=False`

```
$ python3 main.py --train --multihead=False
```

This trains a model without multihead-attention and with default parameters (see above).

The model will be trained on the Difference Task with default parameters (see above).

Vary the number of encoder layers used with the `--enc_layers` flag eg.

```
$ python3 main.py --train --enc_layers 6
```

The above command will train a model with positional encodings and 6 encoder layers. `--enc_layers=1` by default.

Vary the number of heads in multihead-attention with the `--heads` flag eg.

```
$ python3 main.py --train --heads 8
```

The above command will train a model with multihead-attention using 8 heads. `--heads=4` by default and number of heads will be the same for all attention layers (ie. encoder self-attention and encoder-decoder attention).

### Testing

```
$ python3 main.py --test
or
$ python3 main.py --test --multihead=False
```

This tests the trained model (remember to specify `--multihead=False` if multihead-attention was not used during training).

### Help

```
$ python3 main.py --help
```

Run with the `--help` flag to get a list of possible flags.

## Details

Skip ahead to the **Model** section for details about self-attention.

### Input

This is similar to the inputs for Task 1, with two exceptions:

- Additional first three step that serve as the signals, which makes the second dimension `max_len + 3`
- No null character so the last dimension is simply `vocab_size`

The input tensor has shape `(batchsize, max_len + 3, vocab_size)`.

### Output

The output just consists of sequences of length 3, representing the count of each signal-specified letter, which ranges from 0 to `max_len`.

The output tensor is of shape `(batchsize, 3, max_len + 1)`.

### Model

**Multihead-Attention**

In regular attention, we simply compare alignment of the **Queries** with the **Keys** to get a softmax-ed set of weights, which we use to calculate a weighted sum of the **Values**.

In multihead-attention, we first multiply the **Queries**, **Keys** and **Values** by respective weight tensors. Suppose the original tensors are all of shape `batchsize, length, hidden` and all consist of vectors of `hidden=64` dimensions and we want to implement multihead-attention with 4 heads. We then construct `4*3=12` weight tensors ie. 4 each for **Queries**, **Keys** and **Values**. Each weight tensor will be of shape `(hidden, hidden / 4)`, which means the outputs will consist of `64/4=16`-dimensional vectors. 

After multiplying the **Queries**, **Keys** and **Values** by their respective weight tensors (4 each), we will have 4 sets of **Queries**, **Keys** and **Values** tensors. The new tensors each consist of 16-dimensional vectors, as compared to the 64-dimensional vectors in the original tensors. 

We then proceed to do regular scaled dot-product attention with each of the 4 sets of **Queries**, **Keys** and **Values** tensors. We will then end up with 4 outputs, each of shape `(batchsize, length, 16)`. We concatenate these outputs along the last dimension, which gives us a large tensor of shape `(batchsize, length, 16*4=64)`. Notice that we get back `hidden` as the last dimension ie. the shape is also `(batchsize, length, hidden)`.

Finally we construct a output weight tensor of shape `(hidden, hidden)` and multiply the concatenated output with this weight tensor to obtain our final multihead output of shape `(batchsize, length, hidden)`.

The primary advantage of using multihead-attention is the additional complexity/capacity due to two main factors.

1. The additional linear mappings from multiplying by the learned weight tensors (notice the weight tensors are similar to feedforward networks but without bias and activation)
2. In each multihead-attention layer, instead of just a single attention, we get to perform and aggregate over `h` variations of attention

On a last note, while Vaswani et al. (2017) and this script defines `h` as a factor of `hidden`, it is actually possible to use any `h` and each weight tensor can be of shape `(hidden, d)` where `d` can be any size. The only consideration is that the output weight tensor has to be of shape `(h * d, hidden)` so that the final multihead output is still of shape `(batchsize, length, hidden)`.