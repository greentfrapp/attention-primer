# 3 - Signal

## Preface

At the end of Task 1, we mentioned that the order of the input sequence does not matter, since we are just counting letters (see Task 1 Notes section).

Here, we consider a task where the order of the input sequence is paramount. Specifically, the first step of the input sequence acts as a signal to tell the model which letter to count.

The output is a sequence of a single step, which represents the count of the letter specified by the signal.

For instance, if the first step (signal) is 'A', the model should output the counts of 'A's, not including the signal. 

The model has to learn to recognize the signal purely by its position, since no other information is available ie. the signal for 'A' is identical to a regular 'A' in the sequence. This is similar to language processing, where the identical words can have vastly different meanings in different positions/contexts.

## Description

Consider a sequence, where each element is a randomly selected letter. We call the first element the *signal*. 

For example ('B' is the signal here):

```
Input:
['B'], ['B'], ['A'], ['B'], ['C']
Output:
[[2]]
```

The output is a sequence of one step, corresponding to the count of the letter specified by the signal (first input step).

In the above case, the first input step is 'B', which means that we have to count the number of 'B's in the following sequence. The output is hence 2, since there are 2 'B's in the sequence (not counting the signal).

In particular, all the 'B's, including the signal, are represented with the same vector when passed to the model. This means that the model must learn that the meaning of a 'B' changes, depending on whether it is the first letter or not.

Since the task here is slightly more complicated, I implement the entire encoder layer, with self-attention (but not multi-head) (see left half of Figure 1 and Section 3.1 in [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)). For decoding I just implement a simple scaled dot-product attention as per Task 1. There is no self-attention for the decoding.

There are two main models that can be trained with the script, with and without the `--pos_enc` flag, denoting the whether to use positional encodings for the input.

More on positional encodings in the Details section.

### Without `--pos_enc`

Without positional encodings, the model has no way to differentiate whether an input step is a signal or a regular step. As a result, the model simply does not learn well without positional encodings.

Sample of losses when training without `--pos_enc`:

```
Iteration 500 - Loss 1.3295859098434448
Iteration 1000 - Loss 1.227324366569519
Iteration 1500 - Loss 1.0068984031677246
Iteration 2000 - Loss 0.9906516075134277
```

The losses generally stop decreasing after reaching about 1. The decrease in loss can be attributed to the model learning the range of possible output values and possibly choosing to focus on counting a fixed letter, which makes its predictions more reliable than random. However, it clearly does not learn well.

### With `--pos_enc`

With positional encodings, the model is able to encode the positions of the input sequence into the input embeddings, which allows the model to recognize which is the first step in the sequence.

Sample of losses when training with `--pos_enc`:

```
Iteration 500 - Loss 1.3017479181289673
Iteration 1000 - Loss 0.115713931620121
Iteration 1500 - Loss 0.05086364597082138
Iteration 2000 - Loss 0.014654741622507572
```

With positional encodings, the model learns way better. By checking the attention weights in the Encoder-Decoder Attention layer, we can also see that the decoder pays extra attention to the first step of the sequence, which gives the signal.

Sample output from testing the model trained with `--pos_enc`:

```
Input: 
[[['B']
  ['C']
  ['A']
  ['C']
  ['A']
  ['B']
  ['B']
  ['A']
  ['A']
  ['C']
  ['B']]]

Prediction: 
[[3]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [0]
[0.41, 0.046, 0.063, 0.045, 0.063, 0.068, 0.067, 0.063, 0.063, 0.045, 0.067]

L2-Norm of Positional Encoding:
[0.816, 0.181, 0.183, 0.185, 0.186, 0.18, 0.185, 0.183, 0.185, 0.183, 0.178]
```

In the above output, despite Input Steps 0, 5, 6, 10 all being 'B's, the decoder pays extra attention (41%) to Input Step 0.

We can also check the L2-Norm for each vector in the positional encoding, where each vector represents the positional encoding for each step in the input sequence.

Here the L2-Norm of the positional vector is far larger for the first step (0.816) than for the rest of the input sequence (~0.18). This makes perfect sense, since the first step has to be modified to encode its status as the signal. On the other hand, the model should leave the rest of the sequence untouched since a 'B' in the second step should be no different from a 'B' in the last step. As such, the positional vectors for the other steps have a much smaller L2-norm and modifies the steps less. More on this in the Details.

*This experiment started out as an attempt to see if we can explicitly change the attention weights assigned to different letters by changing the signal ie. if the signal is 'B', the attention assigned to 'B's should be higher. However, this only seems to occur infrequently. Perhaps via self-attention, the sequence encodes the relevant information into the first step (signal) so the decoder simply focuses on the first step. Or it could be just spread out across all the steps so the decoder does not explicitly focus on the relevant letters in the input sequence. Maybe more on this in the future.*

### Adjusting `--enc_layers`

Using the script, we can also adjust the number of encoder layers, where each layer comprises self-attention, followed by a regular feed-forward network (see Figure 1 and Section 3.1 in Vaswani et. al (2017)).

The number of encoder layers allow each step in the sequence to incorporate and encode information from other steps. Figure 4 in Vaswani et. al (2017) shows a good example with the following sentence:

```
The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.
```

By inspecting the self-attention weights for the word `its`, Vaswani et. al (2017) found heavier attentions for the words `Law` and `application`, which are very contextually relevant.

In our case, the use of encoder layers allow the input sequence to modify itself based on its own signal. In particular, the decoder is restricted to a single scaled dot-product attention layer, which means that the encoder has to find a good representation for the sequence, so that the decoder can quickly retrieve the relevant information via a single attention layer.

In general, we find that with increasing `--enc_layers`, the decoder's attention on the first step (signal) tends to decrease, in the Encoder-Decoder Attention layer.

The previous sample output was trained with `--enc_layers=1`, where the attention weight on the signal was 32%.

Sample output from testing the model trained with `--pos_enc` and `--enc_layers=6`:

```
Input: 
[[['C']
  ['B']
  ['B']
  ['B']
  ['A']
  ['C']
  ['C']
  ['C']
  ['B']
  ['C']
  ['A']]]

Prediction: 
[[4]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [7]
[0.08, 0.076, 0.076, 0.076, 0.092, 0.108, 0.108, 0.108, 0.076, 0.108, 0.092]

L2-Norm of Positional Encoding:
[0.884, 0.115, 0.119, 0.114, 0.116, 0.12, 0.116, 0.122, 0.118, 0.119, 0.12]
```

In contrast to the previous output from `--enc_layers=1`, the attention weight on the signal here is far smaller at only 8%.

Very hypothetically, it could be that the use of more encoder layers allows the encoding to spread out the information across all the steps, making the decoder less reliant on a single step.

## Commands

### Training Without `--pos_enc`

```
$ python3 main.py --train
```

This trains a model without positional encodings and with default parameters:

- Training steps: `--steps=2000`
- Batchsize: `--batchsize=100`
- Learning rate: `--lr=1e-2`
- Savepath: `--savepath=models/`
- Encoding dimensions: `--hidden=64`
- Encoder layers: `--enc_layers=1`

The model will be trained on the Difference Task with default parameters:

- Max sequence length: `--max_len=10`
- Vocabulary size: `--vocab_size=3`

*Without `--pos_enc`, the model's loss typically does not fall below 0.9. Positional encodings are important for this task.*

### Training With `--pos_enc`

```
$ python3 main.py --train --pos_enc
```

This trains a model with positional encodings and with default parameters (see above).

The model will be trained on the Difference Task with default parameters (see above).

Vary the number of encoder layers used with the `--enc_layers` flag:

```
$ python3 main.py --train --pos_enc --enc_layers 6
```

The above command will train a model with positional encodings and 6 encoder layers. `--enc_layers=1` by default.

### Testing

```
$ python3 main.py --test
or
$ python3 main.py --test --pos_enc
```

This tests the trained model (remember to specify the `--pos_enc` flag if positional encodings were used during training and to specify the number of encoder layers via `--enc_layers` if not default).

### Help

```
$ python3 main.py --help
```

Run with the `--help` flag to get a list of possible flags.

## Details

Skip ahead to the **Model** section for details about self-attention.

### Input

This is similar to the inputs for Task 1, with two exceptions:

- An additional first step that serves as the signal, which makes the second dimension `max_len + 1`
- No null character so the last dimension is simply `vocab_size`

The input tensor has shape `(batchsize, max_len + 1, vocab_size)`.

### Output

The output just consists of sequences of length 1, representing the count of the signal-specified letter, which ranges from 0 to `max_len`.

The output tensor is of shape `(batchsize, 1, max_len + 1)`.

### Model

**Positional Encodings**

We touched on this briefly at the end of Task 1 in the Notes section, where we discussed how the **Queries** tensor is also a form of positional encoding for the output.

First, consider the embedding of the input:

```python3
encoding = tf.layers.dense(
	inputs=self.input,
	units=self.hidden,
	activation=None,
	name="encoding"
)
```

Since this is a simple position-wise feedforward network, an input vector of 'A' will always output the same encoding vector of `hidden` dimension, irregardless of the vector's position. (We use 'A' here as an example but the same logic applies to every other character.)

Likewise, if we pass this straight to the Encoder-Decoder Attention layer, the same attention weights will be assigned to every vector that represents 'A'. In that sense, the decoder has no way of differentiating a signal 'A' from a regular 'A'.

Will self-attention help?

Nope. Just as in the Encoder-Decoder Attention layer, the same attention weights will be assigned to every vector that represents 'A', no matter how many times you apply self-attention to the encodings.

Instead, we simply add positional encoding.

```python3
input_positional_encoding = tf.Variable(
	initial_value=np.zeros((1, self.max_len + 1, self.hidden)),
	trainable=True,
	dtype=tf.float32,
	name="input_positional_coding"
)

if self.pos_enc:
	# Add positional encodings
	encoding += input_positional_encoding
```

Here, we allow the model to learn the positional encoding by initializing it as a trainable `tf.Variable`, just as with the **Queries** tensor from Task 1. Then, we just add it to the encodings.

Notice that the shape of the positional encoding is `(1, max_len + 1, hidden)` ie. for each step in a length `max_len + 1` sequence, there is a vector of `hidden` dimension that represents information about the position. When we add this to the encoding, the addition is broadcasted to each sequence in the batch.

With this, an 'A' in one input step is different from an 'A' in another input step, since the positional vector at each input step is different. Furthermore, in this task, we know that the first step is significantly different as the signal, whereas the positions of the other steps are not very important. In fact, it is best if the positional encoding only changes the first step and leaves the rest of the sequence untouched, since position does not matter for the rest of the sequence. We see this clearly in the L2-norms of the positional vectors in the sample outputs from earlier (reproduced below), which are larger for the first step and much smaller for the rest of the steps.

From model trained with `--pos_enc` and `enc_layers=1`:

```
L2-Norm of Positional Encoding:
[0.816, 0.181, 0.183, 0.185, 0.186, 0.18, 0.185, 0.183, 0.185, 0.183, 0.178]
```

From model trained with `--pos_enc` and `enc_layers=6`:

```
L2-Norm of Positional Encoding:
[0.884, 0.115, 0.119, 0.114, 0.116, 0.12, 0.116, 0.122, 0.118, 0.119, 0.12]
```