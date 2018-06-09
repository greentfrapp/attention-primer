# 2 - Difference

## Preface

In Task 1, we look at how attention can be used to count letters in a sequence. In that task, the output steps were generally independent of each other. Compare this to language translation, where the N-th word in the output translation is generally considered to be dependent on the previously translated N-1 words, as well as the input sentence.

In this task, we try to demonstrate two things:

1. Show that inter-token dependence in the output can actually be reframed as input dependence ie. explicitly modelling inter-token dependence in the output seems to be unnecessary
2. Despite that, show that self-attention can be used to explicitly model inter-token dependence

This Task is pretty similar to Task 1, with the exception of an additional step in the output that computes the absolute difference between the first and second steps. In that sense, this additional step is dependent on the first and second steps in the output sequence.

## Description

Consider a sequence, where each element is a randomly selected letter or null/blank. The task has two parts: 

1. Count how many times each letter appears in the sequence
2. Calculate the absolute difference between the first two counts

For example:

```
Input:
['A'], ['B'], [' '], ['B'], ['C']
Output:
[[1], [2], [1], [1]]
```

The output is also a sequence, where each element corresponds to the count of a letter and the last element corresponds to the difference between the first two elements. In the above case, 'A', 'B' and 'C' appear 1, 2 and 1 times respectively, while counts for 'A' and 'B' differ by 1.

As per Task 1, I implement the Scaled Dot-Product Attention (described in Section 3.2.1 of [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)) for this task. I also demonstrate a simple case of self-attention in this task.

There are two main models that can be trained with the script, with and without the `--self_att` flag, denoting whether to use self-attention.

### Without `--self_att`

Here is a sample output from the script without `--self_att` ie. similar to Task 1 but with an additional output step.

```
Input: 
[[[' ']
  [' ']
  [' ']
  ['A']
  ['B']
  [' ']
  ['C']
  ['C']
  [' ']
  ['A']]]

Prediction: 
[[2 1 2 1]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [3 9]
[0.092, 0.092, 0.092, 0.186, 0.03, 0.092, 0.068, 0.068, 0.092, 0.186]
Output step 1 attended mainly to Input steps: [4]
[0.107, 0.107, 0.107, 0.036, 0.238, 0.107, 0.077, 0.077, 0.107, 0.036]
Output step 2 attended mainly to Input steps: [6 7]
[0.071, 0.071, 0.071, 0.05, 0.043, 0.071, 0.25, 0.25, 0.071, 0.05]
Output step 3 attended mainly to Input steps: [4]
[0.092, 0.092, 0.092, 0.135, 0.139, 0.092, 0.065, 0.065, 0.092, 0.135]
```

As per Task 1, we see the output steps attending to their respective letters in the input sequence eg. Output Step 0 attending to the positions of 'A's at Input Steps 3 and 9. 

More interestingly, we see that Output Step 3 actually attends to positions of both 'A' and 'B'. Looking at the attention of Output Step 3, the maximum attention weight of 0.139 is assigned to Input Step 4, which holds 'B'. But attention weights of 0.135 are also assigned to Input Steps 3 and 9, which hold 'A's.

In other words, Output Step 3 can be modeled as being dependent on Output Steps 1 and 2. But it can also be modeled as simply being dependent on the inputs of Output Steps 1 and 2. With this simple example, we can see that although there is superficial inter-token dependence in the output, this can be reframed as input dependence. 

However, we do concede that explicitly modeling inter-token dependence in the output can have its advantages, such as leading to simpler and more interpretable models, which brings us to the model with `--self_att`.

### With `--self_att`

Here is a sample output from the script with `--self_att` enabled ie. we allow the output to attend to and modify itself, based on itself. More details later.

With the `--self_att` flag enabled, the script prints  the self-attention weights ie. how much attention each output step gives to every step in the output sequence (including itself).

```
Input: 
[[['B']
  ['B']
  ['C']
  [' ']
  ['A']
  ['C']
  ['A']
  ['A']
  ['C']
  ['C']]]

Prediction: 
[[3 2 4 1]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [4 6 7]
[0.05, 0.05, 0.061, 0.075, 0.193, 0.061, 0.193, 0.193, 0.061, 0.061]
Output step 1 attended mainly to Input steps: [0 1]
[0.219, 0.219, 0.073, 0.089, 0.06, 0.073, 0.06, 0.06, 0.073, 0.073]
Output step 2 attended mainly to Input steps: [8 9]
[0.041, 0.041, 0.174, 0.093, 0.042, 0.174, 0.042, 0.042, 0.174, 0.174]
Output step 3 attended mainly to Input steps: [3]
[0.088, 0.088, 0.109, 0.116, 0.091, 0.109, 0.091, 0.091, 0.109, 0.109]

Self-Attention: 
Attention of Output step 0:
[1.0, 0.0, 0.0, 0.0]
Attention of Output step 1:
[0.0, 1.0, 0.0, 0.0]
Attention of Output step 2:
[0.009, 0.006, 0.965, 0.021]
Attention of Output step 3:
[0.696, 0.06, 0.038, 0.205]
```

First, with `--self_att`, the Encoder-Decoder Attention weights for Output Step 3 is now much more evenly divided across all the input steps. Since we now allow the modelling of inter-token dependence in the output via self-attention, Output Step 3 can gather more information later from Output Steps 0 and 1.

In the Self-Attention weights, notice that Output Steps 0, 1 and 2 are generally narcissistic and pay nearly 100% attention to themselves. This is because they are largely independent, just like the output in Task 1.

However, Output Step 3 pays far less attention to itself (20%) and instead pays a lot of attention on Output Step 0 (70%) and a tiny bit of attention to Output Step 1 (6%). Very intuitive, given that Output Step 3 is supposed to show the difference between Output Steps 0 and 1.

By allowing the output sequence to self-attend, we enable the modelling of inter-token dependencies. 

## Commands

### Training Without `--self_att`

```
$ python3 main.py --train
```

This trains a model without self-attention and with default parameters:

- Training steps: `--steps=1000`
- Batchsize: `--batchsize=100`
- Learning rate: `--lr=1e-2`
- Savepath: `--savepath=models/`
- Encoding dimensions: `--hidden=64`

The model will be trained on the Difference Task with default parameters:

- Max sequence length: `--max_len=10`
- Vocabulary size: `--vocab_size=3`

### Training With `--self_att`

```
$ python3 main.py --train --self_att
```

This trains a model with self-attention and with default parameters (see above).

*With `--self_att` usually trains much faster than without ie. achieves smaller loss with the same number of training steps.*

The model will be trained on the Difference Task with default parameters (see above).

### Testing

```
$ python3 main.py --test
or
$ python3 main.py --test --self_att
```

This tests the trained model (remember to specify the `--self_att` flag if self-attention was used during training).

See Task 1 regarding testing with different sequence lengths.

### Help

```
$ python3 main.py --help
```

Run with the `--help` flag to get a list of possible flags.

## Details

Skip ahead to the **Model** section for details about self-attention.

### Input

This is exactly like the inputs for Task 1.

### Output

This is similar to the outputs for Task 1, with the exception of an additional step that computes the absolute difference between the first two steps.

Hence the output tensor is of shape `(batchsize, vocab_size + 1, max_len + 1)`.

### Model

Here we will focus on explaining the self-attention part of the model. For the general attention mechanism, refer to Task 1.

**Self-Attention**

This is an extract of the code that is run when `--self_att` is enabled.

```python3
decoding, self.self_attention_weights = self.attention(
	query=decoding,
	key=decoding,
	value=decoding,
)
```

With the exception of the addition above, the model is exactly the same as in Task 1.

Actually, to be more accurate, the self-attention here could be described as attending to the **previous** `decoding`.

Here is one way to think about it. Let's explicitly term pre-self-attention `decoding` as `decoding_pre` and post-self-attention `decoding` as `decoding_post`. We can rewrite the above code snippet as: 

```python3
decoding_post, self.self_attention_weights = self.attention(
	query=decoding_pre,
	key=decoding_pre,
	value=decoding_pre,
)
```

`decoding_pre` combines the original **Query** with a weighted sum of the inputs (see Task 1). Step 0 in `decoding_pre` focuses on the 'A's in the input sequence, Step 1 in `decoding_pre` focuses on the 'B's in the input sequence and so on. The last step, which should be computing the difference of Output Steps 0 and 1, may just look at the entire input sequence instead of focusing on particular steps.

Then with self-attention, the last step in `decoding_post` can focus on Steps 0 and 1 in `decoding_pre`, to get the difference between the two steps. The other steps in `decoding_post` can focus on themselves in `decoding_pre` ie. copy their own previous values, since they already have enough information to generate the correct output.

This is shown clearly in the printed self-attention weights when you test the model trained with `--self_att`:

```
Self-Attention: 
Attention of Output step 0:
[1.0, 0.0, 0.0, 0.0]
Attention of Output step 1:
[0.0, 1.0, 0.0, 0.0]
Attention of Output step 2:
[0.009, 0.006, 0.965, 0.021]
Attention of Output step 3:
[0.696, 0.06, 0.038, 0.205]
```

Output Step 3 distributes attention to other steps,  while the other steps all focus on themselves.

