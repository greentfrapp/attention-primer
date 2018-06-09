# attention-primer

Some toy experiments to illustrate the concept of attention in machine learning!

Mainly because it took me awhile to understand attention and I couldn't find a clear tutorial.

## References

[Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.](https://arxiv.org/abs/1706.03762)

[Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." *International Conference on Learning Representations*. 2015](https://arxiv.org/abs/1409.0473)

## Experiments \***WIP***


### [1 - Counting Letters](https://github.com/greentfrapp/attention-primer/tree/master/1_counting-letters)

**In this task, we demonstrate a simplified form of attention and apply it to a counting task.**

Consider a sequence, where each element is a randomly selected letter or null/blank. The task is to count how many times each letter appears in the sequence.

For example:

```
Input:
['A'], ['B'], [' '], ['B'], ['C']
Output:
[[1], [2], [1]]
```

The output is also a sequence, where each element corresponds to the count of a letter. In the above case, 'A', 'B' and 'C' appear 1, 2 and 1 times respectively.

We implement the Scaled Dot-Product Attention concept (described in Section 3.2.1 of [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)) for this task and train without the use of recurrent or convolutional networks.

Here is a sample output from the script.

```
Input: 
[[['A']
  ['A']
  ['C']
  ['C']
  ['B']
  [' ']
  ['C']
  ['C']
  ['C']
  ['B']]]

Prediction: 
[[2 2 5]]

Encoder-Decoder Attention: 
Output step 0 attended mainly to Input steps: [0 1]
[0.232, 0.232, 0.061, 0.061, 0.064, 0.101, 0.061, 0.061, 0.061, 0.064]
Output step 1 attended mainly to Input steps: [4 9]
[0.066, 0.066, 0.06, 0.06, 0.232, 0.101, 0.06, 0.06, 0.06, 0.232]
Output step 2 attended mainly to Input steps: [2 3 6 7 8]
[0.044, 0.044, 0.152, 0.152, 0.043, 0.066, 0.152, 0.152, 0.152, 0.043]
```

For each output step, we see the learned attention being intuitively weighted on the relevant letters. In the above example, Output Step 0 counts the number of 'A's and attended mainly to Input Steps 0 and 1, which were the 'A's in the sequence.

Refer to the [task](https://github.com/greentfrapp/attention-primer/tree/master/1_counting-letters) for more details.

### [2 - Difference](https://github.com/greentfrapp/attention-primer/tree/master/2_difference)

**In this task, we demonstrate self-attention and see how it is useful for modeling inter-token/intra-sequence dependency.**

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

There are two main models that can be trained with the script, with and without the `--self_att` flag, denoting the whether to use self-attention.

Refer to the [task](https://github.com/greentfrapp/attention-primer/tree/master/2_difference) for more details.
