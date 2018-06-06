# attention-primer

Some toy experiments to illustrate the concept of attention in machine learning!

Mainly because it took me awhile to understand attention and I couldn't find a clear tutorial.

## References

[Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.](https://arxiv.org/abs/1706.03762)

[Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." *International Conference on Learning Representations*. 2015](https://arxiv.org/abs/1409.0473)

## Experiments \***WIP***


### 1 - Counting Letters \***WIP***

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
[[[' ']
  ['A']
  ['C']
  ['B']
  ['B']
  ['C']
  ['B']
  [' ']
  ['A']
  ['B']]]

Prediction: 
[[2 4 2]]

Output step 0 attended mainly to Input steps: [1 8]
Output step 1 attended mainly to Input steps: [3 4 6 9]
Output step 2 attended mainly to Input steps: [2 5]
```

For each output step, we see the learned attention being intuitively weighted on the relevant letters. In the above example, output step 0 counts the number of 'A's and attended mainly to input steps 1 and 8, which were the 'A's in the sequence.

Just as with a recurrent network, the trained model is able to take in variable sequence lengths, although performance definitely worsens when we deviate from the lengths used in the training set.

Refer to the task's README for more details.

