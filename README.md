# attention-primer

Some toy experiments to illustrate the concept of attention in machine learning!

Mainly because it took me awhile to understand attention and I couldn't find a clear tutorial.

## References

[Vaswani, Ashish, et al. "Attention is all you need." *Advances in Neural Information Processing Systems*. 2017.](https://arxiv.org/abs/1706.03762)

[Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." *International Conference on Learning Representations*. 2015](https://arxiv.org/abs/1409.0473)

## Experiments \***WIP***


### 1 - Counting Letters

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
[[['B']
  ['A']
  ['B']
  [' ']
  ['B']
  ['C']
  ['A']
  ['B']
  [' ']
  [' ']]]

Prediction: 
[[2 4 1]]

Output step 0 attended mainly to Input steps: [1 6]
[ 0.05928046  0.21078663  0.05928046  0.093059    0.05928046  0.06212783
  0.21078663  0.05928046  0.093059    0.093059  ]
Output step 1 attended mainly to Input steps: [0 2 4 7]
[ 0.1613455   0.04682393  0.1613455   0.071399    0.1613455   0.04677311
  0.04682393  0.1613455   0.071399    0.071399  ]
Output step 2 attended mainly to Input steps: [5]
[ 0.06731109  0.07089685  0.06731109  0.1115749   0.06731109  0.25423717
  0.07089685  0.06731109  0.1115749   0.1115749 ]
```

For each output step, we see the learned attention being intuitively weighted on the relevant letters. In the above example, output step 0 counts the number of 'A's and attended mainly to input steps 1 and 6, which were the 'A's in the sequence.

Just as with a recurrent network, the trained model is able to take in variable sequence lengths, although performance definitely worsens when we deviate from the lengths used in the training set.

Refer to the [task](https://github.com/greentfrapp/attention-primer/tree/master/1_counting-letters) for more details.

