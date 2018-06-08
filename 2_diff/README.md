# 1 - Counting Letters \***WIP***

## Description

Consider a sequence, where each element is a randomly selected letter or null/blank. The task is to count how many times each letter appears in the sequence.

For example:

```
Input:
['A'], ['B'], [' '], ['B'], ['C']
Output:
[[1], [2], [1]]
```

The output is also a sequence, where each element corresponds to the count of a letter. In the above case, 'A', 'B' and 'C' appear 1, 2 and 1 times respectively.

I implement the Scaled Dot-Product Attention (described in Section 3.2.1 of [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)) for this task and train without the use of recurrent or convolutional networks.

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

## Details

### Input

The actual input to the model consists of sequences of one-hot embeddings, which means the input tensor has shape `(batchsize, max_len, vocab_size + 1)`. 

*It is `vocab_size + 1` instead of `vocab_size`, because we need to add a one-hot dimension for the null character as well.*

For example, if we set `vocab_size=2`, the possible characters will be 'A', 'B' and ' ' (null). Consider a batch of `batchsize=1` sequence of length `max_len=4`: 

```
[['A', 'B', ' ', 'A']]
```

The embedding for that will be a tensor of shape (1, 4, 3):

```
[[[0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
  [0, 1, 0]]]
```

### Output

Similar to the input, the output predictions consists of sequences of one-hot embeddings ie. we represent the counts as one-hot vectors.

The shape of the predictions is determined by the input parameters. First, the length of the prediction corresponds to the `vocab_size`. Next the dimension of each step in a predicted sequence corresponds to `max_len + 1`.

*It is `max_len + 1` because the minimum count is 0 and the maximum count is `max_len`. So there are `max_len + 1` possible answers.*

If we use the input above as an example, the plain prediction is `[[2, 1]]`, while the embedding is a tensor of shape (1, 2, 5):

```
[[[0, 0, 1, 0, 0],
  [0, 1, 0, 0, 0]]]
```

### Decoder Input

This serves as **Queries** for the Encoder-Decoder Attention mechanism (below).

### Model

There are 3 main parts to the model used in the script:

1. Input Encoding
2. Encoder-Decoder Attention
3. Output Encoding

**1. Input Encoding**

The input is fed into a regular fully-connected feedforward network, which produces outputs of `hidden` dimension. In the script, we use `hidden=64` by default.

Using the example above, with an input tensor of shape (1, 4, 3), the network will output a tensor of shape (1, 4, 64).

**2. Encoder-Decoder Attention**

Here we implement the scaled dot-product attention.

The essence of attention lies in the idea of **Queries**, **Keys** and **Values**. 

Here's an analogy. Suppose we have a supermarket catalogue of product names and respective prices and I want to calculate the average price of drinks. The **Query** is 'drinks', the **Keys** are all the product names and the **Values** are the respective prices. We focus our attention on **Values** (prices) of **Keys** (product names) that align most with our **Query** ('drinks').

See how the **Query** and **Keys** have to be the same type? You can see if two words align ('drinks' vs 'water') but you can't align a word with a number, or temperature with prices. Likewise, in the attention mechanism, the **Query** and **Keys** have the same dimensions.

In scaled dot-product attention, we calculate alignment using the dot product. Intuitively, we see that if the **Query** and the **Key** are in the same direction, their dot product will be large and positive. If the **Query** and the **Key** are orthogonal, the dot product will be zero. Finally, if the **Query** and the **Key** are in opposite directions, their dot product will be large and negative.

We then allocate more attention on **Values** whose **Keys** align to the **Query**. To do this, we simply multiply the softmax of the dot product by the **Values**. This gives us a weighted sum of the **Values**, where aligned **Keys** contribute more to this weighted sum.

Now, we need to think about what are our **Queries**, **Keys** and **Values** here. Using the above input and output example, we want the first element of an output sequence to count the number of 'A's. Put another way, we want the model to pay attention to the 'A's when predicting the first element. Then we want the model to pay attention to the 'B's when predicting the second element. This means we need two **Queries**, one for finding 'A's and one for finding 'B's.

Following the above example, the input given to us here is the encoded input tensor of shape (1, 4, 64), ie. a length-4 sequence of 64-dim vectors, where each vector is a representation of a character. Now consider if we treat each vector as both **Key** and **Value**. If we want to focus on finding 'A's, we take the **Query** vector for finding 'A's, check for alignment with each of the 4 vectors and allocate more weight (attention) to vectors that are closely aligned with our **Query**.

The weighted sum of the **Values** for a **Query** will then be a 64-dim vector. 

Following the Transformer architecture by Vaswani et al. (2017), the weighted sum is added back to the **Query** via residual connections (Figure 1 and Section 3.1). The sum of **Query** and weighted sum is then passed to the next step.

This is because the **Query** contains important positional information that cannot be found in the **Keys**/**Values**. In a way, the network in the next step uses the positional information provided by the **Query** component to decide how to interpret the vector ie. are we counting 'A's now or 'B's?

Extract of the `attention` method under the `AttentionModel` class in the script.

```python
def attention(self, query, key, value):
		# Equation 1 in Vaswani et al. (2017)
		# 	Scaled dot product between Query and Keys
		output = tf.matmul(query, key, transpose_b=True) / (tf.cast(tf.shape(query)[2], tf.float32) ** 0.5)
		# 	Softmax to get attention weights
		attention_weights = tf.nn.softmax(output)
		# 	Multiply weights by Values
		weighted_sum = tf.matmul(attention_weights, value)
		# Following Figure 1 and Section 3.1 in Vaswani et al. (2017)
		# 	Residual connection ie. add weighted sum to original query
		output = weighted_sum + query
		# 	Layer normalization
		output = tf.nn.l2_normalize(output, dim=1)
		return output, attention_weights
```

**3. Output Encoding**
