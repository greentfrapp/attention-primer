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

We implement the Scaled Dot-Product Attention (described in Section 3.2.1 of [Vaswani *et al.* (2017)](https://arxiv.org/abs/1706.03762)) for this task and train without the use of recurrent or convolutional networks.

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

### Output / Labels

Similar to the input, the output predictions consists of sequences of one-hot embeddings ie. we represent the counts as one-hot vectors.

The shape of the predictions is determined by the input parameters. First, the length of the prediction corresponds to the `vocab_size`. Next the dimension of each step in a predicted sequence corresponds to `max_len + 1`.

*It is `max_len + 1` because the minimum count is 0 and the maximum count is `max_len`. So there are `max_len + 1` possible answers.*

If we use the input above as an example, the plain prediction is `[[2, 1]]`, while the embedding is a tensor of shape (1, 2, 5):

```
[[[0, 0, 1, 0, 0],
  [0, 1, 0, 0, 0]]]
```

### Model

**Attention ie. Queries, Keys, Values**

The essence of attention lies in the idea of **Queries**, **Keys** and **Values**. 

Suppose we have a supermarket catalogue of product names and respective prices and I want to calculate the average price of drinks. The **Query** is 'drinks', the **Keys** are all the product names and the **Values** are the respective prices. We focus our attention on **Values** (prices) of **Keys** (product names) that align most with our **Query** ('drinks').

See how the **Query** and **Keys** have to be the same type? You can see if two words align ('drinks' vs 'water') but you can't align a word with a number, or temperature with prices. Likewise, in the attention mechanism, the **Query** and **Keys** have to have the same dimensions.

In scaled dot-product attention, we calculate alignment using the dot product. Intuitively, we see that if the **Query** and the **Key** are in the same direction, their dot product will be large and positive. If the **Query** and the **Key** are orthogonal, the dot product will be zero. Finally, if the **Query** and the **Key** are in opposite directions, their dot product will be large and negative.

We then allocate more attention on **Values** whose **Keys** align to the **Query**. To do this, we simply multiply the softmax of the dot product by the **Values**. This gives us a weighted sum of the **Values**, where aligned **Keys** contribute more to this weighted sum.

Extract of the `attention` method under the `AttentionModel` class in the script:

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

**Add & Norm**

Following the Transformer architecture by Vaswani et al. (2017), the weighted sum is added back to the **Query** via residual connections (Figure 1 and Section 3.1). The sum of **Query** and weighted sum is then layer-normalized and passed to the next step.

Why do we use the sum of the weighted sum and the original **Query** instead of just the weighted sum?

This is because the **Query** may contain important information that cannot be found in the **Values**. To take a simple example, suppose I have two **Key**/**Value** pairs, `[1 0]:[0 1]` and `[0 1]:[0 1]` ie. two different **Keys** with both having the same **Values**. If my **Query** is `[1 0]`, then my weighted sum will be `[0 1]` ie. the **Value** of **Key** `[1 0]`. If we pass the weighted sum to the next layer, we lose information about the **Query**. The next layer has no way of telling whether my **Query** was `[1 0]` or `[0 1]`. If we instead pass the sum of the weighted sum and the original **Query**, we retain the information since the vector that we pass on can be either `[2 0]` or `[1 1]` (assuming no layer normalization).

Now, we need to think about what are our **Queries**, **Keys** and **Values**. 

**Queries**

We want the first step of the output sequence to count the number of 'A's and the second step to count the number of 'B's and so on. Why not have one **Query** vector per output step? The first **Query** vector can check for 'A's and the second **Query** vector can check for 'B's and so on.

In that case, our **Queries** tensor will be of shape (`batch_size`, `vocab_size`, `hidden`). `vocab_size` is the second dimension since we have one **Query** vector for each element in our vocabulary. `hidden` will be the dimension of each **Query** vector.

We will let the model learn the **Queries** tensor, by initializing it as a trainable `tf.Variable`:

```python3
decoder_query = tf.Variable(
			initial_value=np.zeros((1, self.vocab_size, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="decoder_query",
		)
```

*Notice the first dimension for `decoder_query` in the code above is 1. This is because we will use `tf.tile` to change it to `batch_size` at runtime.*

**Keys and Values**

Following the above input/output examples, our input is of shape (1, 4, 3) ie. a length-4 sequence of 3-dim vectors, where each vector is a representation of a character. To create **Key**/**Value** pairs for each character, we can simply pass the input into two regular fully-connected feedforward networks - one for generating a **Key** for each character and one for generating a **Value**.

Like the **Queries** tensor, we set the output dimension for both networks to `hidden` where `hidden=64` by default. We then end up with a **Keys** tensor and a **Values** tensor, where both are of shape (1, 4, 64). Each 64-dim vector in the **Keys** tensor corresponds to a 64-dim vector in the **Values** tensor. 





