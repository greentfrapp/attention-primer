"""
Task 1 - Counting Letters
Simple implementation of attention and tutorial on queries, keys and values
"""

import tensorflow as tf
import numpy as np
import string
from absl import flags
from absl import app
import seaborn
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")
flags.DEFINE_bool("plot", False, "Plot attention heatmap during testing")

# Training parameters
flags.DEFINE_integer("steps", 2000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")
flags.DEFINE_float("lr", 1e-2, "Learning rate")

# Model parameters
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 10, "Maximum input length from toy task")
flags.DEFINE_integer("vocab_size", 3, "Size of input vocabulary, not including ' ' (null character)")
flags.DEFINE_integer("sample_len", 10, "Use this parameter to change sequence length during testing")


class Task(object):

	def __init__(self, max_len=10, vocab_size=3):
		super(Task, self).__init__()
		self.max_len = max_len
		self.vocab_size = vocab_size
		assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

	def next_batch(self, batchsize=100):
		x = np.eye(self.vocab_size + 1)[np.random.choice(np.arange(self.vocab_size + 1), [batchsize, self.max_len])]
		y = np.eye(self.max_len + 1)[np.sum(x, axis=1)[:, 1:].astype(np.int32)]
		return x, y

	def prettify(self, samples):
		samples = samples.reshape(-1, self.max_len, self.vocab_size + 1)
		idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
		dictionary = np.array(list(' ' + string.ascii_uppercase))
		return dictionary[idx]

class AttentionModel(object):

	def __init__(self, sess, sample_len=10, max_len=10, vocab_size=3, hidden=64, name="Counter"):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.sample_len = sample_len
		self.max_len = max_len
		self.vocab_size = vocab_size
		self.hidden = hidden
		self.name = name
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.input = tf.placeholder(
			shape=(None, self.sample_len, self.vocab_size + 1),
			dtype=tf.float32,
			name="input",
		)

		self.labels = tf.placeholder(
			shape=(None, self.vocab_size, self.max_len + 1),
			dtype=tf.float32,
			name="labels",
		)

		query = tf.Variable(
			initial_value=np.zeros((1, self.vocab_size, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="query",
		)

		key_val = tf.layers.dense(
			inputs=self.input,
			units=self.hidden,
			activation=None,
			name="key_val"
		)

		decoding, self.attention_weights = self.attention(
			query=tf.tile(query, multiples=tf.concat(([tf.shape(self.input)[0]], [1], [1]), axis=0)),
			key=key_val,
			value=key_val,
		)
		
		self.logits = tf.layers.dense(
			inputs=decoding,
			units=self.max_len + 1,
			activation=None,
			name="decoding",
		)

		self.predictions = tf.argmax(self.logits, axis=2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(1e-2).minimize(self.loss)

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
		output = tf.contrib.layers.layer_norm(output, begin_norm_axis=2)
		return output, attention_weights

	def save(self, savepath, global_step=None, prefix="ckpt", verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		self.saver.save(self.sess, savepath + prefix, global_step=global_step)
		if verbose:
			print("Model saved to {}.".format(savepath + prefix + '-' + str(global_step)))

	def load(self, savepath, verbose=False):
		if savepath[-1] != '/':
			savepath += '/'
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(self.sess, ckpt)
		if verbose:
			print("Model loaded from {}.".format(ckpt))

def main(unused_args):

	if FLAGS.train:
		tf.gfile.MakeDirs(FLAGS.savepath)
		with tf.Session() as sess:
			model = AttentionModel(sess=sess, sample_len=FLAGS.max_len, max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size, hidden=FLAGS.hidden)
			sess.run(tf.global_variables_initializer())
			task = Task(max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size)
			for i in np.arange(FLAGS.steps):
				minibatch_x, minibatch_y = task.next_batch(batchsize=FLAGS.batchsize)
				feed_dict = {
					model.input: minibatch_x,
					model.labels: minibatch_y,
				}
				_, loss = sess.run([model.optimize, model.loss], feed_dict)
				if (i + 1) % FLAGS.save_every == 0:
					model.save(FLAGS.savepath, global_step=i + 1)
				if (i + 1) % FLAGS.print_every == 0:
					print("Iteration {} - Loss {}".format(i + 1, loss))
			print("Iteration {} - Loss {}".format(i + 1, loss))
			print("Training complete!")
			model.save(FLAGS.savepath, global_step=i + 1, verbose=True)

	if FLAGS.test:
		with tf.Session() as sess:
			model = AttentionModel(sess=sess, sample_len=FLAGS.sample_len, max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size, hidden=FLAGS.hidden)
			model.load(FLAGS.savepath)
			task = Task(max_len=FLAGS.sample_len, vocab_size=FLAGS.vocab_size)
			samples, labels = task.next_batch(batchsize=1)
			print("\nInput: \n{}".format(task.prettify(samples)))
			feed_dict = {
				model.input: samples,
			}
			predictions, attention = sess.run([model.predictions, model.attention_weights], feed_dict)
			print("\nPrediction: \n{}".format(predictions))
			print("\nEncoder-Decoder Attention: ")
			for i, output_step in enumerate(attention[0]):
				print("Output step {} attended mainly to Input steps: {}".format(i, np.where(output_step >= np.max(output_step))[0]))
				print([float("{:.3f}".format(step)) for step in output_step])

			if FLAGS.plot:
				fig, ax = plt.subplots()
				seaborn.heatmap(
					attention[0],
					yticklabels=["output_0", "output_1", "output_2"],
					xticklabels=task.prettify(samples).reshape(-1),
					cbar=False,
					ax=ax,
				)
				ax.set_aspect('equal')
				for tick in ax.get_yticklabels(): tick.set_rotation(0)
				plt.show()


if __name__ == "__main__":
	app.run(main)
