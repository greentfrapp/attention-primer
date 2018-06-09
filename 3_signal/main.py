"""
Task 3 - Signal
Demonstration of positional encodings
"""

import tensorflow as tf
import numpy as np
import string
from absl import flags
from absl import app

FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")

# Training parameters
flags.DEFINE_integer("steps", 2000, "Number of training steps")
flags.DEFINE_integer("print_every", 50, "Interval between printing loss")
flags.DEFINE_integer("save_every", 50, "Interval between saving model")
flags.DEFINE_string("savepath", "models/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")

# Model parameters
flags.DEFINE_bool("pos_enc", False, "Whether to use positional encodings")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 10, "Maximum input length from toy task")
flags.DEFINE_integer("vocab_size", 3, "Size of input vocabulary")
flags.DEFINE_string("signal", None, "Control signal character for test sample")


class Task(object):

	def __init__(self, max_len=10, vocab_size=3):
		super(Task, self).__init__()
		self.max_len = max_len
		self.vocab_size = vocab_size
		assert self.vocab_size <= 26, "vocab_size needs to be <= 26 since we are using letters to prettify LOL"

	def next_batch(self, batchsize=100, signal=None):
		if signal is not None:
			signal = string.ascii_uppercase.index(signal)
			signal = np.eye(self.vocab_size)[np.ones((batchsize, 1), dtype=int) * signal]
		else:
			signal = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, 1])]
		seq = np.eye(self.vocab_size)[np.random.choice(np.arange(self.vocab_size), [batchsize, self.max_len])]
		x = np.concatenate((signal, seq), axis=1)
		y = np.expand_dims(np.eye(self.max_len + 1)[np.sum(np.argmax(signal, axis=2) == (np.argmax(seq, axis=2)), axis=1)], axis=1)
		return x, y

	def prettify(self, samples):
		samples = samples.reshape(-1, self.max_len + 1, self.vocab_size)
		idx = np.expand_dims(np.argmax(samples, axis=2), axis=2)
		# This means max vocab_size is 26
		dictionary = np.array(list(string.ascii_uppercase))
		return dictionary[idx]

class AttentionModel(object):

	def __init__(self, sess, max_len=10, vocab_size=3, hidden=64, name="Signal", pos_enc=True, enc_layers=1):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.max_len = max_len
		self.vocab_size = vocab_size
		self.hidden = hidden
		self.name = name
		self.pos_enc = pos_enc
		self.enc_layers = enc_layers
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.input = tf.placeholder(
			shape=(None, self.max_len + 1, self.vocab_size),
			dtype=tf.float32,
			name="input",
		)

		self.labels = tf.placeholder(
			shape=(None, 1, self.max_len + 1),
			dtype=tf.float32,
			name="labels",
		)

		self.pos_enc = input_positional_encoding = tf.Variable(
			initial_value=np.zeros((1, self.max_len + 1, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="input_positional_coding"
		)

		decoder_input = tf.Variable(
			initial_value=np.zeros((1, 1, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="decoder_input",
		)

		# Input Embedding
		encoding = tf.layers.dense(
			inputs=self.input,
			units=self.hidden,
			activation=None,
			name="encoding"
		)

		if self.pos_enc:
			# Add positional encodings
			encoding += input_positional_encoding

		for i in np.arange(self.enc_layers):
			encoding, _ = self.attention(
				query=encoding,
				key=encoding,
				value=encoding,
			)
			dense = tf.layers.dense(
				inputs=encoding,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="encoder_layer{}_dense1".format(i + 1)
			)
			encoding += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="encoder_layer{}_dense2".format(i + 1)
			)
			encoding = tf.contrib.layers.layer_norm(encoding, begin_norm_axis=2)

		decoding, self.attention_weights = self.attention(
			query=tf.tile(decoder_input, multiples=tf.concat(([tf.shape(self.input)[0]], [1], [1]), axis=0)),
			key=encoding,
			value=encoding,
		)

		decoding = tf.layers.dense(
			inputs=decoding,
			units=self.max_len + 1,
			activation=None,
			name="decoding",
		)

		self.logits = decoding
		self.predictions = tf.argmax(self.logits, axis=2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

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
		# output = tf.nn.l2_normalize(output, dim=1)
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
			model = AttentionModel(sess=sess, max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size, hidden=FLAGS.hidden, pos_enc=FLAGS.pos_enc, enc_layers=FLAGS.enc_layers)
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
			model = AttentionModel(sess=sess, max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size, hidden=FLAGS.hidden, pos_enc=FLAGS.pos_enc, enc_layers=FLAGS.enc_layers)
			model.load(FLAGS.savepath)
			task = Task(max_len=FLAGS.max_len, vocab_size=FLAGS.vocab_size)
			samples, labels = task.next_batch(batchsize=1, signal=FLAGS.signal)
			print("\nInput: \n{}".format(task.prettify(samples)))
			feed_dict = {
				model.input: samples,
			}
			if FLAGS.pos_enc:
				predictions, attention, pos_enc = sess.run([model.predictions, model.attention_weights, model.pos_enc], feed_dict)
			else:
				predictions, attention = sess.run([model.predictions, model.attention_weights], feed_dict)
			print("\nPrediction: \n{}".format(predictions))
			print("\nEncoder-Decoder Attention: ")
			for i, output_step in enumerate(attention[0]):
				print("Output step {} attended mainly to Input steps: {}".format(i, np.where(output_step >= np.max(output_step))[0]))
				print([float("{:.3f}".format(step)) for step in output_step])
			if FLAGS.pos_enc:
				print("\nL2-Norm of Positional Encodings:")
				print([float("{:.3f}".format(step)) for step in np.linalg.norm(pos_enc, ord=2, axis=2)[0]])


if __name__ == "__main__":
	app.run(main)
