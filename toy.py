"""
Toy experiment to illustrate counting using Self-Attention
"""

import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

FLAGS = flags.FLAGS

flags.DEFINE_bool("train", False, "Train")
flags.DEFINE_bool("test", False, "Test")


class ToyTask(object):

	def __init__(self):
		super(ToyTask, self).__init__()

	def next_batch(self, batchsize=100, length=10):
		x = np.eye(4)[np.random.choice(np.arange(4), [batchsize, length])]
		y = np.eye(length + 1)[np.sum(x, axis=1).astype(np.int32)]
		return x, y

class CountingAttentionModel(object):

	def __init__(self, sess, name="Counter"):
		super(CountingAttentionModel, self).__init__()
		self.name = name
		self.sess = sess
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.input = tf.placeholder(
			shape=(None, 10, 4),
			dtype=tf.float32,
			name="input",
		)

		self.labels = tf.placeholder(
			shape=(None, 4, 11),
			dtype=tf.float32,
			name="labels",
		)

		self.dec_input = tf.Variable(
			initial_value=np.expand_dims(np.concatenate((np.ones((1, 8))*-0.02, np.ones((1, 8))*-0.01, np.ones((1, 8))*0.01, np.ones((1, 8))*0.02)), axis=0),
			trainable=True,
			dtype=tf.float32,
			name="dec_input",
		)

		# Does not implement positional encodings for input

		# increase dims
		self.enc_key = tf.layers.dense(
			inputs=self.input,
			units=8,
			activation=None,
			name="enc_key"
		)

		self.enc_val = tf.layers.dense(
			inputs=self.input,
			units=8,
			activation=tf.nn.relu,
			name="enc_val"
		)

		# enc_att_1, self.enc_att_1_w = self.attention(
		# 	query=self.enc, 
		# 	key=self.enc, 
		# 	value=self.enc,
		# )
		# enc_att_1 = tf.layers.dense(
		# 	inputs=enc_att_1,
		# 	units=128,
		# 	activation=tf.nn.relu,
		# 	name="enc_att_dense_1"
		# )
		# enc_att_1 = tf.layers.dense(
		# 	inputs=enc_att_1,
		# 	units=11,
		# 	activation=None,
		# 	name="enc_att_dense_2"
		# )
		# enc_att_1 = tf.nn.l2_normalize(enc_att_1, dim=1)

		# Transit from encoder to decoder
		# Increase dimensions to output dim
		# self.encoding = tf.layers.dense(
		# 	inputs=enc_att_1,
		# 	units=11,
		# 	activation=tf.nn.relu,
		# 	name="enc2dec"
		# )
		# Query for decoder
		dec_att_1, self.dec_att_1_w = self.attention(
			query=tf.tile(self.dec_input, multiples=tf.concat(([tf.shape(self.labels)[0]], [1], [1]), axis=0)),
			key=self.enc_key,
			value=self.enc_val,
		)
		# dec_att_1 = tf.layers.dense(
		# 	inputs=dec_att_1,
		# 	units=128,
		# 	activation=tf.nn.relu,
		# 	name="dec_att_dense_1"
		# )
		dec_att_1 = tf.layers.dense(
			inputs=dec_att_1,
			units=11,
			activation=None,
			name="dec_att_dense_2"
		)
		# dec_att_1 = tf.nn.l2_normalize(dec_att_1, dim=1)

		self.logits = dec_att_1
		self.predictions = tf.argmax(self.logits, axis=2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(3e-3).minimize(self.loss)

	def attention(self, query, key, value):
		output = tf.matmul(query, key, transpose_b=True) #/ (tf.cast(tf.shape(query)[2], tf.float32) ** 2)
		weights = tf.nn.softmax(output)
		output = tf.matmul(weights, value) + query
		output = tf.nn.l2_normalize(output, dim=1)
		return output, weights

	def save(self, savepath, global_step=None):
		self.saver.save(self.sess, savepath, global_step=global_step)

	def load(self, savepath):
		ckpt = tf.train.latest_checkpoint(savepath)
		self.saver.restore(self.sess, ckpt)

def main(unused_args):
	if FLAGS.train:
		with tf.Session() as sess:
			model = CountingAttentionModel(sess)
			sess.run(tf.global_variables_initializer())
			task = ToyTask()
			for i in np.arange(10000):
				minibatch_x, minibatch_y = task.next_batch()
				feed_dict = {
					model.input: minibatch_x,
					model.labels: minibatch_y,
				}
				_, loss = sess.run([model.optimize, model.loss], feed_dict)
				if (i + 1) % 10 == 0:
					model.save("toy_models/", i + 1)
					print("Iteration {} - Loss {}".format(i + 1, loss))
			model.save("toy_models/", i + 1)
			print("Iteration {} - Loss {}".format(i + 1, loss))

	if FLAGS.test:
		with tf.Session() as sess:
			model = CountingAttentionModel(sess)
			# sess.run(tf.global_variables_initializer())
			model.load("toy_models/")
			task = ToyTask()
			samples, labels = task.next_batch(batchsize=1)
			print(samples)
			feed_dict = {
				model.input: samples,
				model.labels: labels,
			}
			predictions, att = sess.run([model.predictions, model.dec_att_1_w], feed_dict)
			print(predictions)
			print(att)
			print()

			dec_input, enc_key = sess.run([model.dec_input, model.enc_key], feed_dict)
			print(dec_input.shape)
			print((np.expand_dims((dec_input[:, 1, :] > 0).astype(int), axis=0)))
			print((enc_key > 0).astype(int))

if __name__ == "__main__":
	app.run(main)
