import tensorflow as tf
import numpy as np
from absl import flags
from absl import app


class Attention(object):

	def __init__(self):
		super(Attention, self).__init__()

class MNISTAttentionModel(object):

	def __init__(self):
		super(MNISTAttentionModel, self).__init__()
		self.hidden = 64

	def build_model(self):

		self.input = tf.placeholder(
			shape=(None, 28, 84, 1),
			dtype=tf.float32,
			name="input",
		)

		self.labels = tf.placeholder(
			shape=(None, 11),
			dtype=tf.float32,
			name="input",
		)

		conv_emb = tf.layers.conv2d(
			inputs=self.input,
			filters=11,
			kernel_size=[1, 1],
			data_format='channels_last',
			activation=tf.nn.relu,
		)

		enc_atten_1 = Attention(
			query=conv_emb,
			key=conv_emb,
			value=conv_emb,
		)

		dec_atten_1 = Attention(
			query=masked_query,
			key=enc_atten_1,
			value=enc_atten_1,
		)

		logits = tf.layers.dense(
			inputs=tf.reduce_sum(dec_atten_1),
			units=11,
			activation=None,
		)


