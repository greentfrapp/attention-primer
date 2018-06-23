"""
Task 3 - Signal
Demonstration of positional encodings
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import string
import codecs
import regex
import json
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
flags.DEFINE_string("savepath", "models_de2en/", "Path to save or load model")
flags.DEFINE_integer("batchsize", 100, "Training batchsize per step")

# Model parameters
flags.DEFINE_bool("multihead", True, "Whether to use multihead or singlehead attention")
flags.DEFINE_integer("heads", 4, "Number of heads for multihead attention")
flags.DEFINE_bool("pos_enc", True, "Whether to use positional encodings")
flags.DEFINE_integer("enc_layers", 1, "Number of self-attention layers for encodings")
flags.DEFINE_integer("hidden", 64, "Hidden dimension in model")

# Task parameters
flags.DEFINE_integer("max_len", 10, "Maximum input length from toy task")
flags.DEFINE_integer("vocab_size", 3, "Size of input vocabulary")
flags.DEFINE_string("signal", None, "Control signal character for test sample")


class Task(object):
	
	def __init__(self):
		self.en_file = "de-en/train.tags.de-en.en"
		self.de_file = "de-en/train.tags.de-en.de"
		self.en_samples = self.get_samples(self.en_file)
		self.de_samples = self.get_samples(self.de_file)
		self.rand_de = np.random.RandomState(1)
		self.rand_en = np.random.RandomState(1)
		self.n_samples = len(self.en_samples)
		self.en_dict = json.load(open("en_dict.json", 'r', encoding='utf-8'))
		self.de_dict = json.load(open("de_dict.json", 'r', encoding='utf-8'))
		self.idx = 0

	def get_samples(self, file):
		text = codecs.open(file, 'r', 'utf-8').read().lower()
		text = regex.sub("<.*>.*</.*>\n", "", text)
		text = regex.sub("[^\n\s\p{Latin}']", "", text)
		samples = text.split('\n')
		return samples

	def embed(self, sample, dictionary, max_len=50):
		sample = sample.split()[:max_len]
		while len(sample) < max_len:
			sample.append('<PAD>')
		tokens = ['<START>']
		tokens.extend(sample)
		tokens.append('<END>')
		idxs = []
		for token in tokens:
			try:
				idxs.append(dictionary.index(token))
			except:
				idxs.append(dictionary.index('<UNK>'))
		idxs = np.array(idxs)
		return np.eye(len(dictionary))[idxs]

	def next_batch(self, batchsize=64):
		start = self.idx
		end = start + batchsize
		if end > self.n_samples:
			end -= self.n_samples
			en_minibatch_text = self.en_samples[start:]
			self.rand_en.shuffle(self.en_samples)
			en_minibatch_text += self.en_samples[:end]
			de_minibatch_text = self.de_samples[start:]
			self.rand_de.shuffle(self.de_samples)
			de_minibatch_text += self.de_samples[:end]
		else:
			en_minibatch_text = self.en_samples[start:end]
			de_minibatch_text = self.de_samples[start:end]
		self.idx = end
		en_minibatch = []
		de_minibatch = []
		for sample in en_minibatch_text:
			en_minibatch.append(self.embed(sample, self.en_dict))
		for sample in de_minibatch_text:
			de_minibatch.append(self.embed(sample, self.de_dict))
		return np.array(de_minibatch), np.array(en_minibatch)

	def prettify(self, sample, dictionary):
		idxs = np.argmax(sample, axis=1)
		return " ".join(np.array(dictionary)[idxs])


class AttentionModel(object):

	def __init__(self, sess, max_len=10, hidden=512, name="Signal", pos_enc=True, enc_layers=6, dec_layers=6, use_multihead=True, heads=8):
		super(AttentionModel, self).__init__()
		self.sess = sess
		self.max_len = 50
		self.en_vocab_size = 1004
		self.de_vocab_size = 1004
		self.hidden = hidden
		self.name = name
		self.pos_enc = pos_enc
		self.enc_layers = enc_layers
		self.dec_layers = dec_layers
		self.use_multihead = use_multihead
		self.heads = heads
		with tf.variable_scope(self.name):
			self.build_model()
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=1)

	def build_model(self):

		self.enc_input = tf.placeholder(
			shape=(None, self.max_len + 2, self.de_vocab_size),
			dtype=tf.float32,
			name="encoder_input",
		)

		self.dec_input = tf.placeholder(
			shape=(None, self.max_len + 2, self.en_vocab_size),
			dtype=tf.float32,
			name="decoder_input",
		)

		self.labels = tf.placeholder(
			shape=(None, self.max_len + 2, self.en_vocab_size),
			dtype=tf.float32,
			name="labels",
		)

		enc_pos_enc = tf.Variable(
			initial_value=tf.zeros((1, self.max_len + 2, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="encoder_positional_coding"
		)

		dec_pos_enc = tf.Variable(
			initial_value=tf.zeros((1, self.max_len + 2, self.hidden)),
			trainable=True,
			dtype=tf.float32,
			name="decoder_positional_coding"
		)

		# Embed inputs to hidden dimension
		enc_input_emb = tf.layers.dense(
			inputs=self.enc_input,
			units=self.hidden,
			activation=None,
			name="encoder_input_embedding",
		)

		dec_input_emb = tf.layers.dense(
			inputs=self.dec_input,
			units=self.hidden,
			activation=None,
			name="decoder_input_embedding",
		)

		# Add positional encodings
		encoding = enc_input_emb + enc_pos_enc
		decoding = dec_input_emb + dec_pos_enc

		for i in np.arange(self.enc_layers):
			# Encoder Self-Attention
			encoding, _ = self.multihead_attention(
				query=encoding,
				key=encoding,
				value=encoding,
				h=self.heads,
			)
			# Encoder Dense
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

		for i in np.arange(self.dec_layers):
			# Decoder Self-Attention
			decoding, _ = self.multihead_attention(
				query=decoding,
				key=decoding,
				value=decoding,
				h=self.heads,
				mask=True,
			)
			# Encoder-Decoder Attention
			decoding, _ = self.multihead_attention(
				query=decoding,
				key=encoding,
				value=encoding,
				h=self.heads
			)
			# Decoder Dense
			dense = tf.layers.dense(
				inputs=decoding,
				units=self.hidden * 2,
				activation=tf.nn.relu,
				name="decoder_layer{}_dense1".format(i + 1)
			)
			decoding += tf.layers.dense(
				inputs=dense,
				units=self.hidden,
				activation=None,
				name="decoder_layer{}_dense2".format(i + 1)
			)
			decoding = tf.contrib.layers.layer_norm(decoding, begin_norm_axis=2)

		decoding = tf.layers.dense(
			inputs=decoding,
			units=self.en_vocab_size,
			activation=None,
			name="decoding",
		)

		self.logits = decoding
		self.predictions = tf.argmax(self.logits, axis=2)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
		self.optimize = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

	def multihead_attention(self, query, key, value, h=4, mask=False):
		W_query = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_key = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_value = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		W_output = tf.Variable(
			initial_value=tf.random_normal((self.hidden, self.hidden), stddev=1e-2),
			trainable=True,
			dtype=tf.float32,
		)
		multi_query = tf.reshape(tf.matmul(tf.reshape(query, [-1, self.hidden]), W_query), [-1, h, tf.shape(query)[1], int(self.hidden/h)])
		multi_key = tf.reshape(tf.matmul(tf.reshape(key, [-1, self.hidden]), W_key), [-1, h, tf.shape(key)[1], int(self.hidden/h)])
		multi_value = tf.reshape(tf.matmul(tf.reshape(value, [-1, self.hidden]), W_value), [-1, h, tf.shape(value)[1], int(self.hidden/h)])
		dotp = tf.matmul(multi_query, multi_key, transpose_b=True) / (tf.cast(tf.shape(multi_query)[-1], tf.float32) ** 0.5)
		attention_weights = tf.nn.softmax(dotp)
		weighted_sum = tf.matmul(attention_weights, multi_value)
		weighted_sum = tf.concat(tf.unstack(weighted_sum, axis=1), axis=-1)
		
		multihead = tf.reshape(tf.matmul(tf.reshape(weighted_sum, [-1, self.hidden]), W_output), [-1, tf.shape(query)[1], self.hidden])
		output = multihead + query
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
			model = AttentionModel(sess=sess, max_len=FLAGS.max_len, hidden=FLAGS.hidden, pos_enc=FLAGS.pos_enc, enc_layers=FLAGS.enc_layers, use_multihead=FLAGS.multihead, heads=FLAGS.heads)
			sess.run(tf.global_variables_initializer())
			task = Task()
			for i in np.arange(FLAGS.steps):
				minibatch_x, minibatch_y = task.next_batch(batchsize=FLAGS.batchsize)
				feed_dict = {
					model.enc_input: minibatch_x,
					model.dec_input: ,
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
			model = AttentionModel(sess=sess, max_len=FLAGS.max_len, hidden=FLAGS.hidden, pos_enc=FLAGS.pos_enc, enc_layers=FLAGS.enc_layers, use_multihead=FLAGS.multihead, heads=FLAGS.heads)
			model.load(FLAGS.savepath)
			task = Task()
			samples, labels = task.next_batch(batchsize=1)
			print("\nInput: \n{}".format(task.prettify(samples[0], task.de_dict)))
			feed_dict = {
				model.input: samples,
			}
			predictions, attention = sess.run([model.logits, model.attention_weights], feed_dict)
			print("\nOutput: \n{}".format(task.prettify(predictions[0], task.en_dict)))

if __name__ == "__main__":
	app.run(main)
