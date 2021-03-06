from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
import scipy.io
import time
import os
import collections
import argparse
from ops import *

tf.compat.v1.disable_eager_execution()

def load_image(im_path):
	im = Image.open(im_path)
	im = np.expand_dims(np.array(im).astype(np.float32), axis=0)
	im = im/127.5 - 1.0
	return im

def save_image(im, im_path, phase, ptype, exp_name):
	im = np.uint8((im+1.)*127.5)
	im = Image.fromarray(np.squeeze(im))
	data_name = im_path.split("/")[3]
	if phase=="train":
		im.save(os.path.join('checkpoints'+exp_name, ptype+data_name))
	else:
		im.save(os.path.join('result'+exp_name,  ptype+data_name))
		print(os.path.join('result'+exp_name,  ptype+data_name))


def discriminator(inputs, targets, opt, name, update_collection=None, reuse=False):
	with tf.compat.v1.variable_scope(name) as scope:
		if reuse:
			scope.reuse_variables()
		#CNN
		ndf = opt.ndf
		x = tf.concat([inputs, targets], axis=3)

		x = conv2d(inputs, ndf, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_1')
		x = leaky_relu(instance_norm(x,'d_bn_1'))

		x = conv2d(x, ndf*2, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_2')
		x = leaky_relu(instance_norm(x,'d_bn_2'))

		x = conv2d(x, ndf*4, 4, strides=2, spectral_normed=True, update_collection=update_collection, name='d_conv_3')
		x = leaky_relu(instance_norm(x,'d_bn_3'))

		x = conv2d(x, ndf*8, 4, strides=1, spectral_normed=True, update_collection=update_collection, name='d_conv_4')
		x = leaky_relu(instance_norm(x,'d_bn_4'))

		x = conv2d(x, 1, 4, strides=1,  spectral_normed=True, update_collection=update_collection, padding='SAME', name = "d_pred_1")
		out = tf.sigmoid(x)
		return out

def generator(inputs, is_training, opt):
	
	with tf.compat.v1.variable_scope("encoder") as scope:
		
		ndf = opt.ndf
		#encoder
		ndf_spec = [ndf, ndf*2, ndf*4, ndf*8, ndf*8, ndf*8, ndf*8]
		encode_layers = []
		x = conv2d(inputs, ndf_spec[0], 4, strides=2, padding='SAME', name='g_conv_1')
		encode_layers.append(x)

		for i in range(1,6): 
			x = leaky_relu(x)
			x = conv2d(x, ndf_spec[i], 4, strides=2, padding='SAME', name='g_conv_%d'%(i+1))
			x = instance_norm(x, 'g_bn_%d'%(i+1))
			encode_layers.append(x)

	with tf.compat.v1.variable_scope("decoder") as scope:
		
		#encoder
		ngf = opt.ngf
		ngf_spec = [ngf*8, ngf*8, ngf*8, ngf*4, ngf*2, ngf]
		
		# print(encode_layers)
		for i in range(0,5):
			if i != 0:
				x = tf.concat([x, encode_layers[5-i]], axis=3)
			x = tf.nn.relu(x)
			x = deconv2d(x, ngf_spec[i], 4, strides=2, padding='SAME', name='g_deconv_%d'%(i+1))
			x = instance_norm(x,'g_bn_%d'%(i+1))
			if i < 3:
				x = dropout(x, rate=0.5, training=is_training)

		x = tf.concat([x, encode_layers[0]], axis=3)
		x = tf.nn.relu(x)
		x = deconv2d(x, 3, strides=2, padding='SAME', name='g_deconv_7')
		output = tf.tanh(x)

		return output


def transform(A, B, C, scale_size, crop_size):
    r_A = tf.image.resize(A, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    r_B = tf.image.resize(B, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
    r_C = tf.image.resize(C, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random.uniform([2], 0, scale_size - crop_size + 1)), dtype=tf.int32)
    if scale_size > crop_size:
        r_A = tf.image.crop_to_bounding_box(r_A, offset[0], offset[1], crop_size, crop_size)
        r_B = tf.image.crop_to_bounding_box(r_B, offset[0], offset[1], crop_size, crop_size)
        r_C = tf.image.crop_to_bounding_box(r_C, offset[0], offset[1], crop_size, crop_size)
    elif scale_size < crop_size:
        raise Exception("scale size cannot be less than crop size")
    return r_A, r_B, r_C

def create_network(opt):
	#parameter
	height = opt.height
	width = opt.width
	in_depth = opt.in_depth
	out_depth = opt.out_depth
	lambda_A = opt.lambda_A
	EPS = 1e-12
	starter_learning_rate = 0.0002
	end_learning_rate = 0.0
	start_decay_step = 1000
	decay_steps = 1000
	beta1 = 0.5
	global_step = tf.Variable(0, trainable=False)
	Model = collections.namedtuple("Model", ['learning_rate','data', 'is_training', 'input_A', 'input_B', 'fake_blur_B', 'fake_B', 'd_1_solver',\
	 'g_1_solver', 'd_2_solver', 'g_2_solver', 'g_1_loss_L1_summary', 'g_1_loss_GAN_summary', 'd_1_loss_sum', 'g_2_loss_L1_summary', 'g_2_loss_GAN_summary', 'd_2_loss_sum'])
	
	#placeholder/input
	data = tf.compat.v1.placeholder(tf.float32, [None, height, width*3, in_depth], name ="data_AB")
	is_training = tf.compat.v1.placeholder(tf.bool, name ="is_training")

	input_B, input_A, blur_B = transform(data[:, :, :opt.width, :], data[:, :, opt.width:opt.width*2-1, :], data[:, :, opt.width*2:, :], width+10, width)

	#generator
	with tf.compat.v1.variable_scope("generatorA"):	
		fake_blur_B = generator(input_A, is_training, opt)


	with tf.compat.v1.variable_scope("generatorB"):	
		fake_B = generator(fake_blur_B, is_training, opt)

	#discriminator
	d_1_real = discriminator(input_A, blur_B, opt, update_collection=None, name="discriminatorA")
	d_1_fake = discriminator(input_A, fake_blur_B, opt, update_collection="NO_OPS", name="discriminatorA", reuse=True)


	d_2_real = discriminator(blur_B, input_B, opt, update_collection=None, name="discriminatorB")
	d_2_fake = discriminator(blur_B, fake_B, opt, update_collection="NO_OPS", name="discriminatorB", reuse=True)

	#loss
	with tf.compat.v1.variable_scope("discriminator_loss"):
		d_1_loss = tf.reduce_mean(input_tensor=-(tf.math.log(d_1_real + EPS) + tf.math.log(1 - d_1_fake + EPS)))
		d_2_loss = tf.reduce_mean(input_tensor=-(tf.math.log(d_2_real + EPS) + tf.math.log(1 - d_2_fake + EPS)))

	with tf.compat.v1.variable_scope("generator_loss"):
		g_1_loss_GAN = tf.reduce_mean(input_tensor=-tf.math.log(d_1_fake + EPS))
		g_1_loss_L1 = tf.reduce_mean(input_tensor=tf.abs(blur_B - fake_blur_B))
		g_1_loss = g_1_loss_GAN  + g_1_loss_L1 * lambda_A

		g_2_loss_GAN = tf.reduce_mean(input_tensor=-tf.math.log(d_2_fake + EPS))
		g_2_loss_L1 = tf.reduce_mean(input_tensor=tf.abs(input_B - fake_B))
		g_2_loss = g_2_loss_GAN  + g_2_loss_L1 * lambda_A

	#tensorboard summary
	g_1_loss_L1_summary = tf.compat.v1.summary.scalar("g_1_loss_L1", g_1_loss_L1)
	g_1_loss_GAN_summary = tf.compat.v1.summary.scalar("g_1_loss_GAN", g_1_loss_GAN)
	d_1_loss_sum = tf.compat.v1.summary.scalar("d_1_loss", d_1_loss)

	g_2_loss_L1_summary = tf.compat.v1.summary.scalar("g_2_loss_L1", g_2_loss_L1)
	g_2_loss_GAN_summary = tf.compat.v1.summary.scalar("g_2_loss_GAN", g_2_loss_GAN)
	d_2_loss_sum = tf.compat.v1.summary.scalar("d_2_loss", d_2_loss)

	# optimizer
	learning_rate = (
		tf.compat.v1.where(
			tf.greater_equal(global_step, start_decay_step),
			tf.compat.v1.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
										decay_steps, end_learning_rate,
										power=1.0),
			starter_learning_rate
		)

	)
	d_1_solver = tf.compat.v1.train.AdamOptimizer(starter_learning_rate, 0.5).minimize(d_1_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorA'))
	g_1_solver = tf.compat.v1.train.AdamOptimizer(starter_learning_rate, 0.5).minimize(g_1_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generatorA'))
	
	d_2_solver = tf.compat.v1.train.AdamOptimizer(starter_learning_rate, 0.5).minimize(d_2_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminatorB'))
	g_2_solver = tf.compat.v1.train.AdamOptimizer(starter_learning_rate, 0.5).minimize(g_2_loss, var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generatorB'))
	
	return Model(
		input_A=input_A,
		input_B=input_B,
		learning_rate=learning_rate,
		is_training=is_training,
	 	data=data,
	 	fake_blur_B=fake_blur_B,
	 	fake_B=fake_B,
	 	d_1_solver=d_1_solver,
	 	g_1_solver=g_1_solver,
	 	d_2_solver=d_2_solver,
	 	g_2_solver=g_2_solver,
	 	g_1_loss_L1_summary=g_1_loss_L1_summary,
	 	g_1_loss_GAN_summary=g_1_loss_GAN_summary,
	 	d_1_loss_sum=d_1_loss_sum,
	 	g_2_loss_L1_summary=g_2_loss_L1_summary,
	 	g_2_loss_GAN_summary=g_2_loss_GAN_summary,
	 	d_2_loss_sum=d_2_loss_sum)



def train(opt, model):
	with tf.compat.v1.Session() as sess:
		sess.run(tf.compat.v1.global_variables_initializer())
		exp_name = opt.experiment_name
		writer = tf.compat.v1.summary.FileWriter("logs", sess.graph)
		saver = tf.compat.v1.train.Saver()
		if not os.path.exists('result'+exp_name):
			os.makedirs('result'+exp_name)
		if not os.path.exists('checkpoints'+exp_name):
			os.makedirs('checkpoints'+exp_name)

		with tf.compat.v1.name_scope("parameter_count"):
			parameter_count = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=tf.shape(input=v)) for v in tf.compat.v1.trainable_variables()])
		print(sess.run(parameter_count))

		initial_step = 0
		ckpt = tf.train.get_checkpoint_state('./checkpoints'+exp_name)
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt, ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			initial_step  = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

		counter = 0
		blur_path = os.path.join(opt.dataset, 'train')
		all_blur = os.listdir(blur_path)
		data_blur = [os.path.join(blur_path,img) for img in all_blur if img.endswith('.png')]
		# data_blur = glob(os.path.join('dataset', opt.dataset, 'train', '*.png'))
		data_clear = glob(os.path.join(opt.dataset, 'train_clear', '*.png'))

		for epoch in range(initial_step, opt.epoch):
			# print('epoch number: {}'.format(epoch))
			# if(epoch < 20):
			data = data_blur
			print('data len:', len(data))
			np.random.shuffle(data)
			for i in range(0, len(data)):
				im = load_image(data[i])

				if epoch<=50:
					gen_1_iter = 2
				else:
					gen_1_iter = 1

				for j in range(gen_1_iter):
					_, g_1_loss_L1_summary_str, g_1_loss_GAN_summary_str = sess.run([model.g_1_solver, model.g_1_loss_L1_summary,\
					 model.g_1_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(g_1_loss_L1_summary_str, counter)
					writer.add_summary(g_1_loss_GAN_summary_str, counter)

					_, d_1_loss_sum_str = sess.run([model.d_1_solver, model.d_1_loss_sum],\
					 feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(d_1_loss_sum_str, counter)

				if epoch > 50:
					_, g_2_loss_L1_summary_str, g_2_loss_GAN_summary_str = sess.run([model.g_2_solver, model.g_2_loss_L1_summary,\
					 model.g_2_loss_GAN_summary], feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(g_2_loss_L1_summary_str, counter)
					writer.add_summary(g_2_loss_GAN_summary_str, counter)

					_, d_2_loss_sum_str = sess.run([model.d_2_solver, model.d_2_loss_sum],\
					 feed_dict ={model.data: im, model.is_training: True})
					writer.add_summary(d_2_loss_sum_str, counter)

				if i % 100 == 0:
					print("Epoch: [%2d] [%4d/%4d]" % (epoch, i, len(data)))
					counter += 1
					# input_A, input_B, gen_fake_blur_B, gen_fake_B = sess.run([model.input_A, model.input_B, model.fake_blur_B, model.fake_B], feed_dict ={model.data: im, model.is_training: False})
					# save_image(gen_fake_B, data[i], 'train', '%3df'%epoch, exp_name)
					# save_image(gen_fake_blur_B, data[i], 'train', '%3df_blur'%epoch, exp_name)
					# save_image(input_A, data[i], 'train', '%3dA'%epoch, exp_name)
					# save_image(input_B, data[i], 'train', '%3dB'%epoch, exp_name)
			if (epoch+1) % 20 == 0:
				print("save model at epoch: {}".format(epoch))
				saver.save(sess, 'checkpoints'+exp_name+'/pix2pix', int(counter/len(data)))

				input_A, input_B, gen_fake_blur_B, gen_fake_B = sess.run([model.input_A, model.input_B, model.fake_blur_B, model.fake_B], feed_dict ={model.data: im, model.is_training: False})
				save_image(gen_fake_B, data[i], 'train', '%3df'%epoch, exp_name)
				save_image(gen_fake_blur_B, data[i], 'train', '%3df_blur'%epoch, exp_name)
				save_image(input_A, data[i], 'train', '%3dA'%epoch, exp_name)
				save_image(input_B, data[i], 'train', '%3dB'%epoch, exp_name)
			

def test(opt, model):
	with tf.compat.v1.Session() as sess:
		saver = tf.compat.v1.train.Saver()
		exp_name = opt.experiment_name
		sess.run(tf.compat.v1.global_variables_initializer())
		
		ckpt = tf.train.get_checkpoint_state('./checkpoints'+exp_name)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print('load model from ', ckpt.model_checkpoint_path)
		
		if not os.path.exists('result'+exp_name):
			os.makedirs('result'+exp_name)

		data = glob(os.path.join(opt.dataset, 'test', '*.png'))

		print('test data len:', len(data))

		np.random.shuffle(data)
		for i in range(0, len(data)):
			im = load_image(data[i])
			gen_fake_B = sess.run(model.fake_B, feed_dict ={model.data: im, model.is_training: False})
			save_image(gen_fake_B, data[i], 'test', 'result'+exp_name, exp_name)


def main():
	parser = argparse.ArgumentParser(description="pix2pix.")
	parser.add_argument('--dataset', help='dataset directory', type=str)
	parser.add_argument('--test_dir', help='L1 weight', default='test', type=str)	
	parser.add_argument('--height', help='image height', default=64, type=int)
	parser.add_argument('--width', help='image width', default=64, type=int)
	parser.add_argument('--epoch', help='image epoch', default=200, type=int)
	parser.add_argument('--in_depth', help='image depth', default=3, type=int)	
	parser.add_argument('--out_depth', help='output image depth', default=3, type=int)	
	parser.add_argument('--lambda_A', help='L1 weight', default=100, type=int)
	parser.add_argument('--lr', help='learning rate', default=0.0002, type=float)
	parser.add_argument('--ndf', help='number of discriminator filer', default=64, type=int)
	parser.add_argument('--ngf', help='number of generator filer', default=64, type=int)
	parser.add_argument('--experiment_name', help='enter what you did in this experiment', default='_v1')

	opts = parser.parse_args()

	model = create_network(opts)
	train(opts, model)
	test(opts, model)

if __name__ == "__main__":
    main()
