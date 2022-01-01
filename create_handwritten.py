import os
import argparse

import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter

import tensorflow as tf
import collections
from ops import *

tf.compat.v1.disable_eager_execution()

def text_to_img(char, font, img_size):
    w, h = font.getsize(char)

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), char, (0, 0, 0), font=font)

    img_array = np.array(img)
    centered_img_array = np.ones((128, 128, 3), dtype=np.uint8)*255
    h , w, d = np.where(img_array==0)
    height = max(h) - min(h)
    weight = max(w) - min(w)
    offset_h = (128 - height) //2
    offset_w = (128 - weight) //2
    centered_img_array[offset_h:offset_h + height + 1, offset_w:offset_w + weight + 1, :]=\
       img_array[min(h):max(h) + 1, min(w):max(w) + 1, :]
    centered_img = Image.fromarray(centered_img_array, 'RGB')

    centered_img = centered_img.resize((img_size, img_size), Image.ANTIALIAS).convert('L')
    centered_img = centered_img.resize((img_size, img_size), Image.ANTIALIAS)

    return centered_img

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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--text", default="サンプルテキスト")
    
    parser.add_argument("--font_pathA", default="./font/meiryo.ttc")
    parser.add_argument("--font_pathB", default="./font/mogihaPen.ttf")
    parser.add_argument("--font_size", default=92, type=int)
    parser.add_argument("--img_size", default=64, type=int)

    parser.add_argument("--blur_size", default=4, type=int)
    parser.add_argument("--crop_size", default=8, type=int)

    parser.add_argument("--model_dir", default="./checkpoints_v2_")
    parser.add_argument("--out_dir", default="./outputs")
    
    parser.add_argument('--height', help='image height', default=64, type=int)
    parser.add_argument('--width', help='image width', default=64, type=int)
    parser.add_argument('--in_depth', help='image depth', default=3, type=int)
    parser.add_argument('--out_depth', help='output image depth', default=3, type=int)
    parser.add_argument('--lambda_A', help='L1 weight', default=100, type=int)
    parser.add_argument('--ndf', help='number of discriminator filer', default=64, type=int)
    parser.add_argument('--ngf', help='number of generator filer', default=64, type=int)

    args = parser.parse_args()

    return (args)

def main():
    args = get_args()

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    text = list(args.text)
    model_dir = args.model_dir

    model = create_network(args)

    img_size = args.img_size
    font_pathA = args.font_pathA
    font_pathB = args.font_pathB
    font_size = args.font_size
    fontA = ImageFont.truetype(font_pathA, font_size)
    fontB = ImageFont.truetype(font_pathB, font_size)
    blur_size = args.blur_size
    crop_size = args.crop_size

    new_im = Image.new('RGB', (img_size*len(text), img_size))

    with tf.compat.v1.Session() as sess:
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())

        print(model_dir)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load model from ', ckpt.model_checkpoint_path)

        for i, char in enumerate( text ):
            centered_imgA = text_to_img(char, fontA, img_size)
            centered_imgB = text_to_img(char, fontB, img_size)

            centered_imgB = centered_imgB.crop((crop_size, crop_size, img_size-crop_size, img_size-crop_size))
            centered_imgB = centered_imgB.resize((img_size,img_size), Image.ANTIALIAS)

            centered_imgA_blur = centered_imgA.filter(ImageFilter.GaussianBlur(blur_size))
            centered_imgB_blur = centered_imgB.filter(ImageFilter.GaussianBlur(blur_size-2))

            target_im = Image.new('RGB', (img_size*3, img_size))
            target_im.paste(centered_imgB, (0,0))
            target_im.paste(centered_imgA_blur, (img_size,0))
            target_im.paste(centered_imgB_blur, (img_size*2,0))

            target_im.save('./outputs/' + char + '.png')

            im = np.expand_dims(np.array(target_im).astype(np.float32), axis=0)
            im = im/127.5 - 1.0

            gen_fake_B = sess.run(model.fake_B, feed_dict ={model.data: im, model.is_training: False})

            convert_im = np.uint8((gen_fake_B+1.)*127.5)
            convert_im = Image.fromarray(np.squeeze(convert_im))

            new_im.paste( convert_im, (img_size*i, 0) )
        
        new_im.save( os.path.join(out_dir, 'result.png') )
            

if __name__ == "__main__":
    main()