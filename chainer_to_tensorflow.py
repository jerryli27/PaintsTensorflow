"""
This file converts chainer parameters in npz format into tensorflow saved models.
"""

"""
This file implements a network that can learn to color sketches.
I first saw it at http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
"""

# import gtk.gdk
from sys import stderr

import cv2
import scipy
import tensorflow as tf

import adv_net_util
import conv_util
import colorful_img_network_connected_rgbbin_util
import colorful_img_network_connected_util
import colorful_img_network_mod_util
import colorful_img_network_util
import sketches_util
import unet_both_util
import unet_bw_util
import unet_color_util
import unet_util
from general_util import *


try:
    image_summary = tf.image_summary
    scalar_summary = tf.scalar_summary
    histogram_summary = tf.histogram_summary
    merge_summary = tf.merge_summary
    SummaryWriter = tf.train.SummaryWriter
except:
    image_summary = tf.summary.image
    scalar_summary = tf.summary.scalar
    histogram_summary = tf.summary.histogram
    merge_summary = tf.summary.merge
    SummaryWriter = tf.summary.FileWriter


COLORFUL_IMG_NUM_BIN = 6  # Temporary

# TODO: change rtype
def convert(height, width, batch_size,
                       learning_rate, npz_file, generator_network='unet',
                       use_adversarial_net = False, use_hint = False,
                       adv_net_weight=1.0, weight_decay_lambda=1e-5, save_dir="model/",
                       input_mode = 'sketch', output_mode = 'rgb', use_cpu = False):
    """
    """

    input_shape = (1, height, width, 3)
    print('The input shape is: %s. Input mode is: %s. Output mode is: %s. Using %s generator network' % (str(input_shape),
          input_mode, output_mode, generator_network))

    # Define tensorflow placeholders and variables.
    with tf.Graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1 if generator_network!= 'unet_bw' else 3],
                                      name='input_sketches' if input_mode=='sketch' else 'input_bw')
        if use_hint:
            input_hint = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 3], name='input_hint')
            input_concatenated = tf.concat(3, (input_images, input_hint))
            if generator_network == 'unet_color':
                assert input_mode == 'sketch'
                bw_output = unet_color_util.net(input_concatenated)
                raise NotImplementedError
            elif generator_network == 'unet_bw':
                assert input_mode == 'color' and not use_adversarial_net and not use_hint
                bw_output = unet_bw_util.net(input_concatenated)
            else:
                # TODO: change the error message.
                raise AssertionError("Please input a valid generator network name. Possible options are: TODO. Got: %s"
                                     % (generator_network))
        else:
            if generator_network == 'unet_color':
                assert input_mode == 'sketch'
                bw_output = unet_color_util.net(input_images)
            elif generator_network == 'unet_bw':
                assert input_mode == 'color' and not use_adversarial_net and not use_hint
                bw_output = unet_bw_util.net(input_images)
            else:
                raise AssertionError("Please input a valid generator network name. Possible options are: TODO. Got: %s"
                                     % (generator_network))

        chainer_to_tensorflow_var_dict = {}
        if generator_network=='unet_color':
            pass
        elif generator_network=='unet_bw':
            for i in range(9):
                chainer_to_tensorflow_var_dict['c%d/W' % i] = 'conv_init_varsconv_down_%d/weights_init' % i
                chainer_to_tensorflow_var_dict['c%d/b' % i] = 'conv_init_varsconv_down_%d/bias_init' % i
                chainer_to_tensorflow_var_dict['bnc%d/beta' % i] = 'spatial_batch_normconv_down_%d/offset' % (i)
                chainer_to_tensorflow_var_dict['bnc%d/gamma' % i] = 'spatial_batch_normconv_down_%d/scale' % (i)
            for i in range(9):
                chainer_to_tensorflow_var_dict['dc%d/W' % i] = 'conv_init_varsconv_up_%d/weights_init' % (8-i)
                chainer_to_tensorflow_var_dict['dc%d/b' % i] = 'conv_init_varsconv_up_%d/bias_init' % (8-i)
                chainer_to_tensorflow_var_dict['bnd%d/beta' % i] = 'spatial_batch_normconv_up_%d/offset' % (8-i)
                chainer_to_tensorflow_var_dict['bnd%d/gamma' % i] = 'spatial_batch_normconv_up_%d/scale' % (8-i)
                # conv_tranpose_layerconv_up_%d/spatial_batch_normconv_up_%d/scale
        else:
            raise AssertionError("Please input a valid generator network name. Possible options are: TODO. Got: %s"
                                 % (generator_network))

        learning_rate_init = tf.constant(learning_rate)
        learning_rate_var = tf.get_variable(name='learning_rate_var', trainable=False,
                                                initializer=learning_rate_init)
        bw_expected_output = tf.placeholder(tf.float32,
                                         shape=[batch_size, input_shape[1], input_shape[2], 3 if generator_network!= 'unet_bw' else 1],
                                         name='bw_expected_output')
        # Use the mean difference loss. Used to use tf.nn.l2_loss. Don't know how big of a difference that makes.
        # bw_loss_non_adv =tf.nn.l2_loss(bw_output - bw_expected_output) / batch_size
        bw_loss_non_adv = tf.reduce_mean(tf.abs(bw_output - bw_expected_output))
        weight_decay_loss_non_adv = conv_util.weight_decay_loss(scope='unet')
        generator_loss_non_adv = bw_loss_non_adv + weight_decay_loss_non_adv * weight_decay_lambda
        # TODO: add loss from sketch. That is, convert both generated and real colored image into sketches and compute their mean difference.

        # tv_loss = tv_weight * total_variation(image)

        if generator_network == 'unet_color' or generator_network == 'unet_bw':
            generator_all_var = unet_util.get_net_all_variables()
        else:
            raise AssertionError("Please input a valid generator network name. Possible options are: TODO.")

        if use_adversarial_net:
            adv_net_input = tf.placeholder(tf.float32,
                                             shape=[batch_size, input_shape[1], input_shape[2], 3], name='adv_net_input')
            adv_net_prediction_image_input = adv_net_util.net(adv_net_input)
            adv_net_prediction_generator_input = adv_net_util.net(bw_output, reuse=True)
            adv_net_all_var = adv_net_util.get_net_all_variables()

            weight_decay_loss_adv= conv_util.weight_decay_loss(scope='adv_net')


            logits_from_i = adv_net_prediction_image_input
            logits_from_g = adv_net_prediction_generator_input

            # One represent labeling the image as coming from real image. Zero represent labeling it as generated.
            adv_loss_from_i = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
            adv_loss_from_g = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([batch_size], dtype=tf.int64))) * adv_net_weight

            adv_loss =  adv_loss_from_i + adv_loss_from_g + weight_decay_loss_adv * weight_decay_lambda
            generator_loss_through_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
            # Beta1 = 0.5 according to dcgan paper
            adv_train_step = tf.train.AdamOptimizer(learning_rate_var, beta1=0.5,
                                   beta2=0.999).minimize(adv_loss, var_list=adv_net_all_var)
            generator_train_step_through_adv = tf.train.AdamOptimizer(learning_rate_var, beta1=0.5,
                                   beta2=0.999).minimize(generator_loss_through_adv, var_list=generator_all_var)
            generator_train_step = tf.train.AdamOptimizer(learning_rate_var, beta1=0.9,
                                   beta2=0.999).minimize(generator_loss_non_adv)

            with tf.control_dependencies([generator_train_step_through_adv, generator_train_step]):
                generator_both_train = tf.no_op(name='generator_both_train')


            adv_loss_real_sum = scalar_summary("adv_loss_real", adv_loss_from_i)
            adv_loss_fake_sum = scalar_summary("adv_loss_fake", adv_loss_from_g)
            adv_loss_weight_decay_sum = scalar_summary("adv_loss_weight_decay", weight_decay_loss_adv)

            generator_loss_through_adv_sum = scalar_summary("g_loss_through_adv", generator_loss_through_adv)
            adv_loss_sum = scalar_summary("adv_loss", adv_loss)
            generator_loss_l2_sum = scalar_summary("generator_loss_non_adv", generator_loss_non_adv)
            generator_loss_weight_decay_sum = scalar_summary("generator_loss_weight_decay", weight_decay_loss_non_adv)


            g_sum = merge_summary([generator_loss_through_adv_sum, generator_loss_l2_sum, generator_loss_weight_decay_sum])
            adv_sum = merge_summary([adv_loss_fake_sum, adv_loss_real_sum, adv_loss_weight_decay_sum, adv_loss_sum])
        else:
            # optimizer setup
            # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
            generator_train_step = tf.train.AdamOptimizer(learning_rate_var, beta1=0.9,
                                   beta2=0.999).minimize(generator_loss_non_adv)
            generator_loss_l2_sum = scalar_summary("bw_loss_non_adv", generator_loss_non_adv)
            generator_loss_weight_decay_sum = scalar_summary("generator_loss_weight_decay", weight_decay_loss_non_adv)
            g_sum = merge_summary([generator_loss_l2_sum, generator_loss_weight_decay_sum])

        saver = tf.train.Saver()

        if use_cpu:
            config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
        else:
            config = None
        with tf.Session(config=config) as sess:
            if '0.12.0' in tf.__version__:
                all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            else:
                all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)

            chainer_to_tensorflow_var = generator_all_var
            if use_adversarial_net:
                chainer_to_tensorflow_var = chainer_to_tensorflow_var + adv_net_all_var
            var_not_saved = [item for item in all_vars if item not in (chainer_to_tensorflow_var)]
            sess.run(tf.initialize_variables(var_not_saved))

            chainer_data = np.load(npz_file)
            chainer_var_names = sorted(chainer_data.files)

            for chainer_var_name in chainer_var_names:
                if chainer_var_name in chainer_to_tensorflow_var_dict:
                    tensorflow_var = None
                    tensorflow_var_name = chainer_to_tensorflow_var_dict[chainer_var_name]
                    for possible_var in chainer_to_tensorflow_var:
                        if tensorflow_var_name in possible_var.name:
                            if tensorflow_var is not None:
                                raise AssertionError('Duplicate variable %s and its corresponding variable %s and %s'
                                                     % (chainer_var_name, possible_var.name, tensorflow_var.name))
                            tensorflow_var = possible_var
                    chainer_to_tensorflow_var.remove(tensorflow_var)
                    if tensorflow_var is None:
                        raise AssertionError('Could not find variable %s and its corresponding variable %s'
                                             %(chainer_var_name, tensorflow_var_name))

                    chainer_var = chainer_data[chainer_var_name]
                    try:
                        if len(chainer_var.shape) == 4:
                            # This works for both conv and deconv.
                            sess.run(tensorflow_var.assign(np.transpose(chainer_var,axes=(2,3,1,0))))
                        else:
                            sess.run(tensorflow_var.assign(chainer_var))
                    except ValueError as e:
                        raise ValueError('Error assigning variable %s. Error message %s' %(tensorflow_var.name, e))

            if len(chainer_to_tensorflow_var) != 0 :
                raise AssertionError('Not all tensorflow variables initialized from chainer: %s' %(str(chainer_to_tensorflow_var)))
            saver.save(sess, save_dir + 'model.ckpt', global_step=0)


if __name__ == "__main__":
    save_dir = './model/lnet_converted/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    convert(128,128,1,0.0001,"/home/xor/PycharmProjects/PaintsChainer_py2/cgi-bin/paint_x2_unet/models/liner_f",'unet_bw',input_mode='color',save_dir=save_dir)