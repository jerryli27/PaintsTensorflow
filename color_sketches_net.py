"""
This file implements a network that can learn to color sketches.
I first saw it at http://qiita.com/taizan/items/cf77fd37ec3a0bef5d9d
"""

# import gtk.gdk
from sys import stderr

import copy
import cv2
import scipy
import time
import tensorflow as tf

import adv_net_util
import conv_util
import colorful_img_network_util
import sketches_util
import lnet_util
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
def color_sketches_net(height, width, iterations, batch_size, content_weight, tv_weight,
                       learning_rate, generator_network='unet',
                       use_adversarial_net = False, use_hint = False,
                       adv_net_weight=1.0, weight_decay_lambda=1e-5,
                       sketch_reconstruct_weight = 10.0 / 255.0, print_iterations=None,
                       checkpoint_iterations=None, save_dir="model/", do_restore_and_generate=False,
                       do_restore_and_train=False, restore_from_noadv_to_adv = False, preprocessed_folder=None,
                       preprocessed_file_path_list = None,
                       content_preprocessed_folder = None, color_rebalancing_folder = None,
                       from_screenshot=False, from_webcam=False, test_img_dir=None, test_img_hint=None,
                       input_mode = 'sketch', output_mode = 'rgb', use_cpu = False):
    """
    Stylize images.
    TODO: modify the description.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :param: lr_decay_steps: learning rate decays by lr_decay_rate after lr_decay steps.
    Default per https://arxiv.org/abs/1603.03417
    :param: min_lr: The minimum learning rate. Default per https://arxiv.org/abs/1603.03417
    :param: lr_decay_rate: learning rate decays by lr_decay_rate after lr_decay steps.
    Default per https://arxiv.org/abs/1603.03417
    :param: use_semantic_masks: If it is true, the input to the generator network will be the semantic masks instead
    of the content image. The content image will serve as ground truth for loss (I haven't decided whether to use content
    or style loss).
    :rtype: iterator[tuple[int|None,image]]
    """

    # Before training, make sure everything is set correctly.
    if use_hint:
        assert test_img_hint is not None
    height, width = get_compatible_shape(height, width)
    input_shape = (1, height, width, 3)
    print('The input shape is: %s. Input mode is: %s. Output mode is: %s. Using %s generator network' % (str(input_shape),
          input_mode, output_mode, generator_network))

    content_img_preprocessed = None
    sketches_preprocessed = None
    prev_content_preprocessed_file_i = 0

    # Define tensorflow placeholders and variables.
    with tf.Graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[batch_size, input_shape[1], input_shape[2], 1 if generator_network!= 'lnet' else 3],
                                      name='input_sketches' if input_mode=='sketch' else 'input_bw')

        if use_hint:
            input_hint = tf.placeholder(tf.float32,
                                shape=[batch_size, input_shape[1], input_shape[2], 3], name='input_hint')
            input_concatenated = tf.concat(3, (input_images, input_hint))
            if generator_network == 'unet_color':
                assert input_mode == 'sketch' or (input_mode == 'raw_sketch' and do_restore_and_generate)
                color_output = unet_color_util.net(input_concatenated)
                sketch_output = lnet_util.net((color_output - 128) / 128) * 255  # This is the reconstructed sketch from the color output.
            elif generator_network == 'lnet':
                assert input_mode == 'color' and not use_adversarial_net and not use_hint
                # This step is not necessary but kept to be in sync with chainer repo.
                input_concatenated = (input_concatenated - 128 ) / 128
                color_output = lnet_util.net(input_concatenated, trainable=True) * 255
            elif generator_network == 'backprop':
                assert input_mode == 'sketch'
                color_output = tf.get_variable('backprop_input_var',shape=[batch_size, input_shape[1], input_shape[2], 3],
                                                   initializer=tf.random_normal_initializer(mean=128,stddev=10.0)) + 0 * input_images
                sketch_output = lnet_util.net((color_output - 128) / 128) * 255  # This is the reconstructed sketch from the color output.
            else:
                # TODO: change the error message.
                raise AssertionError("Please input a valid generator network name. Possible options are: TODO. Got: %s"
                                     % (generator_network))

        else:
            if generator_network == 'unet_color':
                assert input_mode == 'sketch' or (input_mode == 'raw_sketch' and do_restore_and_generate)
                color_output = unet_color_util.net(input_images)
                sketch_output = lnet_util.net((color_output - 128) / 128) * 255  # This is the reconstructed sketch from the color output.
            elif generator_network == 'lnet':
                assert input_mode == 'color' and not use_adversarial_net and not use_hint
                # This step is not necessary but kept to be in sync with chainer repo.
                input_images = (input_images - 128 ) / 128
                color_output = lnet_util.net(input_images, trainable=True) * 255
            elif generator_network == 'backprop':
                assert input_mode == 'sketch'
                color_output = tf.get_variable('backprop_input_var',shape=[batch_size, input_shape[1], input_shape[2], 3],
                                                   initializer=tf.random_normal_initializer()) + 0 * input_images
                sketch_output = lnet_util.net((color_output - 128) / 128) * 255  # This is the reconstructed sketch from the color output.
            else:
                raise AssertionError("Please input a valid generator network name. Possible options are: TODO. Got: %s"
                                     % (generator_network))

        generator_all_var = unet_util.get_net_all_variables()
        sketch_reconstruct_all_var = lnet_util.get_net_all_variables()

        if not do_restore_and_generate:
            assert preprocessed_folder is not None and preprocessed_file_path_list is not None and \
                   preprocessed_folder[-1] == '/'
            learning_rate_init = tf.constant(learning_rate)
            learning_rate_var = tf.get_variable(name='learning_rate_var', trainable=False,
                                                    initializer=learning_rate_init)
            color_expected_output = tf.placeholder(tf.float32,
                                             shape=[batch_size, input_shape[1], input_shape[2], 3 if generator_network!= 'lnet' else 1],
                                             name='color_expected_output')
            # Use the mean difference loss. Used to use tf.nn.l2_loss. Don't know how big of a difference that makes.
            # color_loss_non_adv =tf.nn.l2_loss(color_output - color_expected_output) / batch_size
            color_loss_non_adv = tf.reduce_mean(tf.abs(color_output - color_expected_output))
            weight_decay_loss_non_adv = conv_util.weight_decay_loss(scope='unet') * weight_decay_lambda
            # This is only for unet_color, not for training the lnet,
            sketch_expected_output = lnet_util.net((color_expected_output - 128) / 128, reuse=True) * 255
            sketch_reconstruct_loss_non_adv = tf.reduce_mean(tf.abs(sketch_output - sketch_expected_output)) * sketch_reconstruct_weight

            generator_loss_non_adv = color_loss_non_adv + weight_decay_loss_non_adv + sketch_reconstruct_loss_non_adv
            # tv_loss = tv_weight * total_variation(image)

            if use_adversarial_net:
                adv_net_input = tf.placeholder(tf.float32,
                                                 shape=[batch_size, input_shape[1], input_shape[2], 3], name='adv_net_input')
                adv_net_prediction_image_input = adv_net_util.net(adv_net_input)
                adv_net_prediction_generator_input = adv_net_util.net(color_output, reuse=True)
                adv_net_all_var = adv_net_util.get_net_all_variables()

                weight_decay_loss_adv= conv_util.weight_decay_loss(scope='adv_net') * weight_decay_lambda


                logits_from_i = adv_net_prediction_image_input
                logits_from_g = adv_net_prediction_generator_input

                # One represent labeling the image as coming from real image. Zero represent labeling it as generated.
                adv_loss_from_i = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_i, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
                adv_loss_from_g = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.zeros([batch_size], dtype=tf.int64))) * adv_net_weight

                adv_loss =  adv_loss_from_i + adv_loss_from_g + weight_decay_loss_adv
                generator_loss_through_adv = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits_from_g, tf.ones([batch_size], dtype=tf.int64))) * adv_net_weight
                # Beta1 = 0.5 according to dcgan paper
                adv_train_step = tf.train.AdamOptimizer(learning_rate_var, beta1=0.5,
                                       beta2=0.999).minimize(adv_loss, var_list=adv_net_all_var)
                # adv_train_step_i = tf.train.AdamOptimizer(learning_rate_var, beta1=0.5,
                #                        beta2=0.999).minimize(adv_loss_from_i, var_list=adv_net_all_var)
                # adv_train_step_g = tf.train.AdamOptimizer(learning_rate_var, beta1=0.5,
                #                        beta2=0.999).minimize(adv_loss_from_g, var_list=adv_net_all_var)
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
                generator_loss_weight_decay_sum = scalar_summary("generator_loss_weight_decay",
                                                                 weight_decay_loss_non_adv)
                sketch_reconstruct_loss_non_adv_sum = scalar_summary("sketch_reconstruct_loss_non_adv",
                                                                     sketch_reconstruct_loss_non_adv)

                g_sum = merge_summary(
                    [generator_loss_through_adv_sum, generator_loss_l2_sum, generator_loss_weight_decay_sum,
                     sketch_reconstruct_loss_non_adv_sum])
                adv_sum = merge_summary([adv_loss_fake_sum, adv_loss_real_sum, adv_loss_weight_decay_sum, adv_loss_sum])
            else:
                # optimizer setup
                # Training using adam optimizer. Setting comes from https://arxiv.org/abs/1610.07629.
                generator_train_step = tf.train.AdamOptimizer(learning_rate_var, beta1=0.9,
                                       beta2=0.999).minimize(generator_loss_non_adv)
                generator_loss_l2_sum = scalar_summary("color_loss_non_adv", generator_loss_non_adv)
                generator_loss_weight_decay_sum = scalar_summary("generator_loss_weight_decay", weight_decay_loss_non_adv)
                sketch_reconstruct_loss_non_adv_sum = scalar_summary("sketch_reconstruct_loss_non_adv", sketch_reconstruct_loss_non_adv)
                g_sum = merge_summary([generator_loss_l2_sum, generator_loss_weight_decay_sum, sketch_reconstruct_loss_non_adv_sum])


            def print_progress(i, feed_dict, adv_feed_dict, start_time, total_iterations, last=False):
                stderr.write(
                    'Iteration %d/%d\n' % (i + 1, iterations))
                if last or (print_iterations and i % print_iterations == 0):
                    current_time = time.time()
                    if i > 0:
                        seconds_passed = current_time - start_time
                        seconds_remaining = float(total_iterations - i) / i * seconds_passed
                        m, s = divmod(seconds_remaining, 60)
                        h, m = divmod(m, 60)
                        stderr.write('Estimated time remaining: "%d:%02d:%02d"' % (h, m, s))
                    stderr.write('Learning rate %f\n' % (learning_rate_var.eval()))
                    # TODO: change this
                    stderr.write(' generator l2 loss: %g\n' % generator_loss_non_adv.eval(feed_dict=feed_dict))
                    stderr.write('       sketch loss: %g\n' % sketch_reconstruct_loss_non_adv.eval(feed_dict=feed_dict))
                    if not generator_network == 'backprop':
                        stderr.write('  w decay gen loss: %g\n' % weight_decay_loss_non_adv.eval(feed_dict=feed_dict))
                    # if generator_network == 'unet_both' or generator_network == 'colorful_img_both':
                    #     stderr.write('           bw loss: %g\n' % color_loss_non_adv.eval(feed_dict=feed_dict))
                        # stderr.write('           ab loss: %g\n' % ab_loss_non_adv.eval(feed_dict=feed_dict))
                    if use_adversarial_net:
                        stderr.write('   adv_from_i loss: %g\n' % adv_loss_from_i.eval(feed_dict=adv_feed_dict))
                        stderr.write('   adv_from_g loss: %g\n' % adv_loss_from_g.eval(feed_dict=adv_feed_dict))
                        stderr.write('generator adv loss: %g\n' % generator_loss_through_adv.eval(feed_dict=adv_feed_dict))
                        stderr.write('  w decay adv loss: %g\n' % weight_decay_loss_adv.eval(feed_dict=adv_feed_dict))



        # Optimization
        # It used to track and record only the best one with lowest loss. This is no longer necessary and I think
        # just recording the one generated at each round will make it easier to debug.
        best_image = None
        start_time = time.time()
        if restore_from_noadv_to_adv and use_adversarial_net:
            saver = tf.train.Saver(generator_all_var + [learning_rate_var])
        else:
            saver = tf.train.Saver()

        if use_cpu:
            config = tf.ConfigProto(
                device_count = {'GPU': 0}
            )
        else:
            config = None
        with tf.Session(config=config) as sess:
            if do_restore_and_generate:
                assert batch_size == 1
                ckpt = tf.train.get_checkpoint_state(save_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    stderr("No checkpoint found. Exiting program")
                    return

                if from_screenshot:
                    # This is the x and y offset, the coordinate where we start capturing screen shot.
                    kScreenX = 300
                    kScreenY = 300
                elif from_webcam:
                    cap = cv2.VideoCapture(0)
                    # Set width and height.
                    ret = cap.set(3, 1280)
                    ret = cap.set(4, 960)
                    ret, frame = cap.read()
                    print('The dimension of this camera is : %d x %d' % (frame.shape[1], frame.shape[0]))
                else:
                    assert test_img_dir is not None
                iterator = 0

                while from_screenshot or from_webcam or (iterator == 0):
                    if from_screenshot:
                        pass
                        # w = gtk.gdk.get_default_root_window()
                        # sz = w.get_size()
                        # print "The size of the window is %d x %d" % sz
                        # pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, input_shape[1], input_shape[2])
                        # pb = pb.get_from_drawable(w, w.get_colormap(), kScreenX, kScreenY, 0, 0, input_shape[1],
                        #                           input_shape[2])
                        # content_image = pb.pixel_array
                    elif from_webcam:
                        ret, frame = cap.read()
                        content_image = scipy.misc.imresize(frame, (input_shape[1], input_shape[2]))
                    else:
                        content_image = imread(test_img_dir, (input_shape[1], input_shape[2]))
                    content_image = np.array([content_image])
                    if input_mode == 'sketch':
                        color_expected_output = tf.placeholder(tf.float32,
                                                               shape=[batch_size, input_shape[1], input_shape[2],
                                                                      3 if generator_network != 'lnet' else 1],
                                                               name='color_expected_output')
                        sketch_expected_output = lnet_util.net((color_expected_output - 128) / 128, reuse=True) * 255
                        content_image_yuv = cv2.cvtColor(np.asarray(content_image[0,...], dtype=np.uint8), cv2.COLOR_RGB2YUV)
                        image_sketches = sketch_expected_output.eval(feed_dict={color_expected_output:np.array([content_image_yuv])})

                        # image_sketches = sketches_util.image_to_sketch(content_image)
                        # image_sketches = np.expand_dims(image_sketches, axis=3)
                    elif input_mode == 'bw':
                        content_image_lab = colorful_img_network_util.rgb_to_lab(content_image)
                        image_sketches = content_image_lab[...,0:1]
                        # image_sketches = np.expand_dims(rgb2gray(content_image), axis=3)
                    elif input_mode == 'color':
                        image_sketches = np.zeros(content_image.shape)
                        # image_sketches = np.expand_dims(rgb2gray(content_image), axis=3)
                    elif input_mode == 'raw_sketch':
                        image_sketches = rgb2gray(content_image, keep_dim=True)
                    else:
                        raise AssertionError('Input mode error.')


                    # Do some processing...
                    image_sketches, content_image = sketches_util.generate_training_batch(image_sketches, content_image, train=False)

                    # Now generate an image using the style_blend_weights given.
                    if input_mode == 'color':
                        feed_dict = {input_images: content_image}
                    else:
                        feed_dict = {input_images:image_sketches[..., :1]}

                    if use_hint:
                        image_hint = hint_imread(test_img_hint, (input_shape[1], input_shape[2]))
                        feed_dict[input_hint] = np.array([image_hint])

                    generated_bw = color_output.eval(feed_dict=feed_dict)
                    iterator += 1

                    if generator_network!= 'lnet':
                        # Whenever using cv2.cvtColor, be careful not to use float values... It gives out wierd answers.
                        print(generated_bw[0,0,0:5,:])
                        print(content_image[0,0,0:5,:])
                        generated_image = np.array([cv2.cvtColor(np.asarray(generated_bw[0,...], dtype=np.uint8), cv2.COLOR_YUV2RGB)])
                        # generated_image = image_sketches[...,:1]
                    else:
                        generated_image = generated_bw
                    yield (iterator, generated_image)

            else:
                # Initialize log writer
                summary_writer = SummaryWriter("./logs", sess.graph)

                # initialize pre-processsed numpy array
                if content_preprocessed_folder is not None:
                    if not os.path.isfile(content_preprocessed_folder + 'record.txt'):
                        raise AssertionError(
                            'No preprocessed content images found in %s. To use this feature, first use some '
                            'other file to call read_resize_and_save_all_imgs_in_dir.'
                            % (content_preprocessed_folder))
                    content_preprocessed_record = sketches_util.read_preprocessed_sketches_npy_record(content_preprocessed_folder)
                    if content_preprocessed_record[0][3] != height or content_preprocessed_record[0][4] != width:
                        raise AssertionError(
                            'The height and/or width of the preprocessed numpy files does not '
                            'match those of the current setting.')
                    # Read the first file
                    print('Reading preprocessed content images.')
                    content_img_preprocessed = np.load(content_preprocessed_record[prev_content_preprocessed_file_i][0])
                    sketches_preprocessed = np.load(content_preprocessed_record[prev_content_preprocessed_file_i][1])

                # Do Training.
                iter_start = 0
                if do_restore_and_train:
                    ckpt = tf.train.get_checkpoint_state(save_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        iter_start = get_global_step_from_save_dir(ckpt.model_checkpoint_path)
                    else:
                        raise AssertionError("No checkpoint found. Exiting program")
                        return
                    if restore_from_noadv_to_adv and use_adversarial_net:
                        # Simply running this doesn;t seem to work.
                        # sess.run(tf.initialize_variables(adv_net_all_var))

                        # Get all variables except the generator net and the learning rate
                        if '0.12.0' in tf.__version__:
                            all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                        else:
                            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
                        var_not_saved = [item for item in all_vars if item not in (generator_all_var + [learning_rate_var])]
                        sess.run(tf.initialize_variables(var_not_saved))
                        # Now change the saver back to normal
                        saver = tf.train.Saver()
                        raise NotImplementedError
                else:
                    # # In the past I ran this. Now I have lnet which is a pretrained network.
                    # sess.run(tf.initialize_all_variables())

                    saver = tf.train.Saver(sketch_reconstruct_all_var)
                    ckpt = tf.train.get_checkpoint_state('model/chainer_converted/')
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # Get variables not in lnet and initialize them
                    # Get all variables except the generator net and the learning rate
                    if '0.12.0' in tf.__version__:
                        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    else:
                        all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
                    var_not_saved = [item for item in all_vars if item not in sketch_reconstruct_all_var]
                    sess.run(tf.initialize_variables(var_not_saved))
                    # Now change the saver back to normal
                    saver = tf.train.Saver()


                # Get path to all content images.


                image_subpaths = read_preprocessed_file_path_list(preprocessed_file_path_list)


                # Ignore the ones at the end.
                if batch_size != 1 and len(image_subpaths) % batch_size != 0:
                    image_subpaths = image_subpaths[:-(len(image_subpaths) % batch_size)]
                print('The size of training dataset is %d images.' % len(image_subpaths))
                preprocessed_colored_folder = preprocessed_folder + 'color/'
                preprocessed_sketch_folder = preprocessed_folder + 'line/'

                content_dirs = map(lambda p:preprocessed_colored_folder + p, image_subpaths)
                sketch_dirs = map(lambda p:preprocessed_sketch_folder + p, image_subpaths)


                # # Test training GAN differently***
                # generators_turn = True
                # # END TEST***

                current_lr = learning_rate_var.eval()
                if abs(current_lr-learning_rate) > 0.00000001:
                    print("Given learning rate is different from the learning rate stored. Changing lr %f -> %f"
                          %(current_lr, learning_rate))
                    sess.run(learning_rate_var.assign(learning_rate))

                for i in range(iter_start, iterations):
                    if content_preprocessed_folder is not None:
                        current_content_preprocessed_file_i, index_within_preprocessed =  \
                            sketches_util.find_corresponding_sketches_npy_from_record(
                            content_preprocessed_record, i * batch_size, batch_size)
                        if prev_content_preprocessed_file_i != current_content_preprocessed_file_i:
                            prev_content_preprocessed_file_i = current_content_preprocessed_file_i
                            content_img_preprocessed = np.load(content_preprocessed_record[
                                                                   current_content_preprocessed_file_i][0])
                            sketches_preprocessed = np.load(content_preprocessed_record[
                                                                   current_content_preprocessed_file_i][1])
                        content_pre_list = content_img_preprocessed[
                                           index_within_preprocessed:index_within_preprocessed+batch_size,
                                           ...].astype(np.float32)

                        if input_mode == 'sketch':
                            image_sketches = sketches_preprocessed[
                                               index_within_preprocessed:index_within_preprocessed+batch_size,
                                               ...].astype(np.float32)
                            image_sketches = np.expand_dims(image_sketches, axis=3)
                        elif input_mode == 'bw':
                            content_image_lab = colorful_img_network_util.rgb_to_lab(content_pre_list)
                            image_sketches = content_image_lab[...,0:1]
                            # image_sketches = np.expand_dims(rgb2gray(content_pre_list), axis=3)
                        elif input_mode == 'color':
                            image_sketches = content_pre_list
                        elif input_mode == 'raw_sketch':
                            raise AssertionError('Input mode raw_sketch should not be trained.')
                        else:
                            raise AssertionError('Input mode error.')
                    else:

                        current_content_dirs = get_batch_paths(content_dirs, i * batch_size, batch_size)
                        current_sketch_dirs = get_batch_paths(sketch_dirs, i * batch_size, batch_size)
                        content_pre_list = read_and_resize_batch_images(current_content_dirs, None,
                                                                        None)
                        image_sketches = read_and_resize_bw_mask_images(current_sketch_dirs, None, None,
                                                                        len(current_sketch_dirs), 1)

                        # if input_mode == 'sketch':
                        #     image_sketches = sketches_util.image_to_sketch(content_pre_list)
                        #     image_sketches = np.expand_dims(image_sketches, axis=3)
                        # elif input_mode == 'bw':
                        #     content_image_lab = colorful_img_network_util.rgb_to_lab(content_pre_list)
                        #     image_sketches = content_image_lab[...,0:1]
                        #     # image_sketches = np.expand_dims(rgb2gray(content_pre_list), axis=3)
                        # else:
                        #     raise AssertionError('Input mode error.')

                    # Do some processing...
                    image_sketches, content_pre_list = sketches_util.generate_training_batch(image_sketches, content_pre_list, train=True)
                    if generator_network == 'lnet':
                        feed_dict = {color_expected_output: image_sketches[...,:1]}
                    else:
                        feed_dict = {color_expected_output: content_pre_list}

                    if use_hint:
                        # image_hint = sketches_util.generate_hint_from_image(content_pre_list)
                        # feed_dict[input_hint] = image_hint
                        image_hint = image_sketches[...,1:]
                        feed_dict[input_hint] = image_hint

                    image_sketches = image_sketches[...,:1]

                    if input_mode == 'color':
                        feed_dict[input_images] = content_pre_list
                    else:
                        feed_dict[input_images] = image_sketches


                    last_step = (i == iterations - 1)


                    if use_adversarial_net:
                        # adv_feed_dict = {input_images:image_sketches, adv_net_input: content_pre_list}
                        # if use_hint:
                        #     adv_feed_dict[input_hint] = image_hint
                        adv_feed_dict = copy.copy(feed_dict)
                        adv_feed_dict[adv_net_input] = content_pre_list

                        # TEST printing before training
                        print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, start_time=start_time, total_iterations=iterations, last=last_step)


                        # Update D network
                        _, summary_str = sess.run([adv_train_step, adv_sum],
                                                       feed_dict=adv_feed_dict)
                        summary_writer.add_summary(summary_str, i)

                        # Update G network
                        _, summary_str = sess.run([generator_both_train, g_sum],
                                                       feed_dict=adv_feed_dict)
                        summary_writer.add_summary(summary_str, i)

                    else:
                        adv_feed_dict = None
                        print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, start_time=start_time, total_iterations=iterations, last=last_step)

                        _, summary_str = sess.run([generator_train_step, g_sum], feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str,i)

                    # TEST printing after training
                    print_progress(i, feed_dict=feed_dict, adv_feed_dict=adv_feed_dict, start_time=start_time, total_iterations=iterations, last=last_step)

                    if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                        saver.save(sess, save_dir + 'model.ckpt', global_step=i)
                        print('Checkpoint saved.')

                        if test_img_dir is not None:
                            test_image = imread(test_img_dir)
                            test_image_shape = test_image.shape

                        # The for loop will run once and terminate. Can't use return and yield in the same function so this is a hacky way to do it.
                        # Set use_cpu = true to save graphical memory
                        for _, generated_image in color_sketches_net(test_image_shape[0],
                                                                      test_image_shape[1],
                                                                      iterations,
                                                                      1,
                                                                      content_weight, tv_weight,
                                                                      learning_rate,
                                                                      generator_network=generator_network,
                                                                      use_adversarial_net=False,  # use_adversarial_net=use_adversarial_net,
                                                                      use_hint=use_hint,
                                                                      save_dir=save_dir,
                                                                      do_restore_and_generate=True,
                                                                      do_restore_and_train=False,
                                                                      from_screenshot=False,
                                                                      from_webcam=False,
                                                                      test_img_dir=test_img_dir,
                                                                      test_img_hint=test_img_hint,
                                                                      input_mode=input_mode,
                                                                      output_mode=output_mode,
                                                                      use_cpu=use_cpu):
                            pass
                        best_image = generated_image

                        # Because we now have batch, choose the first one in the batch as our sample image.
                        yield (
                            (None if last_step else i),
                            None if test_img_dir is None else
                             best_image
                        )
