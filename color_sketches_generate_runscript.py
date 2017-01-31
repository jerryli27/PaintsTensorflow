#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

# # This file is for running code that has already finished debugging. This way we don't need to change pycharm
# # configuration and/or type the command each time.
#
# from general_util import *
#
# learning_rate=0.001
# iterations=800000
# batch_size=1 # Optimally 16, but it ran out of memory. #TODO: change it to 8.
# content_weight=5
# checkpoint_iterations=500
# height = 1200
# width = 960
# print_iteration = 100
# do_restore_and_train = False
# do_restore_and_generate = True
# use_adversarial_net = False
# use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
#
# test_img = 'senga/43369404_p1_master1200.jpg' #'source_compressed/chicago.jpg'
# test_img_name = 'test'
#
# do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
# do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
#
# checkpoint_output='output_checkpoint/colorsketches-adv_net-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%%s.jpg' % (str(use_adversarial_net), iterations, batch_size, learning_rate, content_weight)
# output='output/colorsketches-adv_net-%s-iter-%d-batchsize-%d-lr-%f-content-%d-%s.jpg' % (str(use_adversarial_net), iterations, batch_size, learning_rate, content_weight, test_img_name)
# model_save_dir='model/colorsketches-adv_net-%s-iter-batchsize-%d-%d-lr-%f-content-%d/' % (str(use_adversarial_net), iterations, batch_size, learning_rate, content_weight)
# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
#
# assert do_restore_and_generate == True
#
# # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
# os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s'
#           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string))


# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

import shutil
from general_util import *
from sketches_util import *

# with hint
# learning_rate=0.0005 # 0.001
# iterations=800000# 800000
# batch_size=8  # The larger the better.
# content_weight=5
# checkpoint_iterations=1000
# height = 256
# width = 256
# print_iteration = 100
# do_restore_and_train = False  # True
# do_restore_and_generate = True
# use_adversarial_net = False
# use_hint = True
#
# # Without hint
# learning_rate=0.001 # 0.001
# iterations=800000# 800000
# batch_size=4  # The larger the better.
# content_weight=5
# checkpoint_iterations=1000
# height = 128  # 256
# width = 128  # 256
# print_iteration = 100
# do_restore_and_train = False  # True
# do_restore_and_generate = True
# use_adversarial_net = True
# use_hint = False
#
# test_img = '12746957.jpg'#  'senga/40487409_p0.jpg'
#
# # The following is for blank test image
# test_img_dir = '/home/ubuntu/pixiv/pixiv_training_filtered/'
# all_test_img = get_all_image_paths_in_dir(test_img_dir)
# all_test_img = all_test_img[:20]
#
# for i, test_img in enumerate(all_test_img):
#     test_img_hint = 'senga/40487409_p0_hint_blank.png'
#
#     do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
#     do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
#     use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
#     use_hint_string = '--use_hint' if use_hint else ''
#
#     checkpoint_output='output_checkpoint/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%%s.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#     output='output/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%d.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight, i)
#     model_save_dir='model/colorsketches-adv_net-%s-hint-%s-iter-batchsize-%d-%d-lr-%f-content-%d/' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
#
#     # For utf 8
#     # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#     #           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#
#     # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
#     os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#               % (learning_rate, iterations, 1, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))

# # The following is for manually marked test image
# test_img_hint = '12746957_hint.png'
# test_img_name = '12746957'
#
# do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
# do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
# use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
# use_hint_string = '--use_hint' if use_hint else ''
#
# checkpoint_output='output_checkpoint/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%%s.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
# output='output/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%s.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight, test_img_name)
# model_save_dir='model/colorsketches-adv_net-%s-hint-%s-iter-batchsize-%d-%d-lr-%f-content-%d/' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
# if not os.path.exists(model_save_dir):
#     os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
#
# # For utf 8
# # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
# #           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#
# # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
# os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#           % (learning_rate, iterations, 1, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))

# # The following is for machine generated test image hints
# test_img_dir = '/home/ubuntu/pixiv/pixiv_training_filtered/'
# all_test_img = get_all_image_paths_in_dir(test_img_dir)
# all_test_img = all_test_img[:20]
#
# hint_img_save_dir = test_img_dir[:-1] + '_hint/'
# if not os.path.exists(hint_img_save_dir):
#     os.makedirs(hint_img_save_dir)
#
#
# for i, test_img in enumerate(all_test_img):
#     img = imread(test_img)
#     # If you wish to generate the image of the original size, please uncomment this.
#     height = img.shape[0]
#     width = img.shape[1]
#
#     hint_img_path = hint_img_save_dir + get_file_name(test_img) + '_hint_many.png'
#     #  hint_img_path = hint_img_save_dir + get_file_name(test_img) + '_hint.png'
#     if not os.path.isfile(hint_img_path):
#         hint = generate_hint_from_image(img, max_num_hint=200, min_num_hint=200)
#         # hint = generate_hint_from_image(img)
#         imsave(hint_img_path, hint)
#     # Copy the original imnage to the hint directory.
#     original_image_copy_path = hint_img_save_dir + get_file_name(test_img) + '_original.' + test_img[-3:]
#     if not os.path.isfile(original_image_copy_path):
#         shutil.copy(test_img, original_image_copy_path)
#     # Save the sketch to the hint directory as well
#
#     sketch_img_path = hint_img_save_dir + get_file_name(test_img) + '_sketch.png'
#     if not os.path.isfile(sketch_img_path):
#         sketch = image_to_sketch(img)
#         imsave(sketch_img_path, sketch)
#
#     test_img_hint = hint_img_path
#
#     do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
#     do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
#     use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
#     use_hint_string = '--use_hint' if use_hint else ''
#
#     checkpoint_output='output_checkpoint/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%%s.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#     output='output/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%d.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight, i)
#     model_save_dir='model/colorsketches-adv_net-%s-hint-%s-iter-batchsize-%d-%d-lr-%f-content-%d/' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#     if not os.path.exists(model_save_dir):
#         os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
#
#     # For utf 8
#     # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#     #           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#
#     # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
#     os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#               % (learning_rate, iterations, 1, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#



# # The following is for machine generated test image hints with strict criteria on the sketch.
# filter_percentage_upper_threshold = 0.95
# filter_percentage_lower_threshold = 0.90
# num_image_to_generate = 20
#
# test_img_dir = '/home/ubuntu/pixiv/pixiv_training_filtered/'
# all_test_img = get_all_image_paths_in_dir(test_img_dir)
#
# hint_img_save_dir = test_img_dir[:-1] + '_strict_hint/'
# if not os.path.exists(hint_img_save_dir):
#     os.makedirs(hint_img_save_dir)
#
# num_image_generated = 0
# i = 0
# while num_image_generated < num_image_to_generate:
#     test_img = all_test_img[i]
#     img = imread(test_img)
#
#     content_pre_list_sketches = image_to_sketch(img)
#     content_pre_list_shape = img.shape
#     content_pre_list_num_pixels = content_pre_list_shape[0] * content_pre_list_shape[1]
#     content_pre_list_sketches_binarize = (content_pre_list_sketches == 0).astype(np.int32)
#
#     content_pre_list_sketches_percent_sketch = np.sum(content_pre_list_sketches_binarize / 1.0 / content_pre_list_num_pixels)
#     print(content_pre_list_sketches_percent_sketch)
#
#     if content_pre_list_sketches_percent_sketch >= filter_percentage_lower_threshold and content_pre_list_sketches_percent_sketch <= filter_percentage_upper_threshold:
#         # If you wish to generate the image of the original size, please uncomment this.
#         height = img.shape[0]
#         width = img.shape[1]
#
#         hint_img_path = hint_img_save_dir + get_file_name(test_img) + '_hint_many.png'
#         #  hint_img_path = hint_img_save_dir + get_file_name(test_img) + '_hint.png'
#         if not os.path.isfile(hint_img_path):
#             hint = generate_hint_from_image(img, max_num_hint=200, min_num_hint=200)
#             # hint = generate_hint_from_image(img)
#             imsave(hint_img_path, hint)
#         # Copy the original imnage to the hint directory.
#         original_image_copy_path = hint_img_save_dir + get_file_name(test_img) + '_original.' + test_img[-3:]
#         if not os.path.isfile(original_image_copy_path):
#             shutil.copy(test_img, original_image_copy_path)
#         # Save the sketch to the hint directory as well
#
#         sketch_img_path = hint_img_save_dir + get_file_name(test_img) + '_sketch.png'
#         if not os.path.isfile(sketch_img_path):
#             sketch = image_to_sketch(img)
#             imsave(sketch_img_path, sketch)
#
#         test_img_hint = hint_img_path
#
#         do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
#         do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
#         use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
#         use_hint_string = '--use_hint' if use_hint else ''
#
#         checkpoint_output='output_checkpoint/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%%s.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#         output='output/colorsketches-adv_net-%s-hint-%s-iter-%d-batchsize-%d-lr-%f-content-%d_%d.jpg' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight, num_image_generated)
#         model_save_dir='model/colorsketches-adv_net-%s-hint-%s-iter-batchsize-%d-%d-lr-%f-content-%d/' % (str(use_adversarial_net), str(use_hint), iterations, batch_size, learning_rate, content_weight)
#         if not os.path.exists(model_save_dir):
#             os.makedirs(model_save_dir) # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
#
#         # For utf 8
#         # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#         #           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#
#         # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1
#         os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
#                   % (learning_rate, iterations, 1, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
#
#         num_image_generated += 1
#     i += 1



# The following is for color_sketches networks

learning_rate=0.0002 # if colorful_img model, the learning rate the paper was using was 3 * 10^-5. They did not
# saytheir batch size.
# iterations=800000# 800000
epochs = 10
batch_size=4  # The larger the better... Generally.
content_weight=5
checkpoint_iterations=500
height = 256
width = 256
generator_network= 'colorful_img_bias' # 'colorful_img'
input_mode = 'bw'
print_iteration = 100
do_restore_and_train = False  # True
do_restore_and_generate = True
use_adversarial_net = False
# use_adversarial_net_real = True
use_hint = False

test_img = '12746957.jpg'#  'senga/40487409_p0.jpg'

# The following is for blank test image
test_img_dir = '/mnt/pixiv_drive/home/ubuntu/PycharmProjects/PixivUtil2/test_images/'  #'/home/xor/pixiv_testing/' #
all_test_img = get_all_image_paths_in_dir(test_img_dir)
# all_test_img = all_test_img[:1]
# all_test_img = ['12746957.jpg']

for i, test_img in enumerate(all_test_img):
    test_img_hint = 'senga/40487409_p0_hint_blank.png'

    do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
    do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
    use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
    use_hint_string = '--use_hint' if use_hint else ''
    # restore_from_noadv_to_adv_string = '--restore_from_noadv_to_adv' if use_adversarial_net_real != use_adversarial_net else ''

    checkpoint_output = 'output_checkpoint/colorsketches-%s-input_mode-%s-adv_net-%s-hint-%s-epochs-%d-batchsize-%d-lr-%f' \
                        '-content-%d_%%s.jpg' % (generator_network, input_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size,
                        learning_rate, content_weight)
    output = 'output/colorsketches-train-%d-%s-input_mode-%s-adv_net-%s-hint-%s-epochs-%d-batchsize-%d-lr-%f-content' \
             '-%d.jpg' %(i,generator_network, input_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size,
              learning_rate,content_weight)
    model_save_dir = 'model/colorsketches-%s-input_mode-%s-adv_net-%s-hint-%s-epochs-batchsize-%d-%d-lr-%f-content-%d/' % \
                     (generator_network, input_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size,
                      learning_rate, content_weight)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)  # TODO: add %s content_img_style_weight_mask_string to the model_save_dir
    # For utf 8
    # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --iterations=%d --batch_size=%d --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
    #           % (learning_rate, iterations, batch_size, content_weight, checkpoint_iterations, width, height, checkpoint_output, test_img.encode('utf-8'), test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))

    # NOTE: learning rate is a float !!! not an int. so use %f, not %d... That was the bug that causes the model not to train at all when I have lr < 1

    # subprocess.call(['python','color_sketches.py', '--learning_rate=%f --num_epochs=%d '
    #           '--batch_size=%d --generator_network=%s --input_mode=%s '
    #           '--content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
    #           % (learning_rate, epochs, 1, generator_network, input_mode,
    #              content_weight, checkpoint_iterations, width, height,
    #              checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string)])
    # os.system('python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --num_epochs=%d '
    #           '--batch_size=%d --generator_network=%s --input_mode=%s '
    #           '--content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
    #           % (learning_rate, epochs, 1, generator_network, input_mode,
    #              content_weight, checkpoint_iterations, width, height,
    #              checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
    command = 'python ~/PycharmProjects/my-neural-style/color_sketches.py --learning_rate=%f --num_epochs=%d --batch_size=%d --generator_network=%s --input_mode=%s --content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img="%s" --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s' % (learning_rate, epochs, 1, generator_network, input_mode,
                 content_weight, checkpoint_iterations, width, height,
                 checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string)

    os.system(command)
