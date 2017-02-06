#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is for running code that has already finished debugging. This way we don't need to change pycharm
# configuration and/or type the command each time.

from general_util import *

learning_rate=0.0001 # if colorful_img model, the learning rate the paper was using was 3 * 10^-5. They did not
# saytheir batch size.
# iterations=800000# 800000
epochs = 20
batch_size=12  # The larger the better.
content_weight=5
checkpoint_iterations=500
height = 128
width = 128
generator_network='unet_color' # 'unet_bw' #
input_mode = 'sketch' # 'color' #
output_mode = 'rgb'
print_iteration = 100
do_restore_and_train = False  # True
do_restore_and_generate = False
use_adversarial_net = True
# use_adversarial_net_real = True
use_hint = True
test_img = '20750360_p0_128.png'#u'/home/ubuntu/pixiv/pixiv_testing/骨董屋・三千世界の女主人_12746957.jpg'
# #'source_compressed/chicago.jpg'
test_img_hint = '20750360_p0_128_hint.png'

do_restore_and_train_string = '--do_restore_and_train' if do_restore_and_train else ''
do_restore_and_generate_string = '--do_restore_and_generate' if do_restore_and_generate else ''
use_adversarial_net_string = '--use_adversarial_net' if use_adversarial_net else ''
use_hint_string = '--use_hint' if use_hint else ''
# restore_from_noadv_to_adv_string = '--restore_from_noadv_to_adv' if use_adversarial_net_real != use_adversarial_net else ''

preprocessed_folder='/home/ubuntu/pixiv_downloaded_sketches_lnet_128/'
preprocessed_file_path_list = '/home/ubuntu/pixiv_downloaded_sketches_lnet_128/image_files_relative_paths.txt'

checkpoint_output='output_checkpoint/colorsketches-%s-input_mode-%s-output_mode-%s-adv_net-%s-hint-%s-epochs-%d-batchsize-%d-lr-%f' \
                  '-content-%d_%%s' \
                  '.jpg' % (generator_network, input_mode, output_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size,
                            learning_rate, content_weight)
output='output/colorsketches-%s-input_mode-%s-output_mode-%s-adv_net-%s-hint-%s-epochs-%d-batchsize-%d-lr-%f-content-%d.jpg' % \
       (generator_network, input_mode, output_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size, learning_rate, content_weight)
model_save_dir='model/colorsketches-%s-input_mode-%s-output_mode-%s-adv_net-%s-hint-%s-epochs-batchsize-%d-%d-lr-%f-content-%d/' % \
               (generator_network, input_mode, output_mode, str(use_adversarial_net), str(use_hint), epochs, batch_size, learning_rate, content_weight)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

os.system('python ~/PycharmProjects/PaintsTensorflow/color_sketches.py --learning_rate=%f --num_epochs=%d '
          '--batch_size=%d --generator_network=%s --output_mode=%s --input_mode=%s --preprocessed_folder=%s '
          '--preprocessed_file_path_list=%s '
          '--content_weight=%d --checkpoint_iterations=%d --width=%d --height=%d --checkpoint_output=%s --test_img=%s --test_img_hint=%s --output=%s --model_save_dir=%s --print_iterations=%d %s %s %s %s'
          % (learning_rate, epochs, batch_size, generator_network, output_mode, input_mode, preprocessed_folder,
             preprocessed_file_path_list, content_weight, checkpoint_iterations, width, height,
             checkpoint_output, test_img, test_img_hint, output, model_save_dir, print_iteration, do_restore_and_train_string, do_restore_and_generate_string, use_adversarial_net_string, use_hint_string))
