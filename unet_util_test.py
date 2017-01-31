import numpy as np

from unet_util import *


class UnetTest(tf.test.TestCase):
    def test_unet(self):
        with self.test_session() as sess:
            batch_size = 1
            height = 53
            width = 67
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            unet_output = net(input_layer, mirror_padding=False)

            image_shape = input_layer.get_shape().as_list()
            final_shape = unet_output.get_shape().as_list()

            self.assertAllEqual(image_shape, final_shape)

            sess.run(tf.initialize_all_variables())

            feed_input = np.ones((batch_size, height, width, num_features))

            feed_dict = {input_layer:feed_input}
            actual_output = unet_output.eval(feed_dict)
            self.assertTrue(actual_output is not None, 'The unet failed to produce an output.')
    def test_get_net_all_variables(self):
        with self.test_session() as sess:
            batch_size = 1
            height = 53
            width = 67
            num_features = 3

            input_layer = tf.placeholder(dtype=tf.float32, shape=(batch_size, height, width, num_features))
            unet_output = net(input_layer, mirror_padding=False)

            all_var = get_net_all_variables()
            # 17 for number of layers except the last layer.
            # 4 variables per layer, 1 weight, 1 bias, 2 for normalization.
            # Last layer is plus 2 because it is not normalized.
            expected_var_number = 4*17+2
            self.assertEqual(len(all_var), expected_var_number)


if __name__ == '__main__':
    tf.test.main()
