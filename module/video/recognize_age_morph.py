import tensorflow as tf
from age import age_models
import os

model = age_models.AgeModelMorph()
model.build()
predicted_age = model.get_test_output()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
module_path = os.path.dirname(os.path.realpath(__file__))
CKPT_PATH = module_path + "/weights/age/ckpt-30000"


class Runner:
    def __init__(self):
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(allow_empty=True)
        saver.restore(sess, CKPT_PATH)

        self.__input = tf.placeholder(tf.float32, shape=(224, 224, 3))
        rgb = tf.cond(tf.equal(tf.shape(self.__input)[2], 1),
                      lambda: tf.tile(self.__input, [1, 1, 3]),
                      lambda: tf.identity(self.__input))

        boxes = tf.constant([[0, 0, 0.9, 0.9],
                             [0, 0.1, 0.9, 1],
                             [0.1, 0, 1.0, 0.9],
                             [0.1, 0.1, 1.0, 1.0],
                             [0.05, 0.05, 0.95, 0.95]])

        indices = tf.zeros(5, dtype=tf.int32)
        im_cropped = tf.image.crop_and_resize(tf.expand_dims(rgb, 0),
                                              boxes, indices,
                                              model.pretrained.input_shape[1:3])
        self.__img_augment = tf.concat([im_cropped,
                                        tf.gather(im_cropped, tf.range(model.pretrained.input_shape[1] - 1, -1, -1),
                                                  axis=2)
                                        ], axis=0)

    def recognize_age(self, im):
        input_image_augmented = sess.run(self.__img_augment, feed_dict={self.__input: im})
        output_age = sess.run(predicted_age, feed_dict={model.imgs: input_image_augmented})
        return output_age
