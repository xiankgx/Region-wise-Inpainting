import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from inpaint_model import RW_generator


def infer(batch_data, mask, reuse=False):
    # shape = batch_data.get_shape().as_list()
    batch_gt = batch_data/127.5 - 1.
    batch_incomplete = batch_gt * mask

    image_p1, image_p2 = RW_generator(batch_incomplete, mask,
                                      reuse=reuse)

    image_c2 = batch_incomplete * mask + image_p2 * (1. - mask)
    image_c2 = (image_c2 + 1.) * 127.5
    return image_c2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training code')
    parser.add_argument('--test_data_path', type=str, default=" ",
                        help='test_data_path')
    parser.add_argument('--mask_path', type=str, default=" ",
                        help='mask_path')
    parser.add_argument('--model_path', type=str, default=" ",
                        help='model_path')
    parser.add_argument('--file_out', type=str, default="./result",
                        help='result_path')
    parser.add_argument('--width', type=int, default=256,
                        help='images width')
    parser.add_argument('--height', type=int, default=256,
                        help='images height')
    args = parser.parse_args()

    images = tf.placeholder(tf.float32, [None, None, None, 3],
                            name='image')
    mask = tf.placeholder(tf.float32, [None, None, None, 1],
                          name='mask')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    inpainting_result = infer(images, mask)
    saver_pre = tf.train.Saver()
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess.run(init_op)
    saver_pre.restore(sess, args.model_path)

    # file_test = args.test_data_path
    # file_mask = args.mask_path

    if os.path.isfile(args.test_data_path):
        input_files = [args.test_data_path]
    elif os.path.isdir(args.test_data_path):
        input_files = glob.glob(args.test_data_path + "/**/*",
                                recursive=True)
    else:
        assert not os.path.exists(args.test_data_path)
        raise FileNotFoundError(
            f"test_data_path not found: {args.test_data_path}")

    if os.path.isfile(args.mask_path):
        input_masks = [args.mask_path]
    elif os.path.isdir(args.mask_path):
        input_masks = glob.glob(args.mask_path + "/**/*",
                                recursive=True)
    else:
        assert not os.path.exists(args.mask_path)
        raise FileNotFoundError(
            f"mask_path not found: {args.mask_path}")

    if len(input_masks) != len(input_files):
        input_masks = np.random.choice(input_masks, size=len(input_files),
                                       replace=True)

    print(f"Samples: {len(input_files)}")

    os.makedirs(args.file_out, exist_ok=True)
    for file_test, file_mask in tqdm(zip(input_files, input_masks), desc="Inferencing"):
        test_mask = cv2.resize(cv2.imread(file_mask),
                               (args.height, args.width))
        test_mask = test_mask[:, :, 0:1]
        test_mask = 0. + test_mask//255
        test_mask[test_mask >= 0.5] = 1
        test_mask[test_mask < 0.5] = 0
        test_mask = 1 - test_mask
        test_image = cv2.imread(file_test)[..., ::-1]
        img_h, img_w = test_image.shape[:2]
        test_image = cv2.resize(test_image, (args.height, args.width))
        test_mask = np.expand_dims(test_mask, 0)
        test_image = np.expand_dims(test_image, 0)

        img_out = sess.run(inpainting_result,
                           feed_dict={
                               mask: test_mask,
                               images: test_image
                           })

        cv2.imwrite(os.path.join(args.file_out, os.path.basename(file_test)),
                    cv2.resize(img_out[0][..., ::-1], (img_w, img_h),
                               interpolation=cv2.INTER_CUBIC))
        cv2.imwrite(os.path.join(args.file_out, os.path.splitext(os.path.basename(file_test))[0] + "_input.png"),
                    cv2.resize(test_image[0][..., ::-1] * test_mask[0] + 255 * (1 - test_mask[0]), (img_w, img_h),
                               interpolation=cv2.INTER_CUBIC))
