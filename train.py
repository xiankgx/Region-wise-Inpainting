import argparse
import glob
import os
from functools import partial

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from inpaint_model import build_graph_with_loss
from mask_online import continuous_mask, discontinuous_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training code')
    parser.add_argument('--train_data_path', type=str, default="",
                        help='training data path')

    parser.add_argument('--epoch', type=int, default=20,
                        help='training epoch')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of workers for dataloading')

    parser.add_argument('--width', type=int, default=256,
                        help='images width')
    parser.add_argument('--height', type=int, default=256,
                        help='images height')

    parser.add_argument('--mask_area', type=int, default=60,
                        help='mask_area')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='training epoch')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2')

    parser.add_argument('--lambda_style', type=float, default=0.001,
                        help='weight of style_loss')
    parser.add_argument('--lambda_cor', type=int, default=0.00001,
                        help='weight of correlation loss')
    parser.add_argument('--lambda_adv', type=int, default=1.0,
                        help='weight of adversial loss')
    parser.add_argument('--alpha', type=int, default=0.001,
                        help='weight of penalization of discriminator for ground-truth images')

    parser.add_argument('--stage', type=int, default=0,
                        help='training stage')
    parser.add_argument('--mask_type', type=int, default=1,
                        help='0: discontinuous mask 1: continuous mask')
    parser.add_argument('--adv_type', type=str, default=' ',
                        help='type of adversial loss: wgan, gan, hinge')

    parser.add_argument('--vgg_path', type=str, default='./vgg/vgg16.npy',
                        help='path of vgg16')
    parser.add_argument('--pretrained_model', type=str, default='./pretrained_model/v19.ckpt',
                        help='pretrained_model path')
    parser.add_argument('--output', type=str, default='./output/',
                        help='path to save the model and summary')
    args = parser.parse_args()

    # Create placeholders
    image = tf.placeholder(tf.float32, [None, None, None, 3],
                           name="image")
    mask = tf.placeholder(tf.float32, [None, None, None, 1],
                          name='mask')

    # Create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Build graph
    if args.stage == 0:
        g_vars, _, adv_g_loss, _, rec_loss, correlation_loss, style_loss \
            = build_graph_with_loss(image,
                                    args.batch_size,
                                    mask,
                                    args.vgg_path,
                                    args.adv_type,
                                    args.stage,
                                    args.lambda_style,
                                    args.lambda_cor,
                                    args.alpha,
                                    args.lambda_adv)
    else:
        g_vars, d_vars, adv_g_loss, adv_d_loss, rec_loss, correlation_loss, style_loss \
            = build_graph_with_loss(image,
                                    args.batch_size,
                                    mask,
                                    args.vgg_path,
                                    args.adv_type,
                                    args.stage,
                                    args.lambda_style,
                                    args.lambda_cor,
                                    args.alpha,
                                    args.lambda_adv)

        d_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                             beta1=args.beta1, beta2=args.beta2) \
            .minimize(adv_d_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate,
                                         beta1=args.beta1, beta2=args.beta2) \
        .minimize(adv_g_loss, var_list=g_vars)

    # Create and run variables intialization op
    init_op = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())  # tf.initialize_all_variables()
    sess.run(init_op)

    # Restore generator weights
    if args.pretrained_model:
        try:
            print(
                f"Restoring generator weights from checkpoint: {args.pretrained_model}")
            saver_pre = tf.train.Saver(g_vars)
            saver_pre.restore(sess, args.pretrained_model)
            print(" - [x] success")
        except ValueError as e:
            print(" - [!] error")
            print(str(e))

    summarywriter = tf.summary.FileWriter(args.output + '/summary',
                                          tf.get_default_graph())
    merge = tf.summary.merge_all()

    # Create dataset
    fnames = glob.glob(args.train_data_path + '/**/*', recursive=True)
    ds = tf.data.Dataset \
        .from_tensor_slices(fnames) \
        .shuffle(len(fnames)) \
        .map(tf.io.read_file,
             num_parallel_calls=args.workers) \
        .map(partial(tf.image.decode_jpeg, channels=3),
             num_parallel_calls=args.workers) \
        .map(partial(tf.image.resize_images, size=[args.height, args.width]),
             num_parallel_calls=args.workers) \
        .batch(args.batch_size)

    # Training
    low = 0
    high = args.height
    num = args.mask_area
    saver = tf.train.Saver(max_to_keep=20)
    it = 0

    for j in tqdm(range(0, args.epoch), desc=f"Stage #{args.stage}"):
        iterator = tf.data.make_one_shot_iterator(ds).get_next()

        while True:
            try:
                image_ = sess.run(iterator)

                if args.mask_type == 0:
                    mask_ = np.stack([discontinuous_mask(args.height, args.width, num, low, high)
                                      for _ in range(len(image_))],
                                     axis=0)
                else:
                    mask_ = np.stack([continuous_mask(args.height, args.width, num, 360, 32 * 3, 50 * 3)
                                      for _ in range(len(image_))],
                                     axis=0)

                if args.stage == 0:
                    _, g, rec, closs, sloss = sess.run(
                        [
                            g_optimizer,
                            adv_g_loss,
                            rec_loss,
                            correlation_loss,
                            style_loss
                        ],
                        feed_dict={
                            image: image_,
                            mask: mask_,
                        })
                else:
                    _, _, g, d, rec, closs, sloss = sess.run(
                        [
                            g_optimizer,
                            d_optimizer,
                            adv_g_loss,
                            adv_d_loss,
                            rec_loss,
                            correlation_loss,
                            style_loss
                        ],
                        feed_dict={
                            image: image_,
                            mask: mask_,
                        })
                it += 1

                if it % 100 == 0:
                    summary = sess.run(merge,
                                       feed_dict={
                                           image: image_,
                                           mask: mask_
                                       })
                    summarywriter.add_summary(summary, it)

                if it % 20 == 0:
                    if args.stage == 0:
                        tqdm.write(
                            '[{:07d}] rec_loss: {:9.1f} correlation_loss:{:9.1f} style_loss: {:9.1f}'.format(it,
                                                                                                             rec,
                                                                                                             closs,
                                                                                                             sloss))
                    else:
                        tqdm.write(
                            '[{:07d}] g_loss: {:9.1f} rec_loss: {:9.1f} correlation_loss:{:9.1f} style_loss: {:9.1f} d_loss: {:9.1f}'.format(it,
                                                                                                                                             g,
                                                                                                                                             rec,
                                                                                                                                             closs,
                                                                                                                                             sloss,
                                                                                                                                             d))

                if it % 1000 == 0:
                    saver.save(sess, args.output + '/model/model.ckpt',
                               global_step=it)
            except tf.errors.OutOfRangeError:
                break

        saver.save(sess, args.output + f'/model/model-ep{j}.ckpt',
                   global_step=it)
