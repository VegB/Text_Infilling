import os
import tensorflow as tf
import texar as tx
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def prepare_data(FLAGS, config, train_path):
    """Download the PTB or Yahoo dataset
    """
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    ptb_url = 'https://jxhe.github.io/download/ptb_data.tgz'
    coco_url = 'https://VegB.github.io/downloads/coco_data.tgz'

    data_path = FLAGS.data_path

    if not tf.gfile.Exists(train_path):
        url = ptb_url if FLAGS.dataset == 'ptb' else coco_url
        tx.data.maybe_download(url, data_path, extract=True)
        os.remove('%s_data.tgz' % FLAGS.dataset)

        data_path = os.path.join(data_path, '%s_data' % FLAGS.dataset)

        train_path = os.path.join(data_path, "%s.train.txt" % FLAGS.dataset)
        valid_path = os.path.join(data_path, "%s.valid.txt" % FLAGS.dataset)
        test_path = os.path.join(data_path, "%s.test.txt" % FLAGS.dataset)
        vocab_path = os.path.join(data_path, "vocab.txt")

        config.train_data_hparams['dataset'] = {'files': train_path,
                                                'vocab_file': vocab_path}

        config.val_data_hparams['dataset'] = {'files': valid_path,
                                              'vocab_file': vocab_path}

        config.test_data_hparams['dataset'] = {'files': test_path,
                                               'vocab_file': vocab_path}


def _draw_log(config, epoch, loss_list):
    plt.figure(figsize=(14, 10))
    plt.plot(loss_list, '--', linewidth=1, label='loss trend')
    plt.ylabel('training loss till epoch {}'.format(epoch))
    plt.xlabel('every 50 steps, present_rate=%f' % config.present_rate)
    plt.savefig(config.log_dir + '/img/train_loss_curve.png')
