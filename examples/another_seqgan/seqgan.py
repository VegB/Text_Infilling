import tensorflow as tf
import numpy as np
import importlib
from shutil import copyfile

import texar as tx

from generator import Generator
from discriminator import Discriminator
from dataloader import GenDataLoader, DisDataLoader
from utils import print_result, store_output, pad_to_length

config_path = "config"
config = importlib.import_module(config_path)
log = open(config.log_file, "a+")

opt_vars = {
    'learning_rate': config.init_lr,
    'min_learning_rate': config.min_lr,
    'best_valid_ppl': 1e100,
    'steps_not_improved': 0
}


def pretrain_generator(sess, generator, gen_dataloader, valid_dataloader):
    print("-------------Pretrain Generator----------------")
    gen_dataloader.reset()
    loss = 0.
    iters = 0
    while not gen_dataloader.should_stop():
        _, step, mle_loss, outputs = sess.run([generator.train_op, generator.global_step,
                                               generator.teacher_loss, generator.outputs],
                                              feed_dict={generator.data_batch: gen_dataloader.get_batch(),
                                                         generator.learning_rate: opt_vars['learning_rate'],
                                                         tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        # calculate current ppl
        loss += mle_loss
        iters += config.num_steps

        # calculate valid ppl, might change learning rate
        if step % 100 == 0 and opt_vars['learning_rate'] > opt_vars['min_learning_rate']:
            valid_ppl = calculate_ppl(sess, generator, valid_dataloader)
            if valid_ppl < opt_vars['best_valid_ppl']:
                opt_vars['best_valid_ppl'] = valid_ppl
                opt_vars['steps_not_improved'] = 0
            else:
                opt_vars['steps_not_improved'] += 1

            if opt_vars['steps_not_improved'] >= 30:
                opt_vars['steps_not_improved'] = 0
                opt_vars['learning_rate'] *= config.lr_decay
        elif opt_vars['learning_rate'] < opt_vars['min_learning_rate']:
            opt_vars['learning_rate'] = opt_vars['min_learning_rate']

        if step % 200 == 0:
            ppl = np.exp(loss / iters)
            print('global step:', step, ' ' * 4, 'training ppl:', ppl)
            print_result(outputs.sample_id[:config.print_num], gen_dataloader.id2word, gen_dataloader.max_len)

    return np.exp(loss / iters)


def generate_negative_samples(sess, generator, gen_dataloader, dst_path):
    print("-------------Generate Negative Samples----------------")
    gen_dataloader.reset()
    generated_outputs = []
    while not gen_dataloader.should_stop():
        decode_output = sess.run(generator.generated_outputs,
                                 feed_dict={generator.data_batch: gen_dataloader.get_batch(),
                                            tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        generated_outputs.extend(decode_output.sample_id)

    store_output(output=generated_outputs, id2word=gen_dataloader.id2word,
                 data_path=dst_path, max_len=gen_dataloader.max_len)
    print_result(generated_outputs[:config.print_num], gen_dataloader.id2word, gen_dataloader.max_len)


def train_discriminator(sess, discriminator, epoch_num):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, epoch_num=epoch_num, positive_file=config.train_file,
                               negative_file=config.negative_file, vocab_file=config.vocab_file)

    while not dataloader.should_stop():
        r_ids, g_ids = dataloader.get_batch()

        _, step, loss = sess.run([discriminator.train_op, discriminator.global_step,
                                  discriminator.total_loss],
                                 feed_dict={discriminator.real_samples: r_ids,
                                            discriminator.gen_samples: g_ids,
                                            discriminator.global_step: dataloader.step,
                                            tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 200 == 0:
            print("%d: dis_total_loss: %.6f" % (step, loss))


def update_generator(sess, generator, discriminator, gen_dataloader, dis_dataloader):
    print("-------------Update Generator----------------")
    loss = 0.
    iters = 0
    for i in range(config.g_update_batch):
        if gen_dataloader.should_stop():
            gen_dataloader.reset()
        if dis_dataloader.should_stop():
            dis_dataloader.reset()
        # Teacher forcing
        _, mle_loss, step = sess.run([generator.train_op, generator.teacher_loss, generator.global_step],
                                   feed_dict={generator.data_batch: gen_dataloader.get_batch(),
                                              generator.learning_rate: opt_vars['learning_rate'],
                                              tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        # calculate current ppl
        loss += mle_loss
        iters += config.num_steps
        ppl = np.exp(loss / iters)
        print('global step:', step, ' ' * 4, 'training ppl:', ppl)

        gen_data = sess.run(generator.generated_outputs,
                            feed_dict={tx.global_mode(): tf.estimator.ModeKeys.EVAL})
        g_data = [pad_to_length(sent, bos=dis_dataloader.bos_id, eos=dis_dataloader.eos_id,
                                pad=dis_dataloader.pad_id, max_len=dis_dataloader.max_len)
                  for sent in gen_data.sample_id]

        r_ids, _ = dis_dataloader.get_batch()

        _, g_preds = sess.run([discriminator.r_preds, discriminator.g_preds],
                              feed_dict={discriminator.real_samples: r_ids,
                                         discriminator.gen_samples: [line[1:] for line in g_data],
                                         discriminator.global_step: dis_dataloader.step,
                                         tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

        _, update_loss = sess.run([generator.update_op, generator.update_loss],
                                  feed_dict={generator.data_batch: g_data,
                                             generator.rewards: [preds[:-1] for preds in g_preds],
                                             generator.learning_rate: opt_vars['learning_rate'],
                                             tx.global_mode(): tf.estimator.ModeKeys.TRAIN})

    return np.exp(loss / iters)


def calculate_ppl(sess, generator, dataloader):
    loss = 0.
    iters = 0
    for i in range(30):
        if dataloader.should_stop():
            dataloader.reset()
        mle_loss = sess.run(generator.teacher_loss,
                            feed_dict={generator.data_batch: dataloader.get_batch(),
                                       generator.learning_rate: opt_vars['learning_rate'],
                                       tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        loss += mle_loss
        iters += config.num_steps
    return np.exp(loss / iters)


def record_ppl(sess, generator, valid_dataloader, test_dataloader, epoch_id, train_ppl, mode="Pretrain"):
    valid_ppl = calculate_ppl(sess, generator, valid_dataloader)
    test_ppl = calculate_ppl(sess, generator, test_dataloader)
    rst = "epoch %d(%s): learning_rate = %.7f, train_ppl = %f, valid_ppl = %f, test_ppl = %f\n" % \
          (epoch_id, mode, opt_vars["learning_rate"], train_ppl, valid_ppl, test_ppl)
    print(rst)
    log.write(rst)
    log.flush()

if __name__ == "__main__":
    gen_dataloader = GenDataLoader(config, text_file=config.train_file,
                               vocab_file=config.vocab_file, epoch_num=1)
    dis_dataloader = DisDataLoader(config, epoch_num=1, positive_file=config.train_file,
                               negative_file=config.train_file, vocab_file=config.vocab_file)
    valid_dataloader = GenDataLoader(config, text_file=config.valid_file,
                                     vocab_file=config.vocab_file, epoch_num=1)
    test_dataloader = GenDataLoader(config, text_file=config.test_file,
                                    vocab_file=config.vocab_file, epoch_num=1)

    generator = Generator(config, word2id=gen_dataloader.word2id, bos=gen_dataloader.bos_id,
                          eos=gen_dataloader.eos_id, pad=gen_dataloader.pad_id)
    discriminator = Discriminator(config, word2id=dis_dataloader.word2id)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        saver.restore(sess, config.ckpt + '-50')
        opt_vars['learning_rate'] = 0.000003
        opt_vars['min_learning_rate'] = config.min_lr

        for pre_epoch in range(51, config.generator_pretrain_epoch + 1):
            train_ppl = pretrain_generator(sess, generator, gen_dataloader, valid_dataloader)
            record_ppl(sess, generator, valid_dataloader, test_dataloader,
                       epoch_id=pre_epoch, train_ppl=train_ppl)
            if pre_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=pre_epoch)
   
        generate_negative_samples(sess, generator, gen_dataloader, dst_path=config.negative_file)
        
        train_discriminator(sess, discriminator, epoch_num=config.discriminator_pretrain_epoch)
        saver.save(sess, './checkpoint/pretrained/ckpt', global_step=80)

        for update_epoch in range(1, config.adversial_epoch + 1):
            train_ppl = update_generator(sess, generator, discriminator, gen_dataloader, dis_dataloader)
            generate_negative_samples(sess, generator, gen_dataloader, dst_path=config.negative_file)
            record_ppl(sess, generator, valid_dataloader, test_dataloader, epoch_id=update_epoch,
                       train_ppl=train_ppl, mode="Adversarial")
            train_discriminator(sess, discriminator, epoch_num=1)
            if update_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=update_epoch + config.generator_pretrain_epoch)
