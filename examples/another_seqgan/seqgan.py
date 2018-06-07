import tensorflow as tf
import numpy as np
import importlib

import texar as tx

from generator import Generator
from dataloader import GenDataLoader, DisDataLoader
from utils import print_result, store_output, pad_to_length, print_and_write_to_file

config_path = "config"
config = importlib.import_module(config_path)
log = open(config.log_file, "a+")
training_log = open(config.train_log_file, "w")
eval_log = open(config.eval_log_file, "w")


def pretrain_generator(sess, gen_dataloader, valid_dataloader, test_dataloader):
    loss = 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        "final_state": final_state,
        'global_step': global_step,
        'train_op': train_op
    }

    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        state = rets["final_state"]
        iters += config.num_steps

        ppl = np.exp(loss / iters)

        rst = "global step: %d, training ppl: %.6f\n" % (rets['global_step'], ppl)
        print_and_write_to_file(rst, training_log, print_out=False)

        if rets['global_step'] % 100 == 0:
            valid_ppl = calculate_ppl(sess, valid_dataloader)
            test_ppl = calculate_ppl(sess, test_dataloader)
            rst = "global step: %d, training ppl: %.6f, valid ppl: %.6f," \
                  " test ppl: %.6f, learning rate: %.7f\n" % \
                  (rets['global_step'], ppl, valid_ppl, test_ppl, opt_vars['learning_rate'])
            print_and_write_to_file(rst, eval_log)

            if valid_ppl < opt_vars['best_valid_ppl']:
                opt_vars['best_valid_ppl'] = valid_ppl
                opt_vars['steps_not_improved'] = 0
            else:
                opt_vars['steps_not_improved'] += 1

            if opt_vars['steps_not_improved'] >= 30:
                opt_vars['steps_not_improved'] = 0
                opt_vars['learning_rate'] *= config.lr_decay

    ppl = np.exp(loss / iters)
    return ppl


def generate_negative_samples(sess, gen_dataloader, dst_path):
    print("-------------Generate Negative Samples----------------")
    generated_outputs = []
    fetches = {
        'generated_sample_id': generated_sample_id
    }
    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
        }
        rets = sess.run(fetches, feed_dict)
        generated_outputs.extend(rets['generated_sample_id'])

    store_output(output=generated_outputs, id2word=gen_dataloader.id2word,
                 data_path=dst_path, max_len=config.num_steps)
    print_result(generated_outputs[:config.print_num], gen_dataloader.id2word, config.num_steps)


def train_discriminator(sess):
    print("-------------Train Discriminator----------------")
    dataloader = DisDataLoader(config, positive_file=config.train_file,
                               negative_file=config.train_file, word2id=word2id)

    for step, (r_ids, g_ids) in enumerate(dataloader.iter()):
        _, loss, r_loss_, f_loss_, r_logits_, real_label_ = sess.run([dis_train_op, dis_loss, r_loss, f_loss, r_logits, real_label],
                            feed_dict={batch_size: config.batch_size,
                                       real_samples: r_ids,
                                       fake_samples: g_ids,
                                       dis_global_step: step,
                                       tx.global_mode(): tf.estimator.ModeKeys.TRAIN})
        if step % 20 == 0:
            print("%d: dis_total_loss: %.6f" % (step, loss))
            print("r_loss: %f, f_loss: %f" % (r_loss_, f_loss_))
            # print(r_logits_)
            # print(real_label_)


def update_generator(sess, gen_dataloader):
    print("-------------Update Generator----------------")
    loss, epoch_update_loss = 0., 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        'update_loss': update_loss,
        "final_state": final_state,
        'global_step': global_step,
        'train_op': train_op,
        'update_op': update_op
    }

    for step, (x, y) in enumerate(gen_dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            update_learning_rate: opt_vars['update_learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        epoch_update_loss += rets['update_loss']
        state = rets["final_state"]
        iters += config.num_steps

        ppl = np.exp(loss / iters)

        rst = "global step: %d, training ppl: %.6f, update loss: %.6f\n" % \
              (rets['global_step'], ppl, rets['update_loss'])
        print_and_write_to_file(rst, training_log, print_out=False)

        if rets['global_step'] % 100 == 0:
            valid_ppl = calculate_ppl(sess, valid_dataloader)
            test_ppl = calculate_ppl(sess, test_dataloader)
            rst = "global step: %d, training ppl: %.6f, valid ppl: %.6f, " \
                  "test ppl: %.6f, update_loss: %.6f, learning rate: %.7f, " \
                  "update learning rate: %.7f\n" % \
                  (rets['global_step'], ppl, valid_ppl, test_ppl, epoch_update_loss/iters,
                   opt_vars['learning_rate'], opt_vars['update_learning_rate'])
            print_and_write_to_file(rst, eval_log)

            if valid_ppl < opt_vars['best_valid_ppl']:
                opt_vars['best_valid_ppl'] = valid_ppl
                opt_vars['steps_not_improved'] = 0
            else:
                opt_vars['steps_not_improved'] += 1

            if opt_vars['steps_not_improved'] >= 30:
                opt_vars['steps_not_improved'] = 0
                opt_vars['learning_rate'] *= config.lr_decay
                opt_vars['update_learning_rate'] *= config.lr_decay

    ppl = np.exp(loss / iters)
    return ppl


def calculate_ppl(sess, dataloader):
    loss = 0.
    iters = 0
    state = sess.run(initial_state, feed_dict={
        inputs: np.ones((config.batch_size, config.num_steps))})

    fetches = {
        "mle_loss": mle_loss,
        "final_state": final_state,
        'global_step': global_step,
    }

    for step, (x, y) in enumerate(dataloader.iter()):
        feed_dict = {
            batch_size: config.batch_size,
            inputs: x, targets: y,
            learning_rate: opt_vars['learning_rate'],
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
        }
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        rets = sess.run(fetches, feed_dict)
        loss += rets["mle_loss"]
        state = rets["final_state"]
        iters += config.num_steps
    ppl = np.exp(loss / iters)
    return ppl


if __name__ == "__main__":
    word2id = tx.data.make_vocab(config.train_file, newline_token="<EOS>", return_type="dict")
    vocab_size = len(word2id)
    gen_dataloader = GenDataLoader(config, config.train_file, word2id)
    valid_dataloader = GenDataLoader(config, config.valid_file, word2id)
    test_dataloader = GenDataLoader(config, config.test_file, word2id)

    generator = Generator(vocab_size=vocab_size)
    discriminator = tx.modules.UnidirectionalRNNClassifier(hparams={"clas_strategy": "time_wise", "num_classes": 1})
    saver = tf.train.Saver()

    # ------------Pretrain Generator---------------
    batch_size = tf.placeholder(dtype=tf.int32, shape=())
    num_steps = config.num_steps

    opt_vars = {
        'learning_rate': config.init_lr,
        'update_learning_rate': config.update_lr,
        'best_valid_ppl': 1e100,
        'steps_not_improved': 0
    }

    inputs = tf.placeholder(tf.int32, [None, num_steps])
    targets = tf.placeholder(tf.int32, [None, num_steps])

    initial_state, gen_logits, final_state, sample_id, sequence_length = \
        generator(text_ids=inputs, num_steps=config.num_steps * tf.ones((batch_size,), dtype=tf.int32))

    # Losses & train ops
    mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
        labels=targets,
        logits=gen_logits,
        sequence_length=num_steps * tf.ones((batch_size,)))

    l2_loss = sum([tf.nn.l2_loss(t) for t in tf.trainable_variables()])

    # Use global_step to pass epoch, for lr decay
    global_step = tf.Variable(0, dtype=tf.int32)
    learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    train_op = optimizer.minimize(
        mle_loss + config.l2_decay * l2_loss, global_step=global_step)

    # -------------Generate Text-------------------
    generated_outputs, _, _ = generator.decoder(
        decoding_strategy="infer_sample",
        start_tokens=inputs[:, 0],
        end_token=word2id["<EOS>"],
        embedding=generator.embedding_matrix,
        initial_state=initial_state,
        max_decoding_length=num_steps)
    generated_logits = generator.output_layer(generated_outputs.logits)
    generated_sample_id = tf.argmax(generated_logits, 2)

    # --------------Pretrain Discriminator-----------------
    embedder = tx.modules.WordEmbedder(vocab_size=vocab_size, hparams=config.emb)

    dis_global_step = tf.Variable(0, dtype=tf.int32)
    real_samples = tf.placeholder(tf.int32, [None, num_steps])
    fake_samples = tf.placeholder(tf.int32, [None, num_steps])

    r_logits, r_preds = discriminator(embedder(real_samples))
    f_logits, f_preds = discriminator(embedder(fake_samples))
    real_label = tf.Variable(
        np.ones(shape=(config.batch_size, config.num_steps), dtype=np.float32),
        dtype=tf.float32)
    fake_label = tf.Variable(
        np.zeros(shape=(config.batch_size, config.num_steps), dtype=np.float32),
        dtype=tf.float32)

    r_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.ones((config.batch_size, config.num_steps), dtype=tf.float32),
        logits=tf.squeeze(r_logits),
        sequence_length=num_steps * tf.ones((batch_size,)))  # r_preds -> 1.
    r_loss.set_shape(())
    f_loss = tx.losses.sequence_sigmoid_cross_entropy(
        labels=tf.zeros((config.batch_size, config.num_steps), dtype=tf.float32),
        logits=tf.squeeze(f_logits),
        sequence_length=num_steps * tf.ones((batch_size,)))  # g_preds -> 0.
    f_loss.set_shape(())
    dis_loss = r_loss + f_loss

    dis_train_op = tx.core.get_train_op(
        dis_loss, global_step=global_step, increment_global_step=False,
        hparams=config.d_opt)

    # ----------------Adversarial------------------
    reward_logits, reward_preds = discriminator(inputs=embedder(sample_id), sequence_length=sequence_length)
    rewards = reward_logits * tf.one_hot(reward_preds, 2, 1.0, 0.0)

    preds = tf.nn.softmax(gen_logits)
    print(sequence_length.get_shape())
    reward = tx.losses.discount_reward(reward=tf.reduce_sum(rewards, 2), sequence_length=sequence_length)#um_steps * tf.ones((batch_size,)))
    update_loss = -tf.reduce_mean(tf.log(preds) * reward)

    update_step = tf.Variable(0, dtype=tf.int32)
    update_learning_rate = \
        tf.placeholder(dtype=tf.float32, shape=(), name='update_learning_rate')
    update_optimizer = tf.train.AdamOptimizer(
        learning_rate=update_learning_rate,
        beta1=0.,
        beta2=0.999,
        epsilon=1e-9)
    update_op = update_optimizer.minimize(update_loss, global_step=global_step)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        # for pre_epoch in range(1, config.generator_pretrain_epoch + 1):
        #     train_ppl = pretrain_generator(sess, gen_dataloader, valid_dataloader, test_dataloader)
        #     if pre_epoch % 10 == 0:
        #         saver.save(sess, config.ckpt, global_step=pre_epoch)
        #
        # generate_negative_samples(sess, gen_dataloader, dst_path=config.negative_file)

        for dis_epoch in range(config.discriminator_pretrain_epoch):
            train_discriminator(sess)
        
        opt_vars['learning_rate'] = config.update_init_lr if config.update_init_lr > opt_vars['learning_rate'] else opt_vars['learning_rate']

        for update_epoch in range(1, config.adversial_epoch + 1):
            train_ppl = update_generator(sess, gen_dataloader)
            generate_negative_samples(sess, gen_dataloader, dst_path=config.negative_file)
            train_discriminator(sess)
            if update_epoch % 10 == 0:
                saver.save(sess, config.ckpt, global_step=update_epoch + config.generator_pretrain_epoch)
