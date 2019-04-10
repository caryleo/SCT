# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
FILENAME:       train.py
BY:             Gary 2019.3.12
LAST MODIFIED:  2019.3.12
DESCRIPTION:    train core file
"""

import logging
import os
import time

from six.moves import cPickle
import torch
import torch.optim as optim

import utils.Loss
from tool.dataloader import DataLoader
import models
import utils.utils as utils
import eval_utils as eval_utils

try:
    import tensorflow as tf
except ImportError:
    logging.warning("Tensorflow is not installed, no Tensorboard logging.")
    tf = None


def add_summary_value(writer, key, value, iteration):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)


def train(opts, device):
    # load data
    logging.info("STAGE 0: Loading data")
    logging.info("Loading data")
    loader = DataLoader(opts)
    logging.info("Load data complete")
    opts.vocabulary_size = loader.vocabulary_size
    opts.max_caption_length = loader.max_caption_length
    opts.nouns_size = loader.nouns_size
    logging.info("The vocabulary size: %d" % opts.vocabulary_size)
    logging.info("The nouns size: %d" % opts.nouns_size)
    logging.info("Max caption length: %d" % opts.max_caption_length)

    # tensorflow summary
    logging.info("Preparing TensorBoard logging")
    tf_summary_writer = tf and tf.summary.FileWriter(opts.checkpoint_path)

    # 准备处理信息块和结果历史
    info = dict()
    history = dict()
    if opts.start_from is not None:
        logging.info("Starting from checkpoint")
        # open old info and check if models are compatible 存在既有的记录，读取之
        with open(os.path.join(opts.start_from, 'info_' + opts.train_id + '.pkl'), 'rb') as info_file:
            info = cPickle.load(info_file)
            saved_model_opts = info['opts']  # 保存的参数
            entries = ["caption_model", "rnn_type", "rnn_size", "num_layers"]  # 这几个参数要求一致，不然加载模型会出错
            for entry in entries:
                assert vars(saved_model_opts)[entry] == vars(opts)[
                    entry], "Command line argument and saved model disagree on '%s' " % entry
        if os.path.isfile(os.path.join(opts.start_from, 'history_' + opts.train_id + '.pkl')):
            with open(os.path.join(opts.start_from, 'history_' + opts.train_id + '.pkl'), 'rb') as history_file:
                history = cPickle.load(history_file)
        logging.info("Load checkpoint complete")
    # load iter and epoch from info_file
    iteration = info.get('iter', 0)
    epoch = info.get('epoch', 0)
    logging.info("Starting from: iter %6d - epoch %3d" % (iteration, epoch))

    # load results, losses, lr, and ss_prob from history_file 历史部分的主要内容：loss，lr，ssprob和val
    loss_history = history.get('loss_history', dict())
    lr_history = history.get('lr_history', dict())
    ss_prob_history = history.get('ss_prob_history', dict())
    val_result_history = history.get('val_result_history', dict())

    # load iterators, split_index and best score (if existing), or use the dataloader
    loader.iterators = info.get('iterators', loader.iterators)
    loader.split_index = info.get('split_index', loader.split_index)
    if opts.load_best_score == 1:
        best_val_score = info.get('best_val_score', None)

    # 加载制定的模型
    logging.info("Using model: %s" % opts.caption_model)
    model = models.setup(opts)  # 模型的加载在模型部分完成了
    model.to(device=device)

    # 是否更新学习率
    update_lr_flag = True

    # Assure in training mode
    model.train()

    # 语言对数似然损失，Adam优化
    criterion = utils.Loss.LanguageModelCriterion()
    optimizer = optim.Adam(model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay)

    # Load the optimizer 如果存在数据，加载优化数据
    if vars(opts).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opts.start_from, 'optimizer.pth')))

    logging.info("STAGE 1: Training language model")
    logging.info("Start training")
    while True:
        # update learning rate, including lr_decay and schedule_sample 这部分暂时跳过，有关学习率调整的，先放下
        if update_lr_flag:
            # Assign the learning rate
            if epoch > opts.learning_rate_decay_start >= 0:
                frac = (epoch - opts.learning_rate_decay_start) // opts.learning_rate_decay_every
                decay_factor = opts.learning_rate_decay_rate ** frac
                opts.current_lr = opts.learning_rate * decay_factor
                # set the decayed rate
                utils.set_lr(optimizer, opts.current_lr)
            else:
                opts.current_lr = opts.learning_rate

            # Assign the scheduled sampling prob
            if epoch > opts.scheduled_sampling_start >= 0:
                frac = (epoch - opts.scheduled_sampling_start) // opts.scheduled_sampling_increase_every
                opts.ss_prob = min(opts.scheduled_sampling_increase_prob * frac, opts.scheduled_sampling_max_prob)
                model.ss_prob = opts.ss_prob7
            update_lr_flag = False

        start_time = time.time()
        # Load data from train split (0) 获取一个batch
        data = loader.get_batch('train')
        logging.debug("Reading data time: %.3fs" % float(time.time() - start_time))

        torch.cuda.synchronize()

        start_time = time.time()

        # 取出 fc att cap mask四项，这个时候已经对齐了，不用管太多
        tmp = [data['fc_feats'], data['att_feats'], data['captions'], data['masks']]
        tmp = [torch.from_numpy(_).to(device=device) for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        logging.debug("FC Features shape: %s", fc_feats.shape.__str__())
        logging.debug("ATT Features shape: %s", att_feats.shape.__str__())
        logging.debug("Labels shape: %s", labels.shape.__str__())
        logging.debug("Masks shape: %s", masks.shape.__str__())

        # 训练诶嘿嘿
        optimizer.zero_grad()
        # start from 1, 0 as START token没， 这里在算loss的时候做了一个特殊操作，就是把BOS去掉了
        loss = criterion(model(fc_feats, att_feats, labels), labels[:, 1:], masks[:, 1:])
        loss.backward()

        # 对梯度之
        utils.clip_gradient(optimizer, opts.grad_clip)
        optimizer.step()

        train_loss = loss.item()

        torch.cuda.synchronize()
        end_time = time.time()

        logging.debug(
            "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(iteration,
                                                                                  epoch,
                                                                                  train_loss,
                                                                                  end_time - start_time))
        if iteration % 1000 == 0:
            logging.debug(
                "iter {} (epoch {}), train_loss = {:.3f}".format(iteration,
                                                                 epoch,
                                                                 train_loss))

        # Update the iteration and epoch (if wrapped) 判断epoch的位置！！！
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary 特定iter间隔之后写入日志
        logging.debug("Writing loss summary log")
        if iteration % opts.losses_log_every == 0:
            if tf is not None:
                add_summary_value(tf_summary_writer, 'train_loss', train_loss, iteration)
                add_summary_value(tf_summary_writer, 'learning_rate', opts.current_lr, iteration)
                add_summary_value(tf_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
                tf_summary_writer.flush()
            # 写入历史文件
            loss_history[iteration] = train_loss
            lr_history[iteration] = opts.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model 这里是eval的部分，保存模型的位置！！！
        if iteration % opts.save_checkpoint_every == 0:
            logging.info("Start validation")
            # eval model
            eval_kwargs = {'split': 'val',
                           'dataset': opts.input_json,
                           'device': device}
            eval_kwargs.update(vars(opts))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, criterion, loader, eval_kwargs)

            # Write validation result into summary
            if tf is not None:
                add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
                for k, v in lang_stats.items():
                    add_summary_value(tf_summary_writer, k, v, iteration)
                tf_summary_writer.flush() # 刷新？？？

            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model if is improving on validation result 如果取language，就以CIDEr为准，否则是loss
            if opts.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True

            # 写入检查点
            checkpoint_path = os.path.join(opts.checkpoint_path, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logging.info("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opts.checkpoint_path, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscellaneous information
            info['iter'] = iteration
            info['epoch'] = epoch
            info['iterators'] = loader.iterators
            info['split_index'] = loader.split_index
            info['best_val_score'] = best_val_score
            info['opts'] = opts
            info['vocabulary'] = loader.get_vocab()

            history['val_result_history'] = val_result_history
            history['loss_history'] = loss_history
            history['lr_history'] = lr_history
            history['ss_prob_history'] = ss_prob_history

            with open(os.path.join(opts.checkpoint_path, 'info_' + opts.train_id + '.pkl'), 'wb') as infofile:
                cPickle.dump(info, infofile)
            with open(os.path.join(opts.checkpoint_path, 'history_' + opts.train_id + '.pkl'), 'wb') as historyfile:
                cPickle.dump(history, historyfile)
            logging.info("Checkpoint Saved")

            # 选择的最佳模型
            if best_flag:
                checkpoint_path = os.path.join(opts.checkpoint_path, 'model-best.pth')
                torch.save(model.state_dict(), checkpoint_path)
                logging.info("model saved to {}".format(checkpoint_path))
                with open(os.path.join(opts.checkpoint_path, 'info_' + opts.train_id + '-best.pkl'), 'wb') as bestfile:
                    cPickle.dump(info, bestfile)

            logging.info("validation complete")
        # Stop if reaching max epochs
        if epoch >= opts.epoch_num != -1:
            break

    # final round!
    logging.info("Start validation")
    # eval model
    eval_kwargs = {'split': 'val',
                   'dataset': opts.input_json,
                   'device': device}
    eval_kwargs.update(vars(opts))
    val_loss, predictions, lang_stats = eval_utils.eval_split(model, criterion, loader, eval_kwargs)

    # Write validation result into summary
    if tf is not None:
        add_summary_value(tf_summary_writer, 'validation loss', val_loss, iteration)
        for k, v in lang_stats.items():
            add_summary_value(tf_summary_writer, k, v, iteration)
        tf_summary_writer.flush()

    val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

    # Save model if is improving on validation result
    if opts.language_eval == 1:
        current_score = lang_stats['CIDEr']
    else:
        current_score = - val_loss

    best_flag = False
    if best_val_score is None or current_score > best_val_score:
        best_val_score = current_score
        best_flag = True

    checkpoint_path = os.path.join(opts.checkpoint_path, 'model.pth')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opts.checkpoint_path, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)

    # Dump miscellaneous information
    info['iter'] = iteration
    info['epoch'] = epoch
    info['iterators'] = loader.iterators
    info['split_index'] = loader.split_index
    info['best_val_score'] = best_val_score
    info['opts'] = opts
    info['vocabulary'] = loader.get_vocab()

    history['val_result_history'] = val_result_history
    history['loss_history'] = loss_history
    history['lr_history'] = lr_history
    history['ss_prob_history'] = ss_prob_history

    with open(os.path.join(opts.checkpoint_path, 'info_' + opts.train_id + '.pkl'), 'wb') as infofile:
        cPickle.dump(info, infofile)
    with open(os.path.join(opts.checkpoint_path, 'history_' + opts.train_id + '.pkl'), 'wb') as historyfile:
        cPickle.dump(history, historyfile)
    logging.info("Checkpoint Saved")

    if best_flag:
        checkpoint_path = os.path.join(opts.checkpoint_path, 'model-best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        logging.info("model saved to {}".format(checkpoint_path))
        with open(os.path.join(opts.checkpoint_path, 'info_' + opts.train_id + '-best.pkl'), 'wb') as bestfile:
            cPickle.dump(info, bestfile)

    logging.info("validation complete")
    logging.info("training complete")

    logging.info("STAGE 2: Extracting memory")

    logging.info("STAGE 3: Training relation model")
