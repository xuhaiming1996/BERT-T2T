# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
from bert import bert_model_for_PAGE
from bert_transformer_vae_for_PAGE import VaeModel
from tqdm import tqdm
from data_load import get_batch_for_train_or_dev_or_test,saveForTfRecord,load_vocab
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging
os.environ['CUDA_VISIBLE_DEVICES']= '5'
logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.PAGEdir)

logging.info("# Prepare train/eval batches")

logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")


saveForTfRecord(hp.eval,
                hp.maxlen_vae_Encoder,
                hp.maxlen_vae_Decoder_en,
                hp.maxlen_vae_Decoder_de,
                hp.vocab,
                output_file="./data/PAGE/eval.tf_record",
                is_test=False)


eval_batches, num_eval_batches, num_eval_samples = get_batch_for_train_or_dev_or_test(hp.eval,
                                                                                      hp.maxlen_vae_Encoder,
                                                                                      hp.maxlen_vae_Decoder_en,
                                                                                      hp.maxlen_vae_Decoder_de,
                                                                                      hp.eval_batch_size,
                                                                                      input_file="./data/PAGE/eval.tf_record" ,
                                                                                      is_training=False,
                                                                                      is_test= False)


# saveForTfRecord(hp.train,
#                 hp.maxlen_vae_Encoder,
#                 hp.maxlen_vae_Decoder_en,
#                 hp.maxlen_vae_Decoder_de,
#                 hp.vocab,
#                 output_file="./data/PAGE/train.tf_record",
#                 is_test=False)



train_batches, num_train_batches, num_train_samples =  get_batch_for_train_or_dev_or_test(hp.train,
                                                                                      hp.maxlen_vae_Encoder,
                                                                                      hp.maxlen_vae_Decoder_en,
                                                                                      hp.maxlen_vae_Decoder_de,
                                                                                      hp.batch_size,
                                                                                      input_file="./data/PAGE/train.tf_record" ,
                                                                                      is_training=True,
                                                                                      is_test= False)





token2idx, idx2token = load_vocab(hp.vocab)





# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
features=iter.get_next()

xs = (features["input_ids_vae_encoder"], features["input_masks_vae_encoder"], features["segment_ids_vae_encoder"], features["sent1"], features["sent2"])
ys = (features["input_ids_vae_decoder_enc"], features["input_ids_vae_decoder_dec"], features["output_ids_vae_decoder_dec"], features["sent1"], features["sent2"])

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)


logging.info("# Load model")
m = VaeModel(hp)
loss, train_op, global_step, train_summaries, print_demo = m.train(xs, ys, mode="TPAGE")
y_hat, eval_summaries = m.eval(xs, ys, mode="PPAGE")
# y_hat = m.infer(xs, ys)





tvars = tf.trainable_variables()  #这里是可训练的变量
initialized_variable_names = {}
if hp.init_checkpoint_bert and hp.init_checkpoint_PAGE is None:
    (assignment_map, initialized_variable_names
     ) = bert_model_for_PAGE.get_assignment_map_from_checkpoint(tvars, hp.init_checkpoint_bert)
    tf.train.init_from_checkpoint(hp.init_checkpoint_bert, assignment_map)


for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT_BERT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
# 许海明 添加结束






logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs_PAGE)





with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.PAGEdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.PAGEdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.PAGEdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs_PAGE * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary,demo = sess.run([train_op, global_step, train_summaries,print_demo])

        if _gs and _gs % 1000 == 0:
            print("*" * 100)
            print("输入是：", " ".join([m.idx2token[int(id)] for id in demo[0]]))
            print("目标是：", " ".join([m.idx2token[int(id)] for id in demo[1]]))
            print("预测是："," ".join([m.idx2token[int(id)]  for  id in demo[2]]))
            print("\n\n")


        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % 50000 == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss)  # train loss

            logging.info("# save models")
            model_output = "PAGE.ckpt"
            ckpt_name = os.path.join(hp.PAGEdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
