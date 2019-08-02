# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf

from bert_transformer_vae_for_PAGE import VaeModel

from data_load import get_batch_for_train_or_dev_or_test,saveForTfRecord
from utils import save_hparams, get_hypotheses
import os
from hparams import Hparams
import logging
os.environ['CUDA_VISIBLE_DEVICES']= '5'
logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.PAGEdir)


logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")
logging.info("# 许海明提醒你： 这里需要准备tfRecord")



saveForTfRecord(hp.test,
                hp.maxlen_vae_Encoder,
                hp.maxlen_vae_Decoder_en,
                hp.maxlen_vae_Decoder_de,
                hp.vocab,
                output_file="./data/PAGE/test.tf_record",
                is_test=False)


test_batches, num_test_batches, num_test_samples = get_batch_for_train_or_dev_or_test(hp.test,
                                                                                      hp.maxlen_vae_Encoder,
                                                                                      hp.maxlen_vae_Decoder_en,
                                                                                      hp.maxlen_vae_Decoder_de,
                                                                                      hp.test_batch_size,
                                                                                      input_file="./data/PAGE/test.tf_record" ,
                                                                                      is_training=False,
                                                                                      is_test= False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
features=iter.get_next()

xs = (features["input_ids_vae_encoder"], features["input_masks_vae_encoder"], features["segment_ids_vae_encoder"], features["sent1"], features["sent2"])
ys = (features["input_ids_vae_decoder_enc"], features["input_ids_vae_decoder_dec"], features["output_ids_vae_decoder_dec"], features["sent1"], features["sent2"])


test_init_op = iter.make_initializer(test_batches)


logging.info("# Load model")
m = VaeModel(hp)

y_hat, _ = m.eval(xs, ys, mode="TPAGE")
# y_hat = m.infer(xs, ys)













with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.PAGEdir)
    saver = tf.train.Saver(max_to_keep=hp.num_epochs_LM)
    saver.restore(sess, ckpt)
    sess.run(test_init_op)


    logging.info("# get hypotheses")
    hypotheses = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat, m.idx2token)

    logging.info("# write results")
    model_output = "test_2.txt"
    translation = os.path.join(hp.PAGEdir,model_output)
    with open(translation, mode='w',encoding="utf-8") as fout:
        fout.write("\n".join(hypotheses))




logging.info("Done")
