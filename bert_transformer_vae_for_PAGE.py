from bert import bert_model_for_PAGE
from bert.bert_model_for_PAGE import get_shape_list,layer_norm
import tensorflow as tf
from modules import ff,  multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging

from data_load import load_vocab

class VaeModel:

    '''
    这是模型的主题框架
    '''
    def __init__(self,hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)


    # 变积分自动编码器的编码器
    def encoder_vae(self, xs, training, mode):

        '''
        :param xs:
        :param training:
        :param mode: TPAGE表示训练复述模型     LM 表示训练transformer的解码器  PPAGE：代表预测的时候
        return:
        '''

        input_ids_vae_encoder, input_masks_vae_encoder, segment_ids_vae_encoder, sents1, sents2 = xs

        if mode == "TPAGE" or mode == "PPAGE":
            # 这是整体训练 vae
            encoder_vae = bert_model_for_PAGE.BertModel(
                config=self.hp,
                is_training=training,
                input_ids=input_ids_vae_encoder,
                input_mask=input_masks_vae_encoder,
                token_type_ids=segment_ids_vae_encoder)

            self.embeddings = encoder_vae.get_embedding_table()
            self.full_position_embeddings = encoder_vae.get_full_position_embeddings()
            pattern_para = tf.squeeze(encoder_vae.get_sequence_output()[:, 0:1, :], axis=1)
            gaussian_params = tf.layers.dense(pattern_para, 2 * self.hp.z_dim)
            mean = gaussian_params[:, :self.hp.z_dim]
            stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.hp.z_dim:])
        else:
            raise("您好，你选择的mode 出现错误")

        return mean, stddev



    # 变积分自动编码器的解码器
    def decoder_vae(self, ys, z, training, mode):
        input_ids_vae_decoder_enc, input_ids_vae_decoder_dec,output_ids_vae_decoder_dec,  sents1, sents2 = ys
        if mode == "TPAGE" or mode == "PPAGE":
            with tf.variable_scope("decoder_enc_vae", reuse=tf.AUTO_REUSE):
                enc = tf.nn.embedding_lookup(self.embeddings, input_ids_vae_decoder_enc)  # (N, T1, d_model)
                input_shape = get_shape_list(enc, expected_rank=3)
                seq_length = input_shape[1]
                width = input_shape[2]
                output = enc
                assert_op = tf.assert_less_equal(seq_length, self.hp.max_position_embeddings)
                with tf.control_dependencies([assert_op]):
                    position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                                   [seq_length, -1])
                    num_dims = len(output.shape.as_list())
                    position_broadcast_shape = []
                    for _ in range(num_dims - 2):
                        position_broadcast_shape.append(1)
                    position_broadcast_shape.extend([seq_length, width])
                    position_embeddings = tf.reshape(position_embeddings,
                                                     position_broadcast_shape)

                    output += position_embeddings  # 添加位置信息

                enc = output
                ## Blocks
                for i in range(self.hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # self-attention
                        enc = multihead_attention(seq_k=input_ids_vae_decoder_enc,
                                                  seq_q=input_ids_vae_decoder_enc,
                                                  queries=enc,
                                                  keys=enc,
                                                  values=enc,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=training,
                                                  causality=False,
                                                  scope="self_attention")
                        # feed forward
                        enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])


            memory = enc


            # 下面是VAE的decoder中的 decoder
            with tf.variable_scope("decoder_dec_vae", reuse=tf.AUTO_REUSE):
                dec = tf.nn.embedding_lookup(self.embeddings, input_ids_vae_decoder_dec)  # (N, T, d_model)
                input_shape = get_shape_list(dec, expected_rank=3)
                seq_length = input_shape[1]
                width = input_shape[2]
                output = dec
                assert_op = tf.assert_less_equal(seq_length, self.hp.max_position_embeddings)
                with tf.control_dependencies([assert_op]):
                    position_embeddings = tf.slice(self.full_position_embeddings, [0, 0],
                                                   [seq_length, -1])
                    num_dims = len(output.shape.as_list())
                    position_broadcast_shape = []
                    for _ in range(num_dims - 2):
                        position_broadcast_shape.append(1)
                    position_broadcast_shape.extend([seq_length, width])
                    position_embeddings = tf.reshape(position_embeddings,
                                                     position_broadcast_shape)


                    output += position_embeddings                          # 添加位置信息

                dec=output

                # 在这里加上 规则模式 采用的是concate的方式!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                z = tf.expand_dims(z, axis=1)
                z = tf.tile(z, multiples=[1, seq_length, 1])
                dec = tf.concat([dec, z], axis=-1)
                dec = tf.layers.dense(dec, self.hp.d_model)
                # Blocks
                for i in range(self.hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                        # Masked self-attention (Note that causality is True at this time)
                        dec = multihead_attention(seq_k=input_ids_vae_decoder_dec,
                                                  seq_q=input_ids_vae_decoder_dec,
                                                  queries=dec,
                                                  keys=dec,
                                                  values=dec,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=training,
                                                  causality=True,
                                                  scope="self_attention")

                        # Vanilla attention
                        dec = multihead_attention(seq_k=input_ids_vae_decoder_enc,
                                                  seq_q=input_ids_vae_decoder_dec,
                                                  queries=dec,
                                                  keys=memory,
                                                  values=memory,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=training,
                                                  causality=False,
                                                  scope="vanilla_attention")
                        ### Feed Forward
                        dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])



        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (d_model, vocab_size)
        # 这里为了适应迎合 强制加上一个 FF
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # (N, T2, vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))
        return logits, y_hat, output_ids_vae_decoder_dec, sents2



    def train(self, xs, ys,mode):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''
        # forward

        mu, sigma = self.encoder_vae(xs, training=True, mode=mode)
        if mode == "TPAGE" or mode == "PPAGE":
            # 表示 训练VAE
            # 这里提醒自己一下 将embeding 全部设为True
            z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        else:
            raise ("许海明在这里提醒你：出现非法mode")



        logits, preds, y, sents2 = self.decoder_vae(ys, z,training=True,mode=mode)

        # train scheme

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["[PAD]"]))  # 0: <pad>
        loss_decoder = tf.reduce_sum(ce * nonpadding) / tf.to_float(get_shape_list(xs[0],expected_rank=2)[0])

        # 这里加上KL loss
        if mode == "TPAGE":
            KL_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1]))
        else:
            KL_loss = 0.0

        loss = loss_decoder + KL_loss


        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(preds)[0] - 1, tf.int32)
        print_demo=(xs[0][n],
                    y[n],
                    preds[n])


        tf.summary.scalar('lr', lr)
        tf.summary.scalar("KL_loss", KL_loss)
        tf.summary.scalar("loss_decoder", loss_decoder)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("global_step", global_step)

        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries, print_demo



    def eval(self, xs,ys,mode):

        mu, sigma = self.encoder_vae(xs, training=False, mode=mode)   #这里主要是为了获取 embeding 其他的都没用

        if mode == "TPAGE" or mode == "PPAGE":
              # 表示 训练VAE
              # 这里提醒自己一下 将embeding 全部设为True
              z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        else:
              raise ("许海明在这里提醒你：出现非法mode")



        # z = tf.random_normal([get_shape_list(xs[0], expected_rank=2)[0], self.hp.z_dim]) #自动生成采样因子

        input_ids_vae_decoder_enc, input_ids_vae_decoder_dec, output_ids_vae_decoder_dec,  sents1, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0])[0], 1), tf.int32) * self.token2idx["<S>"]
        ys = (input_ids_vae_decoder_enc, decoder_inputs, output_ids_vae_decoder_dec,  sents1, sents2)

        logging.info("Inference graph is being built. Please be patient.")
        for _ in tqdm(range(self.hp.maxlen_vae_Decoder_de)):
            logits, y_hat, y, sents2 = self.decoder_vae(ys, z, training=False, mode=mode)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["[PAD]"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (input_ids_vae_decoder_enc, _decoder_inputs, output_ids_vae_decoder_dec, sents1, sents2)

        # monitor a random sample
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0]-1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries