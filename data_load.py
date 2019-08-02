# -*- coding: utf-8 -*-
#/usr/bin/python3

import tensorflow as tf
from utils import calc_num_batches
import copy
import collections
from bert import tokenization



def load_vocab(vocab_fpath):
    '''Loads vocabulary file and returns idx<->token maps
    vocab_fpath: string. vocabulary file path.
    Note that these are reserved
    0: <pad>, 1: <unk>, 2: <s>, 3: </s>
    注意 这里
    0：pad
    [CLS]的位置 我让他学习复述规则
    [UNK]
    [SEP]  两个句子分隔符号 同时也是句子结束符号
    [MASK]
    <S> 一个句子的开始符号
    Returns
    two dictionaries.
    '''
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_fpath, mode="r", encoding="utf-8") as fr:
        for token in fr:
            print(token)
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1

    token2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2token = {idx: token for idx, token in enumerate(vocab)}
    return token2idx, idx2token



def load_data_for_test(fpath):
    '''

    :param fpath:文件的路径
    :param sep  分隔符
    :return:
    '''
    sents1, sents2 = [], []
    with open(fpath, mode='r',encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line is not None and line != "":
                assert len(line.strip())>3
                sents1.append(line.strip())
    sents2=copy.deepcopy(sents1)
    return sents1, sents2




def load_data_for_train_or_eval(fpath, sep="---xhm--"):
    '''

    :param fpath:文件的路径
    :param sep  分隔符
    :return:
    '''
    sents1, sents2 = [], []
    num = 0
    with open(fpath, mode='r',encoding="utf-8") as fr:
        for line in fr:
            num+=1
            if num%1000000==0:
                print("数据正在预处理请你耐心等待。。。。",num)
            line = line.strip()
            if line is not None and line != "":
                sens = line.split(sep)
                if len(sens)!=3:
                    print("语料格式错误的数据：",line)
                    continue
                else:
                    # assert len(sens[0].strip())>=3
                    # assert len(sens[1].strip())>=3
                    #
                    if len(sens[1].strip())<=3 or len(sens[2].strip())<=3:
                        print("数据长度不合格",line)
                        continue
                    sents1.append(sens[1].strip())
                    sents2.append(sens[2].strip())
    return sents1, sents2




def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()



# 许海明
class InputFeatures(object):
  """A single set of features of data."""
  '''
      yield (input_ids_vae_encoder, input_masks_vae_encoder, segment_ids_vae_encoder, sent1, sent2), \
          (input_ids_vae_decoder_enc, input_ids_vae_decoder_dec, output_ids_vae_decoder_dec, sent1, sent2)
  '''


  def __init__(self,
               input_ids_vae_encoder,
               input_masks_vae_encoder,
               segment_ids_vae_encoder,
               input_ids_vae_decoder_enc,
               input_ids_vae_decoder_dec,
               output_ids_vae_decoder_dec,
               sent1,
               sent2,
               is_real_example=True):

            self.input_ids_vae_encoder=input_ids_vae_encoder
            self.input_masks_vae_encoder=input_masks_vae_encoder
            self.segment_ids_vae_encoder=segment_ids_vae_encoder
            self.input_ids_vae_decoder_enc=input_ids_vae_decoder_enc
            self.input_ids_vae_decoder_dec=input_ids_vae_decoder_dec
            self.output_ids_vae_decoder_dec=output_ids_vae_decoder_dec
            self.sent1=sent1
            self.sent2=sent2
            self.is_real_example = is_real_example




# 许海明
def convert_single_example(ex_index,sent1, sent2,
                                  maxlen_vae_Encoder,
                                  maxlen_vae_Decoder_en,
                                  maxlen_vae_Decoder_de, tokenizer):



    sent1_c = copy.deepcopy(sent1)
    sent2_c = copy.deepcopy(sent2)

    '''
          # VAE 的编码器的输入 简单就是一个bert的两句话的合在一起
    '''
    tokens_sent1 = tokenizer.tokenize(sent1)
    tokens_sent2 = tokenizer.tokenize(sent2)
    _truncate_seq_pair(tokens_sent1, tokens_sent2, maxlen_vae_Encoder - 3)
    tokens_vae_encoder = []
    segment_ids_vae_encoder = []

    tokens_vae_encoder.append("[CLS]")
    segment_ids_vae_encoder.append(0)

    for token in tokens_sent1:
      tokens_vae_encoder.append(token)
      segment_ids_vae_encoder.append(0)

    tokens_vae_encoder.append("[SEP]")
    segment_ids_vae_encoder.append(0)

    for token in tokens_sent2:
      tokens_vae_encoder.append(token)
      segment_ids_vae_encoder.append(1)

    tokens_vae_encoder.append("[SEP]")
    segment_ids_vae_encoder.append(1)

    input_ids_vae_encoder = tokenizer.convert_tokens_to_ids(tokens_vae_encoder)
    input_masks_vae_encoder = [1] * len(input_ids_vae_encoder)
    # print("\n\ntokens_vae_encoder",tokens_vae_encoder)
    # print("\n\ninput_ids_vae_encoder",input_ids_vae_encoder)
    while len(input_ids_vae_encoder) < maxlen_vae_Encoder:
        input_ids_vae_encoder.append(0)
        input_masks_vae_encoder.append(0)
        segment_ids_vae_encoder.append(0)

    assert len(input_ids_vae_encoder) == maxlen_vae_Encoder
    assert len(input_masks_vae_encoder) == maxlen_vae_Encoder
    assert len(segment_ids_vae_encoder) == maxlen_vae_Encoder

    # vae的解码器的编码器输入 用符号 3
    tokens_sent3 = tokenizer.tokenize(sent1_c)

    if len(tokens_sent3) > maxlen_vae_Decoder_en - 1:
      tokens_sent3 = tokens_sent3[0:(maxlen_vae_Decoder_en - 1)]

    tokens_vae_decoder_enc = []
    for token in tokens_sent3:
      tokens_vae_decoder_enc.append(token)
      # segment_ids_vae_decoder_enc.append(0)
    tokens_vae_decoder_enc.append("[SEP]")

    # segment_ids_vae_decoder_enc.append(0)
    input_ids_vae_decoder_enc = tokenizer.convert_tokens_to_ids(tokens_vae_decoder_enc)
    # print("\n\ntokens_vae_decoder_enc:",tokens_vae_decoder_enc)
    # print("input_ids_vae_decoder_enc:", input_ids_vae_decoder_enc)

    # input_mask_vae_decoder_enc = [1] * len(input_ids_vae_decoder_enc)
    while len(input_ids_vae_decoder_enc) < maxlen_vae_Decoder_en:
        input_ids_vae_decoder_enc.append(0)
        # input_mask_vae_decoder_enc.append(0)
        # segment_ids_vae_decoder_enc.append(0)

    # vae解码器的解码器的输入和输出 用符号 4
    # 这是训练 PAGE模型的代码
    tokens_sent4 = tokenizer.tokenize(sent2_c)

    if len(tokens_sent4) > maxlen_vae_Decoder_de - 1:
        tokens_sent4 = tokens_sent4[0:(maxlen_vae_Decoder_de - 1)]

    tokens_sent4_input = ["<S>"]
    tokens_sent4_input.extend(copy.copy(tokens_sent4))
    tokens_sent4_output = copy.copy(tokens_sent4)
    tokens_sent4_output.append("[SEP]")

    input_ids_vae_decoder_dec = tokenizer.convert_tokens_to_ids(tokens_sent4_input)
    output_ids_vae_decoder_dec = tokenizer.convert_tokens_to_ids(tokens_sent4_output)


    while len(input_ids_vae_decoder_dec) < maxlen_vae_Decoder_de:
        input_ids_vae_decoder_dec.append(0)
        output_ids_vae_decoder_dec.append(0)
        # input_mask_vae_decoder_dec.append(0)

    assert len(input_ids_vae_decoder_dec) == maxlen_vae_Decoder_de
    assert len(output_ids_vae_decoder_dec) == maxlen_vae_Decoder_de
    # assert len(input_mask_vae_decoder_dec) == maxlen_vae_Decoder_de

    # x = encode(sent1, "x", token2idx)
    # y = encode(sent2, "y", token2idx)
    # decoder_input, y = y[:-1], y[1:]
    #
    # x_seqlen, y_seqlen = len(x), len(y)
    if ex_index<=3:
      print("*** Example ***")
      print("guid: %s" % (ex_index))
      print("\n\ntokens_vae_encoder",tokens_vae_encoder)
      print("input_ids_vae_encoder",input_ids_vae_encoder)


      print("\n\ntokens_vae_decoder_enc:", tokens_vae_decoder_enc)
      print("input_ids_vae_decoder_enc:", input_ids_vae_decoder_enc)



      print("\n\n tokens_input_vae_decoder_dec", tokens_sent4_input)
      print("input_ids_vae_decoder_dec", input_ids_vae_decoder_dec)

      print("\n\n tokens_output_vae_decoder_dec", tokens_sent4_output)
      print("output_ids_vae_decoder_dec", output_ids_vae_decoder_dec)

    feature = InputFeatures( input_ids_vae_encoder=input_ids_vae_encoder,
                        input_masks_vae_encoder=input_masks_vae_encoder,
                        segment_ids_vae_encoder=segment_ids_vae_encoder,
                        input_ids_vae_decoder_enc=input_ids_vae_decoder_enc,
                        input_ids_vae_decoder_dec=input_ids_vae_decoder_dec,
                        output_ids_vae_decoder_dec=output_ids_vae_decoder_dec,
                        sent1=sent1,
                        sent2=sent2,
                         is_real_example=True)
    return feature




# 许海明
def file_based_convert_examples_to_features(sentS1,
                                            sentS2,
                                            maxlen_vae_Encoder,
                                            maxlen_vae_Decoder_en,
                                            maxlen_vae_Decoder_de,
                                            tokenizer,
                                            output_file):

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, (sent1,sent2)) in enumerate(zip(sentS1,sentS2)):
    if ex_index % 500000 == 0:
      print("Writing example %d of %d" % (ex_index, len(sentS1)))

    feature = convert_single_example(ex_index,sent1, sent2,
                                      maxlen_vae_Encoder,
                                      maxlen_vae_Decoder_en,
                                      maxlen_vae_Decoder_de, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    features = collections.OrderedDict()

    features["input_ids_vae_encoder"] = create_int_feature(feature.input_ids_vae_encoder)
    features["input_masks_vae_encoder"] = create_int_feature(feature.input_masks_vae_encoder)
    features["segment_ids_vae_encoder"] = create_int_feature(feature.segment_ids_vae_encoder)
    features["input_ids_vae_decoder_enc"] = create_int_feature(feature.input_ids_vae_decoder_enc)
    features["input_ids_vae_decoder_dec"] = create_int_feature(feature.input_ids_vae_decoder_dec)
    features["output_ids_vae_decoder_dec"] = create_int_feature(feature.output_ids_vae_decoder_dec)

    features["sent1"] = create_bytes_feature(feature.sent1)
    features["sent2"] = create_bytes_feature(feature.sent2)


    features["is_real_example"] = create_int_feature( [int(feature.is_real_example)] )


    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()









def file_based_input_fn_builder(input_file,
                                maxlen_vae_Encoder,
                                maxlen_vae_Decoder_en,
                                maxlen_vae_Decoder_de,
                                is_training):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids_vae_encoder": tf.FixedLenFeature([maxlen_vae_Encoder], tf.int64),
      "input_masks_vae_encoder": tf.FixedLenFeature([maxlen_vae_Encoder], tf.int64),
      "segment_ids_vae_encoder": tf.FixedLenFeature([maxlen_vae_Encoder], tf.int64),
      "input_ids_vae_decoder_enc": tf.FixedLenFeature([maxlen_vae_Decoder_en], tf.int64),
      "input_ids_vae_decoder_dec": tf.FixedLenFeature([maxlen_vae_Decoder_de], tf.int64),
      "output_ids_vae_decoder_dec": tf.FixedLenFeature([maxlen_vae_Decoder_de], tf.int64),
      "sent1": tf.FixedLenFeature([], tf.string),
      "sent2": tf.FixedLenFeature([], tf.string),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(batch_size):
    """The actual input function."""
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size))

    return d

  return input_fn










def saveForTfRecord(fpath,
                     maxlen_vae_Encoder,
                     maxlen_vae_Decoder_en,
                     maxlen_vae_Decoder_de,
                     vocab_fpath,
                     output_file,
                     is_test=False):
    '''

    :param fpath:
    :param maxlen_vae_Encoder:
    :param maxlen_vae_Decoder_en:
    :param maxlen_vae_Decoder_de:
    :param vocab_fpath:
    :param output_dir:   这个是输出保存路径
    :param is_test:      是否是测试数据
    :return:
    '''
    if is_test:
        # 是测试流程
        sents1, sents2 = load_data_for_test(fpath)
    else:
        sents1, sents2 = load_data_for_train_or_eval(fpath)
        # sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)

    print("读取完成")

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fpath, do_lower_case=True)
    # output_file = os.path.join(output_dir, "train.tf_record")
    file_based_convert_examples_to_features(sents1,
                                            sents2,
                                            maxlen_vae_Encoder,
                                            maxlen_vae_Decoder_en,
                                            maxlen_vae_Decoder_de,
                                            tokenizer,
                                            output_file)






def get_batch_for_train_or_dev_or_test(  fpath,
                                         maxlen_vae_Encoder,
                                         maxlen_vae_Decoder_en,
                                         maxlen_vae_Decoder_de,
                                         batch_size,
                                         input_file,
                                         is_training,
                                         is_test):
    '''

    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''

    if is_test:
        # 是测试流程
        sents1, sents2 = load_data_for_test(fpath)
    else:
        sents1, sents2 = load_data_for_train_or_eval(fpath)
        # sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)

    input_fn=file_based_input_fn_builder(input_file,
                                            maxlen_vae_Encoder,
                                            maxlen_vae_Decoder_en,
                                            maxlen_vae_Decoder_de,
                                            is_training)

    batches=input_fn(batch_size)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)






