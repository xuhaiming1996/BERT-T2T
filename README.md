# 介绍
NLP中，对于生成问题如NMT，QA, Paraphrase 任务来说通常会存在生成多样性不足的问题，
通常我们会采用beamSearch来增加多样性。但是beamSeach 生成的句子还是有很大的相似度，无法满足项目落地需求。
我采用了这篇[A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/abs/1709.05074)
的基于CVAE的结构思想构造了一个模型，试图解决生成任务的多样性。


## 模型结构图
提示：请先看这篇论文[A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/abs/1709.05074)
的思想和结构，再看我下面的这个模型结构图

## 文件说明
### /data/PAGE  训练语料
train.txt 格式：id---xhm--src---xhm--tgt

eval.txt 格式：id---xhm--src---xhm--tgt

test.txt 格式：id---xhm--src---xhm--tgt

### results
#### /results/bert
该文件是预训练的好中文bert模型，大家可以去[这里](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)下载，解压后放在这里
#### /results/PAGE
该文件夹是复述模型保存路径



### 运行命令
模型训练使用的是tf.data.* API 从tfrecord文件中构造的迭代器（感慨一下：非常强大的API.建议大家都采用这种方式）

    python train_TPAGE.py   \
         --train=data/PAGE/train.txt \
         --eval=data/PAGE/eval.txt \
         --init_checkpoint_bert=results/bert/bert_model.ckpt \
         --batch_size=32 \
         --eval_batch_size=32 \
         --num_epochs_PAGE=10   \
         --maxlen_vae_Encoder=80 \
         --maxlen_vae_Decoder_en=40\
         --maxlen_vae_Decoder_de=40\
         
####  温馨一刻
大家若对于KL loss的计算公式有疑问，请看这里的[公式推导](https://blog.csdn.net/qq_32806793/article/details/95652645) 你就会明白代码为啥这样写了

    KL_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, [1]))
         