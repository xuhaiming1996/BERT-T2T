import argparse

class Hparams:
    parser = argparse.ArgumentParser()




    ## files
    parser.add_argument('--train',  help="训练数据")
    parser.add_argument('--eval',   help="验证数据")
    parser.add_argument('--test',   help="测试数据")


    ## vocabulary
    parser.add_argument('--vocab', default='results/bert/vocab.txt',
                        help="vocabulary file path")

    parser.add_argument('--init_checkpoint_LM', help="语言模型的初始路径-fine_tune")
    parser.add_argument('--init_checkpoint_bert', help="bert模型的初始路径-fine_tune")
    parser.add_argument('--init_checkpoint_PAGE', help="复述模型的初始路径-fine_tune")


    # training scheme
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=32, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)

    parser.add_argument('--lr', default=5e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=100, type=int)

    parser.add_argument('--num_epochs_LM', default=20, type=int, help="语言模型训练的epoch")
    parser.add_argument('--num_epochs_PAGE', default=20, type=int, help="复述模型训练的epoch")

    parser.add_argument('--LMdir', default="results/LM",     help="这是语言模型的的路径")
    parser.add_argument('--PAGEdir', default="results/PAGE", help="这是复述模型的的路径")

    # model
    parser.add_argument('--d_model', default=768, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=4, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")

    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")




    parser.add_argument('--maxlen_vae_Encoder', default=80, type=int,
                        help="VAE编码器的最大的长度 len(sen1)+len(sen2)")
    parser.add_argument('--maxlen_vae_Decoder_en', default=40, type=int,
                        help="源句子的最大长度")
    parser.add_argument('--maxlen_vae_Decoder_de', default=40, type=int,
                        help="复述句的最大长度")



    # vae Z
    parser.add_argument('--z_dim', default=768, type=int,
                        help="VAE中那个Z的维度")



    # bert的参数
    parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float)
    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument('--initializer_range', default=0.02, type=float)


    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--intermediate_size', default=3072, type=int)
    parser.add_argument('--max_position_embeddings', default=512, type=int)
    parser.add_argument('--num_attention_heads', default=12, type=int)
    parser.add_argument('--num_hidden_layers', default=12, type=int)
    parser.add_argument('--pooler_fc_size', default=768, type=int)
    parser.add_argument('--pooler_num_attention_heads', default=12, type=int)


    parser.add_argument('--pooler_num_fc_layers', default=3, type=int)
    parser.add_argument('--pooler_size_per_head', default=128, type=int)
    parser.add_argument('--type_vocab_size', default=2, type=int)
    parser.add_argument('--vocab_size', default=21128, type=int)

    parser.add_argument('--directionality', default="bidi")
    parser.add_argument('--hidden_act', default="gelu")
    parser.add_argument('--pooler_type', default="first_token_transform")

