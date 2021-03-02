# encoding=utf8
import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import load_word2vec, create_input, input_from_line, BatchManager

flags = tf.app.flags  # 规定命令行传入参数格式，例如train=True，clean=True，model_type=idcnn
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", True, "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")
#  iobes：包含了全部的5种标签，文本块由单个字符组成的时候，使用S标签来表示，由一个以上的字符组成时，首字符总是使用B标签，尾字符总是使用E标签，中间的字符使用I标签。
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
# flags.DEFINE_boolean("pre_emb", False, "Wither use pre-trained embedding")
flags.DEFINE_boolean("pre_emb", False, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", False, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", True, "Wither lower case")

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")  # 保存训练好的模型的路径
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", os.path.join("data", "vec.txt"), "Path for pre_trained embedding")  # 已预训练好的词向量
# flags.DEFINE_string("train_file", os.path.join("data", "example.train"), "Path for train data")
# flags.DEFINE_string("dev_file", os.path.join("data", "example.dev"), "Path for dev data")
# flags.DEFINE_string("test_file", os.path.join("data", "example.test"), "Path for test data")
flags.DEFINE_string("train_file", os.path.join("data", "train.txt"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join("data", "valid.txt"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "test.txt"), "Path for test data")

flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
# flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

# tf.app.flags.FLAGS 用来命令行运行代码时传递参数，传入参数格式line19-50
# FLAGS.XXX中的XXX表示命令行输入的字符
FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    """
    计算dev和test集的f1值
    """
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def train():
    """
    train函数：传入数据、处理数据、模型训练、输出测试集f1值
    :return:
    """
    # load data sets传入数据集，做基本处理包括转小写、换0、去除空格提取word等，将训练集word和tag放在list中。 .dev_file用作cross validation
    train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)  # FLAGS.zeros = False
    # train_sentences格式 ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC']
    dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)

    # Use selected tagging scheme (IOB / IOBES) 将IOB格式标签转换文IOBES。I：中间，O：其他，B：开始 | E：结束，S：单个
    # 调用loder.py中的update_tag_scheme函数进行tag转换，在此函数内又调用data_utils.py中的iob_iobes函数转换tag
    update_tag_scheme(train_sentences, FLAGS.tag_schema)
    update_tag_scheme(test_sentences, FLAGS.tag_schema)
    update_tag_scheme(dev_sentences, FLAGS.tag_schema)

    # create maps if not exist 创建词映射字典
    if not os.path.isfile(FLAGS.map_file):
        # create dictionary for word
        if FLAGS.pre_emb:  # 数据增强 添加预训练词向量到训练字典中
            dico_chars_train = char_mapping(train_sentences, FLAGS.lower)[0]  # 调用loader.py中的char_mapping函数，只输出一个被转换为小写的数据集字典，frequency降序排列
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(  # 调用loader.py中的augment_with_pretrained函数
                # 添加原字典中没有的pretrain字符到原字典中，pretrain必须在test集中有出现过
                dico_chars_train.copy(),
                FLAGS.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])  # 使用test集作为预训练词向量的基准
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS.lower)
            # _c是无序字典，即列出了每个key出现的次数。char_to_id是有序字典，但是value不是frequency，是序号，但key排列顺序是按frequence降序

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)  # 调用loader.py中的tag_mapping函数创建tag字典，_t是不重复tag的字典
        # tag_to_id: {'O': 0, 'S-MISC': 1, 'B-ORG': 2, 'B-PER': 3, 'E-ORG': 4, 'E-PER': 5, 'S-LOC': 6, 'S-ORG': 7, 'I-PER': 8, 'S-PER': 9}
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)  # 将上述字典保存到map file中
    else:
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(  # 调用loader.py中的prepare_dataset函数生成 训练集word字符——word的frequency——分词后的word特征——标签的frequency
        train_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    # 生成bach_size大小 可以调用batch_data和len_data两个内置变量
    # BatchManager用来统一输入的训练数据的array的长度
    train_manager = BatchManager(train_data, FLAGS.batch_size)  # data_utils.py传入BatchManager类
    dev_manager = BatchManager(dev_data, 100)
    test_manager = BatchManager(test_data, 100)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)  # output配置文件config_file
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)  # 打印生成的log并储存在文件夹内

    # 迭代原理 训练 loss值如何产生
    # limit GPU memory
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # 英文：steps_per_epoch = 703 即一共需要处理的训练数据批次量, steps_per_epoch * 20 = 总共的句子个数
    # 中文：steps_per_epoch = 1044
    steps_per_epoch = train_manager.len_data
    # 开始训练模型
    with tf.compat.v1.Session(config=tf_config) as sess:  # 使用tf.Session激活配置参数，使用utils.py中create_model函数下session.run执行
        # 创建模型框架，包括init函数中定义的模型各个层参数和相应函数调用，先生成num_chars * 100的word embedding权重矩阵
        model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger)
        # 调用utils.py中的Model类创建模型，传入训练字典。调用data_utils中的load_word2vec函数
        logger.info("start training")
        loss = []
        # 这层循环的意义是共训练模型100次，不断传入train和验证集来调整模型的参数，得到最优F1值。括号内的range（100）可调参
        for i in range(100):
            # 开始训练模型 传入数据集
            # 先在模型中根据batch创建输入字典feed_dict，每20个一组，包括每句话的word id，每句话的word feature，每句话tag id
            # 依次执行模型每一层，从embedding layer开始
            # 生成词向量，按批次传入参数包括每句话的char id；每句话feature和是否存在句子维度的预先定义值，生成120维包含所有训练数据的词向量
            # 用dropout随机去除部分词向量防止过拟合，将词向量喂给CNN模型进行卷积训练。
            for batch in train_manager.iter_batch(shuffle=True):  # iter_batch：data_utils.py中的iter_batch函数
                # batch是产生随机顺序的句子，输出上述array
                # batch组成：4个大list，每个list包含：
                # 1. 随机输出的所有句子，['Fairview', ',', 'Texas', ',', '$', '1.82', 'million', 'deal', 'Baa1', '-'],
                # 2. word出现在字典中的位置。
                # 3. 每句话对应的表征word长度特征的list。
                # 4. 每句话对应的tag在tag字典中出现的位置
                step, batch_loss = model.run_step(sess, True, batch)
                # loss：60.648315 76.53908 54.006336 108.96472
                # step从1开始增加，每100次输出一次当前loss值
                loss.append(batch_loss)
                # 5个batch输出一次loss值，step=100，总batch
                if step % FLAGS.steps_check == 0:  # 每迭代100次输出一次loss，
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)


# def evaluate_line():
#     config = load_config(FLAGS.config_file)
#     logger = get_logger(FLAGS.log_file)
#     # limit GPU memory
#     tf_config = tf.ConfigProto()
#     tf_config.gpu_options.allow_growth = True
#     with open(FLAGS.map_file, "rb") as f:
#         char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
#     with tf.Session(config=tf_config) as sess:
#         model = create_model(sess, Model, FLAGS.ckpt_path, load_word2vec, config, id_to_char, logger, False)
#         while True:
#             # try:
#             #     line = input("请输入测试句子:")
#             #     result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
#             #     print(result)
#             # except Exception as e:
#             #     logger.info(e)
#
#             line = input("请输入测试句子:")
#             result = model.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)
#             print(result)


def main(_):
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)  # 调用utils.py程序中的clean函数，清空folder内之前训练过的log和model
        train()
    else:
        evaluate_line()  # 训练完模型后执行，进行单句测试，调用line197 evaluate_line函数测试


if __name__ == "__main__":
    tf.app.run(main)
