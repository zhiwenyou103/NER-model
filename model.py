# encoding=utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers

import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config, is_train=True):

        self.config = config
        self.is_train = is_train
        
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]
        self.lstm_dim = config["lstm_dim"]
        self.seg_dim = config["seg_dim"]

        self.num_tags = config["num_tags"]
        self.num_chars = config["num_chars"]
        '''
        Q：num_segs代表什么？？
        '''
        self.num_segs = 4

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()  # 用于初始化权重的初始化程序,用来保持每一层的梯度大小都差不多相同
        
        

        # add placeholders for the model
        # char_inputs和seg_inputs是在create_feed_dict中传入数据的
        # placeholder 用于传递进来的训练样本，类似一个占位符，不用指定初始值，可在运行时，通过Session.run()的feed_dict参数指定
        # 因为没有传入数据，所以print出都是"?"
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                         shape=[None, None],
                                         name="SegInputs")

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        # dropout keep prob
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")

        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        # batch_size: Tensor("strided_slice:0", shape=(), dtype=int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        # num_steps: Tensor("strided_slice_1:0", shape=(), dtype=int32)
        # num_steps是每个char（20*x）中的x，即该batch下最长的句子长度
        self.num_steps = tf.shape(self.char_inputs)[-1]

        
        
        #Add model type by crownpku， bilstm or idcnn
        self.model_type = config['model_type']
        #parameters for idcnn
        # 膨胀卷积 在filter中加入空行和列来扩大卷积范围 =1和普通卷积一样；=2时膨胀数为1，每个原始卷积矩阵的每个矩阵元素被一行和一列分隔开
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]
        # 卷积filter的长度=3，高度都为1
        self.filter_width = 3
        # idcnn模型中每个卷积层output时的通道数 120->100，4个卷积层就是400维
        self.num_filter = self.lstm_dim
        # 词向量维度=神经网络通道数=120 ，相当于一个单词的向量数据由120层的网络层叠加
        self.embedding_dim = self.char_dim + self.seg_dim
        self.repeat_times = 4
        self.cnn_output_width = 0
        
        

        # embeddings for chinese character and segmentation representation
        # embedding: [batch_size, num_steps, emb_size]
        embedding = self.embedding_layer(self.char_inputs, self.seg_inputs, config)

        if self.model_type == 'bilstm':
            # apply dropout before feed to lstm layer
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # bi-directional lstm layer
            model_outputs = self.biLSTM_layer(model_inputs, self.lstm_dim, self.lengths)

            # logits for tags
            self.logits = self.project_layer_bilstm(model_outputs)
        
        elif self.model_type == 'idcnn':
            '''
            # apply dropout before feed to idcnn layer
            # tf.nn.dropout: 防止或减轻过拟合.丢掉一部分语义信息，将其向量置为0，main中设置drop值为0.5
            # tf.nn.dropout（）中的参数 x：指输入，输入tensor；
            # keep_prob: float类型，每个元素被保留下来的概率，设置数据被选中的概率,在初始化时keep_prob是一个占位符, 
            # keep_prob = tf.placeholder(tf.float32) 。tensorflow在run时设置keep_prob具体的值，例如keep_prob: 0.5
            # noise_shape  : 一个1维的int32张量，代表了随机产生“保留/丢弃”标志的shape。seed : 整形变量，随机数种子。
            # name：指定该操作的名字
            '''
            # model_inputs = [batch_size, num_steps, emb_size]
            model_inputs = tf.nn.dropout(embedding, self.dropout)

            # ldcnn layer
            # model_outputs: Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)
            # 输出卷积后提取出的word特征
            model_outputs = self.IDCNN_layer(model_inputs)

            # logits for tags
            # 对输入句子的每一个字生成一个logits，是每个词分类的概率
            # 中文logits output: Tensor("project/Reshape:0", shape=(?, ?, 13), dtype=float32)
            # logits中的13表示13个tags
            # 英文版：Tensor("project/Reshape:0", shape=(?, ?, 17), dtype=float32)
            self.logits = self.project_layer_idcnn(model_outputs)
        else:
            raise KeyError

        # loss of the model
        # 将idcnn输出的logits放入CRF层用Viterbi算法解码出标注结果
        # self.loss: Tensor("crf_loss/Mean:0", shape=(), dtype=float32)
        self.loss = self.loss_layer(self.logits, self.lengths)  # 从cnn模型输出的logits作为loss layer的输入

        with tf.variable_scope("optimizer"):
            # 本模型选择的是adam
            #
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                # raise KeyError
                raise Exception("优化器错误")
            # apply grad clip to avoid gradient explosion
            # self.opt设置防止梯度爆炸的优化器
            # compute_gradients内的参数是需要minimize的value，必须是Tensor格式
            # grads_vars返回的是（梯度，变量）的list，变量是可以minimize loss值得一组变量
            grads_vars = self.opt.compute_gradients(self.loss)
            # clip_by_value：输入张量g，设置min和max，让g中每一个元素值都压缩在min-max之间，小于min的=min，大于max的=max
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            # apply_gradients：应用执行完梯度更新capped_grads_vars的梯度平滑规则
            # global step：设置梯度增量，这里为False
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def embedding_layer(self, char_inputs, seg_inputs, config, name=None):
        """
        embedding第一步是通过索引对该句子进行编码，索引就是每个word对应其在字典中的value
        :param char_inputs: one-hot encoding of sentence
        :param seg_inputs: segmentation feature
        :param config: wither use segmentation feature
        :return: [1, num_steps, embedding size],
        """
        # char_inputs：Tensor("ChatInputs:0", shape=(?, ?), dtype=int32)
        embedding = []
        with tf.compat.v1.variable_scope("char_embedding" if not name else name), tf.device('/cpu:0'):
            self.char_lookup = tf.compat.v1.get_variable(  # get_variable作用：创建新的tensorflow变量,这里创建随机矩阵21011*100
                    name="char_embedding",
                    shape=[self.num_chars, self.char_dim],
                    initializer=self.initializer)
            # char lookup 行是字典中词个数，列是100维矩阵，将char input映射到这个100维矩阵
            # 英文char_lookup： <tf.Variable 'char_embedding/char_embedding:0' shape=(21011, 100) dtype=float32_ref>
            # self.num_chars = 21011 英文
            # self.char_dim = 100 英文
            # 创建初始化embedding矩阵
            # 查找char_inputs在char_lookup中对应的随机生成的权重
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, char_inputs))  # 将输入序列转换成WordEmbedding表示作为隐藏层的输入
            if config["seg_dim"]:
                with tf.variable_scope("seg_embedding"), tf.device('/cpu:0'):
                    self.seg_lookup = tf.get_variable(
                        name="seg_embedding",
                        shape=[self.num_segs, self.seg_dim],
                        initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, seg_inputs))
            embed = tf.concat(embedding, axis=-1)
        # embed: Tensor("char_embedding/concat:0", shape=(?, ?, 120), dtype=float32, device=/device:CPU:0)
        # embed: [batch_size, num_steps, emb_size] 三维浮点数张量形式
        return embed

    def biLSTM_layer(self, model_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],
                lstm_cell["backward"],
                model_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)
    
    #IDCNN layer 
    def IDCNN_layer(self, model_inputs, 
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size]——[20, 该batch句子最大长度，120]
        :return: [batch_size, num_steps, cnn_output_width]
        作用：对输入句子的每一个字生成一个logits，提取原始输入数据的全部特征信息
        主要包含卷积层和和激活函数层
        """
        # tf.expand_dims：给定一个input，在axis轴处给input增加一个为1的维度 eg：=1时，从2*3*5变为2*1*3*5
        model_inputs = tf.expand_dims(model_inputs, 1)
        # model_inputs:Tensor("ExpandDims:0", shape=(?, 1, ?, 120), dtype=float32)
        # 问号形式的数字即不可人工调参，其余可显示部分可手动调参
        reuse = False
        if not self.is_train:
            reuse = True
        # filter_width = 3， embedding_dim = char_dim + seg_dim = 120,  num_filter = 100
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim,
                       self.num_filter]
            # 英文shape:[1, 3, 120, 100]
            print(shape)
            # 初始化过滤器的权重矩阵
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializer)
            # 英文filter_weights.shape = (1, 3, 120, 100)——1*3的卷积核100个，作用于120个通道

            """
            shape of input = model_inputs = [batch, in_height, in_width, in_channels] = [?, 1, ?, 120]
            in_height: 文本高度。in_width：句子长度。in_channels：文本通道数
            shape of filter = filter_weights = [filter_height, filter_width, in_channels, out_channels] = [1, 3, 120, 100]
            filter_height：卷积核高度。filter_width：卷积核宽度。out_channels：卷积核个数
            """
            # tf.nn.conv2d: 给定input和4D filters张量计算2D卷积。具体参数如下：
            # input必须是4维tensor，要求类型为float32和float64其中之一。此模型是1*n，120个通道的神经元
            # filter是神经网络滤波器，维度=input，输入为(1, 3, 120, 100)——1*3的卷积核100个，作用于120个通道,最终卷积后通道数变为100
            # strides：控制卷积核的移动步数，是一维向量，同input维度一样，第一个和最后一个1固定值，可变中间两参数，即在x和y轴移动的步长，这里是1
            # padding：只有两个取值，'SAME'和'VALID'，第一个是填充边界，第二个是当不足以移动时直接舍弃
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            # layerInput：得到一个1 *（n-2）的，通道数为100的feature map，n是该batch的句子长度
            # layerInput.shape = (20, 1, n-2, 100)
            # layerInput是先进行普通非膨胀卷积计算，只到最后一层有膨胀系数=2才进行膨胀计算
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                # repeat_times = 4 4个大的相同结构的Dilated CNN block，上一个block的输出作为下一个的输入
                for i in range(len(self.layers)):
                    # dilation=2，filter还=1*3，输入进来的卷积后的数据被输入进更大的receptive field提取特征，从而快速覆盖所有数据
                    dilation = self.layers[i]['dilation']
                    # 进行卷积操作
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.compat.v1.AUTO_REUSE):
                        # w创建随机矩阵 shape = (1, 3, 100, 100)
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        # b构建一维向量，shape = (1 * 100)
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        # conv进行膨胀卷积，输入的参数如下：
                        # layerInput：输入的卷积后数据，[batch, height, width, channels] = (20, 1, n-2, 100)
                        # filters：卷积核，[filter_height, filter_width, channels, out_channels]，通常NLP相关height设为1。
                        # rate：正常的卷积通常会有stride，即卷积核滑动的步长，而膨胀卷积通过定义卷积核当中穿插的rate个0的个数，实现对原始数据采样间隔变大。
                        # padding：”SAME”：补零；”VALID”：丢弃多余的
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        # conv.shape = (?, 1, ?, 100) ,分别为batch，句子高度，卷积后句子长度，100维）
                        # 将b这个一维向量加到conv矩阵上，向量与conv矩阵每一行相加
                        conv = tf.nn.bias_add(conv, b)
                        # 计算激活函数，将卷积后矩阵中大于0的保持不变，小于0的数置为0
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            # 因为这里要重复循环4次，共4个卷积block，所以totalWidthForLastDim = 400
                            totalWidthForLastDim += self.num_filter
                        # 上一层的卷积output=下一层的卷积input
                        layerInput = conv
            # tf.concat在第四维度拼接，即100这个维度，
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            # reuse = False，模型训练时置为0.5，防止过拟合设置的参数，去除卷积后数据的50%
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)
            # tf.squeeze将原始finalOut中所有维度为1的那些维都删掉的结果，即删掉height
            # 删除后finalOut = （20，m，100），m是卷积后句子的长度
            finalOut = tf.squeeze(finalOut, [1])
            # tf.reshape把finalOut转化为20 * m行 100列的矩阵，-1代表不管有多少行数据（其实是20行），totalWidthForLastDim表示要转换为的列数
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            # finalOut：Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)
            # self.cnn_output_width = 400 即通道数为400
            return finalOut

    def project_layer_bilstm(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project" if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])
    
    # Project layer for idcnn by crownpku
    # Delete the hidden layer, and change bias initializer
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param idcnn_outputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, num_tags]
        """
        # idcnn_outputs 中英文: Tensor("idcnn/Reshape:0", shape=(?, 400), dtype=float32)
        with tf.variable_scope("project" if not name else name):
            # project to score of tags
            with tf.variable_scope("logits"):

                # cnn_output_width = 400，英文num_tags = 17
                # W: <tf.Variable 'project/logits/W:0' shape=(400, 17) dtype=float32_ref>
                W = tf.compat.v1.get_variable("W", shape=[self.cnn_output_width, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                # b: <tf.Variable 'project/logits/b:0' shape=(17,) dtype=float32_ref>
                # tf.constant创建常量，此处b为一维矩阵，共17行值，每行值为0.001的
                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_tags]))

                # xw_plus_b主要计算X*W+b的值，W是权重，b是biases
                # idcnn_outputs：输入的矩阵，是网络结构上一层的输出，维度为[batch_size, in_units]表示输入的样本数*每个样本用多少个单元表示
                # W维度为[in_units, out_units]。第一个维度和idcnn_outputs的最后一个维度一致，因为需要和idcnn_outputs进行矩阵相乘的计算
                # b：偏置。维度为一维，[out_units],和W的最后一维一致，因为最终要加在XW的结果矩阵上。而X*W矩阵的维度为[batch_size, out_units]
                # pred = Tensor("project/logits/xw_plus_b:0", shape=(?, 17), dtype=float32)
                # pred: 对输入句子的每一个字生成一个logits，是一个向量，logits值没有标准化，每一行的和不等于1，使用crf分类函数对其进行归一化处理
                pred = tf.compat.v1.nn.xw_plus_b(idcnn_outputs, W, b)
            # output: [batch_size, num_steps, num_tags] 转换为tag数量维度的形式，即句子中的每个字被分类为标签集中每个tag的分类概率值
            # 每个word都会生成tag数量的概率值，只不过概率值相加不等于1
            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss标量输出
        传入参数：project_logits：project_layer_idcnn函数输出的logits，即每个word映射到所有tag的非归一化概率；lengths：根据输入训练数据的sentence长度得到的lengths
        利用CRF层来学习一个最优路径，为每一个原始数据的正确tag定义一个概率值，训练cnn传入的logits数据，最大化似然其概率值即可
        CRF可以学习tag的上下文，而不是特征的上下文
        通过crf层，可以学习出train数据的最优状态序列，即训练出转移矩阵，以便后续使用维特比算法解码并预测给定句子的标注tag。
        """
        # 中文：project_logits：Tensor("project/Reshape:0", shape=(?, ?, 13), dtype=float32)
        # project_logits: Tensor("project/Reshape:0", shape=(?, ?, 17), dtype=float32)
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0

            # pad logits for crf loss
            # start_logits = Tensor("crf_loss/concat:0", shape=(?, 1, 18), dtype=float32)
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)

            # tf.cast：执行 tensorflow 中张量数据类型转换，这里转化为浮点型
            # pad_logits = Tensor("crf_loss/mul_1:0", shape=(?, ?, 1), dtype=float32)
            # pad_logits.shape = (20, n, 1) n代表句子长度
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)

            # tf.concat：拼接张量,axis=-1表示最后一个维度，沿着tag的维度拼接成18层的三维矩阵
            # logits.shape: Tensor("crf_loss/concat_1:0", shape=(?, ?, 18), dtype=float32)
            # 使每个字对应一个tag，而不是每个字对应所有tag
            logits = tf.concat([project_logits, pad_logits], axis=-1)

            # logits.shape:: Tensor("crf_loss/concat_2:0", shape=(?, ?, 18), dtype=float32)
            # 让每句话的句子长度+1
            logits = tf.concat([start_logits, logits], axis=1)

            # self.targets是训练数据中每句话每个词对应的tag，即真实句子标签，print后是" ？"
            # target.shape : （batch，tags)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            # self.trans是18*18的转移矩阵，这个转移矩阵的维度是固定的，m*m，m代表tag个数
            # 转移矩阵可以随机初始化，random score会在下面计算log过程中自动更新
            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)

            # crf_log_likelihood计算标签序列的对数似然值，参数如下
            # inputs: 预测出的tag，一个形状为[batch_size, max_seq_len, num_tags] 的tensor,project layer格式转换后的输出作为CRF层的输入.
            # tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,就是真实句子标签.
            # sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度.
            # transition_params: 形状为[num_tags, num_tags] 的转移矩阵
            # 此函数目的是使输入的logits尽可能的靠近targets，不断收敛使得loss值最小
            # 通过结合emission scores和transition scores可以得到每个句子的total path scores，
            # loss value = true path score / total path score
            # 真实路径score = 初始随机生成的emission score转换为true tag的转换概率值加和，这个真实路径score会随着迭代逐渐增大
            log_likelihood, self.trans = crf_log_likelihood(  # 对数可以把乘法运算转换为加法，除法转换为减法，求导数时就可以分别求导，简化运算
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, chars, segs, tags = batch  # 英文的seg是list类型
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),  # 生成每句话中单词的feature map，即data_utils中的get_seg_features生成的
            self.dropout: 1.0,
        }
        # print(np.asarray(chars).shape)
        # issue: 每次英文数据print chars的shape都不一样
        # 中文np.asarray(segs).shape：(20, 55)、(20, 45)、(20, 68)、(20, 35)
        # 英文np.asarray(chars).shape：（20， 9)/(20, 22)
        # 中文np.asarray(chars).shape： (20, 70)、(20, 56)、(20, 41)、(20, 31)、(20, 59)、(20, 45)、(20, 35)、(20, 50)、(20, 44)

        # print(segs.shape)
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)  # feed_dict是字典
        # feed_dict有三个array，char_inputs是word对应的value，seg_inputs是每句话特征0/1/2/3表示，targets是tag对应的value
        # train_op是梯度优化器
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = iobes_iob([id_to_tag[int(x)] for x in tags[i][:lengths[i]]])
                pred = iobes_iob([id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval()
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(inputs[0][0], tags)
