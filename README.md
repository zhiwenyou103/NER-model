# NER-model
Using IDCNN and CRF to create a sequence labeling model. 

> 本项目使用IDCNN和CRF，结合Word2Vec，实现中英文数据的IOBES命名实体标注

[主要参考的模型](https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF)
在本项目的基础上做了英文版的优化工作，并逐行做了comments加深模型理解，十分感谢作者的开源代码，受益匪浅！

---

#### 项目的组成
- main程序——训练和测试模型，调用其他程序中的方法。
- model程序——设置模型参数，建立模型，包括CNN、CRF以及词向量创建。
- 其他程序——数据预处理函数、格式转换函数、数据加载函数等。
- 数据集——包括中文和英文数据集

Enjoy！
