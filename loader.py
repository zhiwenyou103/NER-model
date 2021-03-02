# encoding=utf8
import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):  # path是待传入data，包括train、dev、test
        num+=1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()  # eg：['word', 'tag']
                # word[0] = " "
            else:
                word = line.split()
            assert len(word) >= 2, print([word[0]])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):  # s是sentence，[['Russian', 'B-MISC'], ['military', 'O']
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):  # zip将s和new tags两个list合并为一个元组的形式
                #  word: ['BRUSSELS', 'B-LOC']
                #  new_tag是转换为iobes后的纯标签
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]  # 首先将每个字符转换为小写
    dico = create_dico(chars)  # 调用data_utils.py中的create_dico函数创建key-value的映射，生成含有每个字出现的次数的字典
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)  # 生成key-value和value-key两种字典映射形式，按value值降序排列
    print("Found %i unique words (%i in total)" % (  # print字典中共有多少不同字符和一共有多少字符
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes  每个单词在大字典中出现时的对应id
        - word char indexes
        - tag indexes
    """
    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:  # 对于训练集list中的每个已划分好的sentence句子
        string = [w[0] for w in s]  # string是sentence中除去tag的纯word，word用逗号分隔
        # ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        # segs = get_seg_features("".join(string))  # 调用data_utils.py中的get_seg_features函数，利用jieba划分词，生成代表分词后每个词的长度特征
        segs = get_seg_features(" ".join(string))  # 英文不能直接去除空格合并成一句话，还要保留空格以便之后取词
        # segs:[1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 0, 1, 3, 1, 3, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 2, 3, 0, 0, 1, 3, 1, 3, 0, 0, 1, 2, 2, 3, 0]
        if train:
            tags = [tag_to_id[w[-1]] for w in s]  # tags是训练集标签出现的frequency
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])  # output：训练集word字符——word的frequency——每句话分词后的word特征——标签的frequency
    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:  # pretrain集任意字符在test中出现过并且通过train生成的字典中未出现的预训练字符放入字典，作为新训练字典
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word  # dictionary未排序，其他两个是排好序的字典


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

