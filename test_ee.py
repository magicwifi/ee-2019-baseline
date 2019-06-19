#! -*- coding: utf-8 -*-


import json
from tqdm import tqdm
import os, re
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
#from keras.models import load_model

if not os.path.exists('../classes.json'):
    id2class = dict(enumerate(D[2].unique()))
    class2id = {j:i for i,j in id2class.items()}
    json.dump([id2class, class2id], open('../classes.json', 'w'))
else:
    id2class, class2id = json.load(open('../classes.json'))



if not os.path.exists('../all_chars_me.json'):
    chars = {}
    for d in tqdm(iter(train_data)):
        for c in d[0]:
            chars[c] = chars.get(c, 0) + 1
    chars = {i:j for i,j in chars.items() if j >= min_count}
    id2char = {i+2:j for i,j in enumerate(chars)} # 0: mask, 1: padding
    char2id = {j:i for i,j in id2char.items()}
    json.dump([id2char, char2id], open('../all_chars_me.json', 'w'))
else:
    id2char, char2id = json.load(open('../all_chars_me.json'))


if not os.path.exists('../random_order_train.json'):
    random_order = list(range(len(train_data)))
    np.random.shuffle(random_order)
    json.dump(
        random_order,
        open('../random_order_train.json', 'w'),
        indent=4
    )
else:
    random_order = json.load(open('../random_order_train.json'))


class Attention(Layer):
    """多头注意力机制
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')
    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10
    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head**0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)




mode = 0
min_count = 2
char_size = 128
maxlen = 256


x_in = Input(shape=(None,)) # 待识别句子输入
c_in = Input(shape=(1,)) # 事件类型
s1_in = Input(shape=(None,)) # 实体左边界（标签）
s2_in = Input(shape=(None,)) # 实体右边界（标签）




x, c, s1, s2 = x_in, c_in, s1_in, s2_in
x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x)

x = Embedding(len(id2char)+2, char_size)(x)
c = Embedding(len(class2id), char_size)(c)
c = Lambda(lambda x: x[0] * 0 + x[1])([x, c])

x = Add()([x, c])
x = Dropout(0.2)(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

x = Bidirectional(LSTM(char_size//2, return_sequences=True))(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])
x = Bidirectional(LSTM(char_size//2, return_sequences=True))(x)
x = Lambda(lambda x: x[0] * x[1])([x, x_mask])

xo = x
x = Attention(8, 16)([x, x, x, x_mask, x_mask])
x = Lambda(lambda x: x[0] + x[1])([xo, x])

x = Concatenate()([x, c])

x1 = Dense(char_size, use_bias=False, activation='tanh')(x)
ps1 = Dense(1, use_bias=False)(x1)
ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])

x2 = Dense(char_size, use_bias=False, activation='tanh')(x)
ps2 = Dense(1, use_bias=False)(x2)
ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])

model = Model([x_in, c_in], [ps1, ps2])
model.load_weights('./best_model.weights')



D = pd.read_csv('../ccks2019_event_entity_extract/event_type_entity_extract_eval.csv', encoding='utf-8', header=None)
test_data = []
for id,t,c in zip(D[0], D[1], D[2]):
    test_data.append((id, t, c))

def extract_entity(text_in, c_in):
    """解码函数，应自行添加更多规则，保证解码出来的是一个公司名
    """
    if c_in not in class2id:
        return 'NaN'
    _x = [char2id.get(c, 1) for c in text_in]
    _x = np.array([_x])
    _c = np.array([[class2id[c_in]]])
    _ps1, _ps2  = model.predict([_x, _c])
    start = _ps1[0].argmax()
    end = _ps2[0][start:].argmax() + start
    return text_in[start: end+1]


def test(test_data):
    """注意官方页面写着是以\t分割，实际上却是以逗号分割
    """
    F = open('result.txt', 'w')
    for d in tqdm(iter(test_data)):
        s = u'"%s","%s"\n' % (d[0], extract_entity(d[1].replace('\t', ''), d[2]))
        s = s.encode('utf-8')
        F.write(s)
    F.close()

test(test_data)

