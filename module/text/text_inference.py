import re
from keras.models import load_model
import os
from collections import OrderedDict
import numpy as np
from konlpy.tag import Okt  
from gensim.models import Word2Vec
from keras.layers import Layer
from keras import backend as K
from keras.models import Model
import sys


char_maxlen = 48
word_maxlen = 27

text = "나는 너무 행복합니다"

tw = Okt()

module_path = os.path.dirname(os.path.realpath(__file__))

model_name = module_path + "/model/5Y_text_emotion_200728.h5"
word2vec_file = module_path + "/model/5Y_w2v.model"
w2v_model = Word2Vec.load(word2vec_file)
index2word_set = set(w2v_model.wv.index2word)

# =============================================================================
# attention_layer
# =============================================================================
class AttentionLayer(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], self.attention_dim),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(self.attention_dim, ),
                                 initializer='random_normal',
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(self.attention_dim, 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
        u_it = K.tanh(K.dot(x, self.W) + self.b)
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)
        return a_it

model = load_model(model_name, custom_objects={'AttentionLayer':AttentionLayer})
print("model load compelete!")
feature_model = Model(model.input, model.get_layer('concatenate_1').output)
feature_model._make_predict_function()

def convert_to_word(data):
    feature_vector = np.zeros((word_maxlen, 200))
    sentence = [xx for xx in tw.morphs(data, norm = True, stem = True) if xx in index2word_set]
    if len(sentence) > word_maxlen:
        sentence = sentence[:word_maxlen]
    for j, word in enumerate(sentence):
        if word in index2word_set:
            feature_vector[j, :]= np.stack(w2v_model.wv.word_vec(word))
            
    return feature_vector

def predict(text):

    text = text.lower()
    
    comment = [ord(xx) for xx in re.sub('[^\da-z가-힣]', '', text.strip())][:char_maxlen]
    
    for tmp_i, tmp_x in enumerate(comment):
        tmp_warning = True
        for s_num, e_num, t_num in [(32,32,0), (48,57,1), (97,122,1+10), (44032,55203,1+10+26)]:
            if tmp_x>=s_num and tmp_x<=e_num:
                comment[tmp_i] = tmp_x-s_num+t_num
                tmp_warning = False
                break
        if tmp_warning:
            print(tmp_i, ''.join([chr(xx) for xx in comment]))
            raise Warning

    if len(comment) <= char_maxlen:
        comment = ([ord(' ')-32] * (char_maxlen - len(comment))) + comment

    feature_vector = np.zeros((word_maxlen, 200))
    sentence = [xx for xx in tw.morphs(text, norm = True, stem = True) if xx in index2word_set]
    if len(sentence) > word_maxlen:
        sentence = sentence[:word_maxlen]
    for j, word in enumerate(sentence):
        if word in index2word_set:
            feature_vector[j, :]= np.stack(w2v_model.wv.word_vec(word))
    comment2 = feature_vector.reshape(1,27,200)
    ret = model.predict([np.array([comment]).reshape(1,48), comment2])

    text_feature = feature_model.predict([np.array([comment]).reshape(1,48), comment2])

    predict_result = np.argmax(ret)

    
    # emotion_list = {'ANGER': 10002, 'DISGUST' : 10003, 'FEAR' : 10004, 'HAPPINESS' : 10001, 'NEUTRAL' : 10005, 'SADNESS' : 10006, 'SURPRISE' : 10007}
    result = OrderedDict()
    result["10002"] = round(float(ret[0][0]),4)
    result["10003"] = round(float(ret[0][1]),4)
    result["10004"] = round(float(ret[0][2]),4)
    result["10001"] = round(float(ret[0][3]),4)
    result["10005"] = round(float(ret[0][4]),4)
    result["10006"] = round(float(ret[0][5]),4)
    result["10007"] = round(float(ret[0][6]),4)

    return result, text_feature

if __name__=='__main__':
    print("test text recognition")
    print(predict(text))
    print("Done!")
