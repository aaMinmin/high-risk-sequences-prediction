import os
import random, csv
from statistics import mean
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers, Sequential
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
import math
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

THEANO_FLAGS = device = 'GPU:0'
devices = tf.config.list_physical_devices()
gpu_device = devices[0]
floatX = 'float32'
# from keras.preprocessing.sequence import Sequence
from keras.utils import Sequence
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from tensorflow.keras import backend as K
# Get the GPU device
gpu = tf.config.experimental.list_physical_devices("GPU")[0]

# Enable memory growth for the GPU device
tf.config.experimental.set_memory_growth(gpu, enable=True)
from tensorflow.keras import Model
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Masking, BatchNormalization, Flatten, \
    Softmax, Bidirectional, Activation, Permute, LayerNormalization,GRU,Conv1D,add
import pickle
import matplotlib.pyplot as plt
import keras.initializers
# from tensorflow.keras.distribute import Strategy
# path_ = os.path.dirname(__file__)
# canshu
path_ = "./"
batchsz = 64
max_review_len = 147
units = 64
epochs = 300
# strategy = Strategy()
dir_name = os.path.dirname(__file__)

with open(os.path.join(dir_name, "../", '../datamaker/4kmer_3w/traindata_3w.pkl'), 'rb+') as f1:
    seqs_train1 = pickle.load(f1)
f1.close()

#real_dataset
# with open(os.path.join(dir_name, '../', '../datamaker/software-4kmer/testdata.pkl'), 'rb+') as f3:
#     seqs_test1 = pickle.load(f3)
# f3.close()

#5%test
with open(os.path.join(dir_name, '../', '../datamaker/4kmer_3w/testdata_3w.pkl'), 'rb+') as f3:
    seqs_test1 = pickle.load(f3)
f3.close()


seq_train2 = []
score_train2 = []
for i in range(0, len(seqs_train1)):
    seq_train2.append(seqs_train1[i][0])  # 150长 Batch_size*150*10(10维)
    score_train2.append(seqs_train1[i][1])

score_train2 = [np.float16(x) for x in score_train2]
score_train2 = np.array(score_train2)

train_size = int(0.85 * len(seq_train2))
val_size = len(seq_train2) - train_size


train_dataset = tf.data.Dataset.from_tensor_slices((seq_train2[:train_size], score_train2[:train_size]))
train_dataset = train_dataset.batch(batchsz, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((seq_train2[train_size:], score_train2[train_size:]))
val_dataset = val_dataset.batch(batchsz, drop_remainder=True)

# seq_train2 = []
# score_train2 = []
# for i in range(0, len(seqs_train1)):
#     seq_train2.append(seqs_train1[i][0])  # 150长 Batch_size*150*10(10维)
#     score_train2.append(seqs_train1[i][1])

# score_train2 = [np.float32(x) for x in score_train2]
# score_train2 = np.array(score_train2)

seq_test2 = []
score_test2 = []
for j in range(0, len(seqs_test1)):
    seq_test2.append(seqs_test1[j][0])  # 150长 Batch_size*150*10(10维)
    score_test2.append(seqs_test1[j][1])

seq_test2 = [np.float32(x) for x in seq_test2]
seq_test2 = np.array(seq_test2)
score_test2 = [np.float32(x) for x in score_test2]
score_test2 = np.array(score_test2)

# #real_dataset
# with open(os.path.join(dir_name, 'Test_seq_relu-software.csv'), 'w', newline='') as f4:  # 实验结果保存的地方
#     writer = csv.writer(f4)
#     for z in seq_test2:
#         writer.writerow([z])
# #
# with open(os.path.join(dir_name, 'Truth_score_relu-software.csv'), 'w', newline='') as f6:  # 实验结果保存的地方
#     writer = csv.writer(f6)
#     for q in score_test2:
#         writer.writerow([q])

#5%test
with open(os.path.join(dir_name, '5%test/Test_seq_relu.csv'), 'w', newline='') as f4:  # 实验结果保存的地方
    writer = csv.writer(f4)
    for z in seq_test2:
        writer.writerow([z])

with open(os.path.join(dir_name, '5%test/Truth_score_relu.csv'), 'w', newline='') as f6:  # 实验结果保存的地方
    writer = csv.writer(f6)
    for q in score_test2:
        writer.writerow([q])


# seqs_train = tf.convert_to_tensor(seq_train2, dtype=tf.float32)
# scores_train = tf.convert_to_tensor(score_train2, dtype=tf.float32)

seqs_test = tf.convert_to_tensor(seq_test2, dtype=tf.float32)
# # 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
# db_train = tf.data.Dataset.from_tensor_slices((seqs_train, scores_train))
# db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
# db_test = tf.data.Dataset.from_tensor_slices((seqs_test))
# db_test = db_test.batch(batchsz, drop_remainder=True)


# Implement embedding layer.
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        lay2 = self.layernorm2(out1 + ffn_output)

        attn_output2 = self.att(lay2, lay2)
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(lay2 + attn_output2)
        ffn_output2 = self.ffn(out2)
        ffn_output2 = self.dropout2(ffn_output2, training=training)
        return self.layernorm2(out2 + ffn_output2)


def attention(inputs, d_model):
    Q = Dense(d_model)(inputs)
    K1 = Dense(d_model)(inputs)
    V = Dense(d_model)(inputs)
    d_k = d_model
    K_T = Permute((2, 1))(K1)
    scores = tf.matmul(Q, K_T / math.sqrt(d_k))
    alp = tf.nn.softmax(scores)
    context = tf.matmul(alp, V)
    output1 = K.sum(context, axis=1)
    return output1, alp


embed_dim = 10  # 32
num_heads = 8  # 16
ff_dim = 1024
maxlen = 147
vocab_size = 626

inputs = layers.Input(shape=(maxlen,))
inputs1 = Masking(mask_value=625, input_shape=(147,))(inputs)
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
embed_x = embedding_layer(inputs1)
lstm_out = Bidirectional(LSTM(units, use_bias=True, recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                              return_sequences=True), name='bilstm')(embed_x)
transformer_block = TransformerBlock(units * 2, num_heads, ff_dim)
# transformer_block = TransformerBlock(num_heads=8, d_model=1024,activation="relu")
x1 = transformer_block(lstm_out, training=True)
x, _ = attention(x1, units)
# x = layers.GlobalAveragePooling1D()(x1)
x = layers.Dropout(0.2)(x)
# x = LeakyReLU(alpha=0.1)(x)
# x = layers.Dense(32, activation="relu")(x)
# x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(32)(x)
x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1)(x)
model1 = tf.keras.Model(inputs=inputs, outputs=outputs)
model1.summary()

def r2(y_true, y_pred):
    rsquare = 1 - K.sum((y_true - y_pred) ** 2) / K.sum((y_true - K.mean(y_true)) ** 2)
    return rsquare


def get_huber_loss_fn(**huber_loss_kwargs):
    def custom_huber_loss(y_true, y_pred):
        return tf.losses.huber(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss


def loss_(weight):
    def loss_function(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=K.floatx())
        r2 = 1 - (1 - K.sum((y_true - y_pred) ** 2) / K.sum((y_true - K.mean(y_true)) ** 2))
        first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
        second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
        rmsle = K.sqrt(K.mean(K.square(first_log - second_log)))
        huber_loss = tf.losses.huber(y_true, y_pred)
        loss = 3 * r2 + 3 * huber_loss + 3 * rmsle
        return loss

    return loss_function


def mae(y_true, y_pred):
    maevalue = tf.losses.mae(y_true, y_pred)
    return maevalue


qall = []
qaall = []
r2all = []
# for i in range(n_epoch):
checkpoint_filepath = '../4-3/model2_epoch.289_valloss.1.5261_valR2.0.9435.hdf5'
checkpoint_filepath_many = os.path.join('../4-3/',
                                        'model2_epoch.{epoch:03d}_valloss.{val_loss:.4f}_valR2.{val_r2:.4f}.hdf5')
if os.path.exists(checkpoint_filepath):
    print("------------load Model!-----------")
    model1 = load_model(checkpoint_filepath, custom_objects={'loss_function': loss_(0.3), 'r2': r2,
                                                             'TokenAndPositionEmbedding': TokenAndPositionEmbedding(
                                                                 maxlen, vocab_size, embed_dim),
                                                             'TransformerBlock': TransformerBlock(embed_dim, num_heads,
                                                                                                  ff_dim)})
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath_many,
    #     save_weights_only=False,
    #     save_best_only=True
    # )
    # model1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
    #                #  loss=get_huber_loss_fn(delta=0.1),
    #                loss=loss_(0.3),
    #                metrics=[r2]
    #                )  # Adam
    # q = model1.fit(train_dataset, epochs=epochs, callbacks=[model_checkpoint_callback,keras.callbacks.CSVLogger('log.csv')], verbose=2,
    #                validation_data=val_dataset,use_multiprocessing=True,workers=2)

    cl = model1.predict(seqs_test, batch_size=1)
    R_2 = r2_score(score_test2, cl)
    print("R_2：")
    print(R_2)
    # MRE = K.abs(cl-scores_test)/scores_test
    # print("MRE：")
    # print(MRE)

else:
    print("------------Train Model!-----------")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath_many,
        save_weights_only=False,
        save_best_only=True
    )

    model1.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   #  loss=get_huber_loss_fn(delta=0.1),
                   loss=loss_(3),
                   metrics=[r2]
                   )  # Adam
    q = model1.fit_generator(train_dataset, epochs=epochs, callbacks=[model_checkpoint_callback,keras.callbacks.CSVLogger('log.csv')], verbose=2,
                   validation_data=val_dataset,use_multiprocessing=True,workers=2)

with open(os.path.join('../4-3/', '5%test/prediction.csv'), 'w', newline='') as f5:
    writer = csv.writer(f5)
    for a in cl:
        writer.writerow(a)
print(cl)