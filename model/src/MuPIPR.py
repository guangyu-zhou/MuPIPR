import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

import sys
# from seq2tensor import s2t
import keras


from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional 
# Merge, 
from keras.layers import BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D

from keras.optimizers import Adam,  RMSprop

import keras.backend.tensorflow_backend as KTF

import numpy as np
from tqdm import tqdm

from keras.layers import Input, CuDNNGRU, GRU
from numpy import linalg as LA
import scipy
from sklearn.model_selection import KFold, ShuffleSplit
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

id2seq_file = '/workspace/PPI-Binding/iSee-master/processed/skempi_v1.trainAB.mut4.seq.txt'
# id2seq_file = '/workspace/PPI-Binding/BindProfX/processed/seq.txt'
# id2seq_file = '/workspace/PPI-Binding/iSee-master/processed/skempi_v1_v2.seq.txt'

# ds_file, label_index, rst_file, use_emb, hiddem_dim
ds_file = '/workspace/PPI-Binding/iSee-master/processed/4g_scores.txt'
label_index = 4
rst_file = 'results/temp.txt'
use_emb = 3
sid1_index = 0
sid2_index = 1
sid3_index = 2
sid4_index = 3
model_dim = 0
hidden_dim = 5
n_epochs = 5 

if len(sys.argv) > 1:
    ds_file, id2seq_file, label_index, rst_file, hidden_dim, n_epochs, model_dim, max_data = sys.argv[1:]
    label_index = int(label_index)
    hidden_dim = int(hidden_dim)
    n_epochs = int(n_epochs)
    model_dim = model_dim
    max_data = int(max_data)

id2index = {}
seqs = []
index = 0
for line in open(id2seq_file):
    line = line.strip().split('\t')
    id2index[line[0]] = index
    if len(line)==2:
        seqs.append(line[1])
    else:
        seqs.append("MMMMM") # random sequence: this is a quick fix to a bug....
    index += 1
seq_array = []
id2_aid = {}
sid = 0

# max_data = 30
limit_data = max_data > 0
raw_data = []
raw_ids = []
skip_head = False
x = None
count = 0    

## Serving contextualized embeddings of amino acids ================================


vocab_file='../../biLM/corpus/vocab.txt'
options_file='../../biLM/model/behm_'+model_dim+'skip_2l.ckpt/options.json'
weight_file='../../biLM/model/behm_'+model_dim+'skip_2l.hdf5'
token_embedding_file='../../biLM/model/vocab_embedding_'+model_dim+'skip_2l.hdf5'

print("Using options_file", options_file)
# options_file='../model/behm_3_2l.ckpt/options.json'
# weight_file='../model/behm_3_2l.hdf5'
# token_embedding_file='../model/vocab_embedding_dim3.hdf5'

# sequences = [['A', 'K','J','T','C','N'], ['C','A','D','A','A']]
## Now we can do inference.
# Create a TokenBatcher to map text to token ids.

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph. Adding reuse
# http://www.ifepoland.com/jiangxinyang/p/10235054.html
with tf.variable_scope("bilm", reuse=True):
    bilm = BidirectionalLanguageModel(
        options_file,
        weight_file,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file
    )

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)

def contextualize(sequences):
    batcher = TokenBatcher(vocab_file)


    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        # Create batches of data.
        context_ids = batcher.batch_sentences(sequences)

        # Compute ELMo representations (here for the input only, for simplicity).
        elmo_context_output_ = sess.run(
            [elmo_context_output['weighted_op']],
            feed_dict={context_token_ids: context_ids}
        )
    # print(np.array(elmo_context_output_).shape)
    # print(elmo_context_output_) #contextualized embedding vector sequences
    return elmo_context_output_

## Serving contextualized embeddings of amino acids ================================

def get_session(gpu_fraction=0.25):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def abs_diff(X):
    assert(len(X) == 2)
    s = X[0] - X[1]
    s = K.abs(s)
    return s

def abs_diff_output_shape(input_shapes):
    return input_shapes[0]

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    seq_input3 = Input(shape=(seq_size, dim), name='seq3')
    seq_input4 = Input(shape=(seq_size, dim), name='seq4')

    # l1=Conv1D(hidden_dim, 3)
    # r1=Bidirectional(GRU(hidden_dim, return_sequences=True))
    # l2=Conv1D(hidden_dim, 3)
    # r2=Bidirectional(GRU(hidden_dim, return_sequences=True))
    # l3=Conv1D(hidden_dim, 3)
    # r3=Bidirectional(GRU(hidden_dim, return_sequences=True))
    # l3_end=Conv1D(hidden_dim, 3)

    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    # l3_end=Conv1D(hidden_dim, 3)
    
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    r6=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    
    l_end=Conv1D(hidden_dim, 3)
    D1=Dense(100, activation='linear')
    # D2=Dense(1, activation='sigmoid')
    D2=Dense(1, activation='linear')

    D3=Dense(100, activation='linear')
    # D2=Dense(1, activation='sigmoid')
    D4=Dense(1, activation='linear')
    
    s1=MaxPooling1D(2)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(2)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])

    # s1=MaxPooling1D(3)(l4(s1))
    # s1=concatenate([r4(s1), s1])
    # s1=MaxPooling1D(3)(l5(s1))
    # s1=concatenate([r5(s1), s1])
    # s1=MaxPooling1D(3)(l6(s1))
    # s1=concatenate([r6(s1), s1])
    
    s1=l_end(s1)

    s1=GlobalAveragePooling1D()(s1)
    
    s2=MaxPooling1D(2)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(2)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])

    # s2=MaxPooling1D(3)(l4(s2))
    # s2=concatenate([r4(s2), s2])
    # s2=MaxPooling1D(3)(l5(s2))
    # s2=concatenate([r5(s2), s2])
    # s2=MaxPooling1D(3)(l6(s2))
    # s2=concatenate([r6(s2), s2])
    s2=l_end(s2)

    # s2=l3_end(s2)
    s2=GlobalAveragePooling1D()(s2)
    
    subtract_abs1 = keras.layers.Lambda(abs_diff, abs_diff_output_shape)
    
    merge_text1 = multiply([s1, s2])

    # merge_text2 = merge([s1, s2], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
    merge_text2 = subtract_abs1([s1,s2])
    
    merge_text_12 = concatenate([merge_text2, merge_text1])
    # merge_text_12 = multiply([s1, s2])
    # x12 = Dense(100, activation='linear')(merge_text_12)
    x12 = D1(merge_text_12)

    x12 = keras.layers.LeakyReLU(alpha=0.3)(x12)
    # main_output12 = Dense(1, activation='sigmoid')(x12)
    # main_output12 = Dense(1, activation='linear')(x12)
    main_output12 = D2(x12)
    
    
    s3=MaxPooling1D(2)(l1(seq_input3))
    s3=concatenate([r1(s3), s3])
    s3=MaxPooling1D(2)(l2(s3))
    s3=concatenate([r2(s3), s3])
    s3=MaxPooling1D(3)(l3(s3))
    s3=concatenate([r3(s3), s3])

    # s3=MaxPooling1D(3)(l4(s3))
    # s3=concatenate([r4(s3), s3])

    s3=l_end(s3)
    s3=GlobalAveragePooling1D()(s3)

    s4=MaxPooling1D(2)(l1(seq_input4))
    s4=concatenate([r1(s4), s4])
    s4=MaxPooling1D(2)(l2(s4))
    s4=concatenate([r2(s4), s4])
    s4=MaxPooling1D(3)(l3(s4))
    s4=concatenate([r3(s4), s4])

    # s4=MaxPooling1D(3)(l4(s4))
    # s4=concatenate([r4(s4), s4])

    s4=l_end(s4)
    s4=GlobalAveragePooling1D()(s4)
    
    subtract_abs2 = keras.layers.Lambda(abs_diff, abs_diff_output_shape)
    
    merge_text1 = multiply([s3, s4])

    # merge_text2 = merge([s1, s2], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
    merge_text2 = subtract_abs2([s3, s4])
    
    merge_text_34 = concatenate([merge_text2, merge_text1])

    # merge_text_34 = multiply([s3, s4])
    # x34 = Dense(100, activation='linear')(merge_text_34)
    x34 = D1(merge_text_34)

    x34 = keras.layers.LeakyReLU(alpha=0.3)(x34)
    # main_output34 = Dense(1, activation='sigmoid')(x34)
    # main_output34 = Dense(1, activation='linear')(x34)
    main_output34 = D2(x34)

    merge_text_1234 = concatenate([merge_text_12, merge_text_34])
    
    # main_output = main_output34 - main_output12
    # main_output = subtract([main_output34, main_output12])
    x1234 = D3(merge_text_1234)
    x1234 = keras.layers.LeakyReLU(alpha=0.3)(x1234)

    main_output = D4(x1234)
    
    # merge_model = Model(inputs=[seq_input1, seq_input2, seq_input3, seq_input4], outputs=[main_output12, main_output34])
    merge_model = Model(inputs=[seq_input1, seq_input2, seq_input3, seq_input4], outputs=[main_output12, main_output34, main_output])

    return merge_model

def scale_back(v):
    # if use_log:
    #     return np.exp(v * (all_max[dim] - all_min[dim]) + all_min[dim])
    # else:
    return v * (all_max - all_min) + all_min

KTF.set_session(get_session())


for line in tqdm(open(ds_file)):
    if skip_head:
        skip_head = False
        continue
    line = line.rstrip('\n').rstrip('\r').replace('\t\t','\t').split('\t')
    raw_ids.append((line[sid1_index], line[sid2_index], line[sid3_index], line[sid4_index]))
    if id2_aid.get(line[sid1_index]) is None:
        id2_aid[line[sid1_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid1_index]]])
    line[sid1_index] = id2_aid[line[sid1_index]]
    
    if id2_aid.get(line[sid2_index]) is None:
        id2_aid[line[sid2_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid2_index]]])
    line[sid2_index] = id2_aid[line[sid2_index]]

    if id2_aid.get(line[sid3_index]) is None:
        id2_aid[line[sid3_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid3_index]]])
    line[sid3_index] = id2_aid[line[sid3_index]]

    if id2_aid.get(line[sid4_index]) is None:
        id2_aid[line[sid4_index]] = sid
        sid += 1
        seq_array.append(seqs[id2index[line[sid4_index]]])
    line[sid4_index] = id2_aid[line[sid4_index]]

    raw_data.append(line)
    
    if limit_data:
        count += 1
        if count >= max_data:
            break

# print (len(raw_data))
# print (len(raw_data[0]))

len_m_seq = np.array([len(line) for line in seq_array])
avg_m_seq = int(np.average(len_m_seq)) + 1
max_m_seq = max(len_m_seq)
print (avg_m_seq, max_m_seq)


seq_index1 = np.array([line[sid1_index] for line in tqdm(raw_data)])
seq_index2 = np.array([line[sid2_index] for line in tqdm(raw_data)])
seq_index3 = np.array([line[sid3_index] for line in tqdm(raw_data)])
seq_index4 = np.array([line[sid4_index] for line in tqdm(raw_data)])

# print(seq_index4[:10])

print("Num of samples",len(raw_data))
print("seq_array", len(seq_array))
# print(seq_array, id2_aid)

batcher_size = 128
seq_tensor = []
max_seq_size = max_m_seq
for i in range(0, len(seq_array), batcher_size):
    print(i, min(len(seq_array),i+batcher_size))
    # contextualize(seq_array[i:min(len(seq_array),i+batcher_size)])
    # seq_array[i:min(len(seq_array),i+batcher_size)]
    cur_tensor = contextualize(seq_array[i:min(len(seq_array),i+batcher_size)])[0]
    if cur_tensor.shape[1] < max_seq_size:
        # print(cur_tensor.shape[1])
        npad = ((0, 0), (0, max_seq_size - cur_tensor.shape[1]), (0, 0))
        cur_tensor = np.pad(cur_tensor, pad_width=npad, mode='constant', constant_values=0)
        # print(cur_tensor.shape)
    if seq_tensor == []:
        seq_tensor = cur_tensor
    else:
        # print(seq_tensor.shape, cur_tensor.shape)
        seq_tensor = np.concatenate((seq_tensor, cur_tensor), axis = 0)
    # seq_tensor

# npad = ((0, 0), (1, 2), (2, 1))
# b = np.pad(a, pad_width=npad, mode='constant', constant_values=0)
print("seq_tensor shape",seq_tensor.shape)
seq_size, dim = seq_tensor.shape[1], seq_tensor.shape[2]
print("seq_size, dim", seq_size, dim)

num_scores = 3
score_labels = np.zeros((len(raw_data),num_scores))
# use_log = True
for i in range(len(raw_data)):
    score_labels[i] = raw_data[i][label_index:]

all_min, all_max = 99999999, -99999999
for i in range(len(raw_data)):
    score_labels[i] = raw_data[i][label_index:]

for j in range(2):
    min_j = min(score_labels[:,j])
    max_j = max(score_labels[:,j])
    if min_j < all_min:
        all_min = min_j
    if max_j > all_max:
        all_max = max_j

# ddG is normalized differently
for j in range(2):        
    score_labels[:,j] = (score_labels[:,j] - all_min )/(all_max - all_min)

score_labels[:,2] = (score_labels[:,2])/(all_max - all_min)    

print("All max, min",all_min, all_max)    

batch_size1 = 32
adam = Adam(lr=0.005, amsgrad=True, epsilon=1e-5)

# kf = ShuffleSplit(n_splits=5)
kf = KFold(n_splits=5, shuffle = True, random_state=13)

tries = 11
cur = 0
recalls = []
accuracy = []
total = []
total_truth = []
train_test = []
for train, test in kf.split(score_labels):
    train_test.append((train, test))
    # print (train[:10])
    cur += 1
    if cur >= tries:
        break

print ("train_test", len(train_test))
num_total = 0.
total_mse = 0.
total_mae = 0.
total_cov = 0.
# fp2 = open('records/muhao.'+rst_file[rst_file.rfind('/')+1:], 'w')

fp2 = open('records/pred_record_3G_test.'+rst_file[rst_file.rfind('/')+1:], 'w')
n_fold = 0

for train, test in train_test:
    print("In fold: ", n_fold)
    n_fold+=1
    merge_model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)

    merge_model.compile(optimizer=adam, loss='mse', metrics=['mse', pearson_r])

    # New instances: m1, w1; m2, w2; g2, g1 => ddg
    # new_seq_1 = np.concatenate((seq_index3[train],seq_index1[train]))
    # new_seq_2 = np.concatenate((seq_index4[train],seq_index2[train]))
    # new_labels = np.concatenate((score_labels[train,1], score_labels[train,0]))
    
    # merge_model.fit([seq_tensor[new_seq_1], seq_tensor[new_seq_2]], new_labels, batch_size=batch_size1, epochs=n_epochs, verbose = 0)
    # merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]], seq_tensor[seq_index3[train]], seq_tensor[seq_index4[train]]], [score_labels[train,2]], batch_size=batch_size1, epochs=n_epochs, verbose = 0)
    merge_model.fit([seq_tensor[seq_index1[train]], seq_tensor[seq_index2[train]], seq_tensor[seq_index3[train]], seq_tensor[seq_index4[train]]], [score_labels[train,0], score_labels[train,1], score_labels[train,2]], batch_size=batch_size1, epochs=n_epochs, verbose = 0)

    pred = merge_model.predict([seq_tensor[seq_index1[test]], seq_tensor[seq_index2[test]], seq_tensor[seq_index3[test]], seq_tensor[seq_index4[test]]])

    this_mae, this_mse, this_cov = 0., 0., 0.
    this_num_total = 0

    dG_pred0 = pred[0]
    dG_pred1 = pred[1]
    ddG_pred = pred[2]

    print("evaluating ......")
    for i in range(len(score_labels[test,num_scores-1])):
        this_num_total += 1
        # ddG_label_i = score_labels[test,2][i]
        # ddG_pred_i = ddG_pred[i] 
        ddG_label_i = (all_max - all_min)*score_labels[test,2][i]
        ddG_pred_i = (all_max - all_min)*ddG_pred[i]
        
        diff = abs(ddG_label_i - ddG_pred_i)
        this_mae += diff
        this_mse += diff**2


    num_total += this_num_total
    total_mae += this_mae
    total_mse += this_mse
    mse = total_mse / num_total
    mae = total_mae / num_total
    this_cov = scipy.stats.pearsonr(np.ndarray.flatten(ddG_pred), score_labels[test,num_scores-1])[0]
    # print("ddG: pred, truth",pred, score_labels[test,2])


    # fp2.write("id1\tid2\tid3\tid4\tdG0_predict\tdG0_label\tdG1_predict\tdG1_label\tddG_pred-scale\tddG_label-scale\n")
    for i in range(len(test)):
        fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index])  + '\t' + str(raw_ids[test[i]][sid3_index])  + '\t' + str(raw_ids[test[i]][sid4_index]) 
            + '\t' + str(scale_back(np.ndarray.flatten(dG_pred0)[i])) + '\t' + str(scale_back(score_labels[test[i], 0]))
            + '\t' + str(scale_back(np.ndarray.flatten(dG_pred1)[i])) + '\t' + str(scale_back(score_labels[test[i], 1]))
            + '\t' + str(np.ndarray.flatten((all_max - all_min)*ddG_pred)[i]) + '\t' + str((all_max - all_min)*score_labels[test[i], 2]) + '\n')
    # this_cov = scipy.stats.pearsonr(pred, score_labels[test,2])[0]
    # for i in range(len(test)):
        # fp2.write(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]) + '\n')
    # print(str(raw_ids[test[i]][sid1_index]) + '\t' + str(raw_ids[test[i]][sid2_index]) + '\t' + str(scale_back(np.ndarray.flatten(pred)[i])) + '\t' + str(scale_back(score_labels[test[i]])) + '\t' + str(np.ndarray.flatten(pred)[i]))
    total_cov += this_cov
    print (mse[0], this_cov, sep = '\t')
    # break
# fp2.close()
mse = total_mse / num_total
mae = total_mae / num_total
total_cov /= len(train_test)
print("Using options_file", options_file, hidden_dim, n_epochs)
print ("Average", mse[0], total_cov)

with open(rst_file, 'w') as fp:
    fp.write('mse=' + str(mse) + '\ncorr=' + str(total_cov))


# '''
