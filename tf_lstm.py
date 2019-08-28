'''
- - - - - - Team Members and IDs:- - - - - - - - 

Rushikesh Nalla - 111971011
Purvik Shah - 112045155
Sriram Gattu - 112007687
Kaustubh Sivalenka - 111987216

The purpose of this program is to run the lstm model to predict year average crop yield for year 2013. We have used hypothesis testing (permutataion test) to validate the results.
Hypothesis - "Are these features a good representation for predicting the crop yield"  

DataFrame used in this program is Tensorflow
Data Concept "Hypothesis testing" is used in this file

- - - - - - - Type of System used:- - - - - - - 

NVIDIA DGX-1 was used for computational requirements.

A Singularity image was built and was mounted to the working directory (on DGX-1) to use the requisite
software libraries to realize  project objectives.

https://drive.google.com/file/d/1WATPH8xna_xSQyvWrnMhAdm98rwYC5-L contains the singularity image
tf_writable.simg - Singularity image mounted to working directory.

'''

#Necessary libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading the histogram file generated using Spark
df = pd.read_pickle("trainingdata.txt")
#Removing erroneous data
df = df[df['Year'] != "2016"]
df = df.reset_index()

#Prediction year
ref_year = 2013
print("Prediction Year: ", ref_year)
#Splitting the data into lists
histograms = df['Histogram Array']
yields = df['Crop Yeild']
years = df['Year']

#Casting strings to float and integers
years = np.array(list(map(int, years)))
yields = np.array(list(map(float, yields)))

#indices for the prediction year
index_predict = np.nonzero(years == ref_year)[0]

#indices for all the years before the prediction year
index_train = np.nonzero(years < ref_year)[0]

#Length of both indices
print("Length of data for prediction year: ", len(index_predict))
print("Length of data for all the years before the prediction year: ", len(index_train))

#input and output for the prediction year
histogram_predict = []
yield_predict =  []
for x in index_predict:
    histogram_predict.append(histograms[x])
    yield_predict.append(yields[x])

histogram_predict = np.array(histogram_predict)
yield_predict  = np.array(yield_predict)


#Hyper parameters
batch_size = 32
bins = 32
days = 46
channels = 9
dense_size  = 128 #256
lstm_size = 128

#Defining placeholders and constants for input, output, and rates
inp = tf.placeholder(tf.float32, [None, bins, days, channels])
out = tf.placeholder(tf.float32, [None])
learning_rate = tf.constant(0.001, tf.float32)
keep_prob = tf.constant(0.8, tf.float32)
input_data = tf.reshape(inp, [days, -1, bins * channels])

'''
LSTM network using MultiRNNCell
Reference Tutorial - https://medium.com/@erikhallstrm/using-the-tensorflow-multilayered-lstm-api-f6e7da7bbe40
The multi-layered LSTM is created by first making a single LSMTCell, 
applying the dropout wrapper and then duplicating this cell, and supplying it to the MultiRNNCell.
Dynamic RNN outputs the last state of every layer in the network as an LSTMStateTuple and a tensor
containing the hidden state of the last layer across all time-steps.
'''

lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple = True, activation="tanh")
lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = 1, output_keep_prob = keep_prob)
lstm = tf.contrib.rnn.MultiRNNCell([lstm] * 1, state_is_tuple=True)

states_series, current_state = tf.nn.dynamic_rnn(lstm, input_data, initial_state = lstm.zero_state(batch_size, tf.float32), time_major=True)
states_series = tf.reshape(tf.slice(states_series, [days-1, 0, 0] , [1, -1, -1]), [batch_size, dense_size])

#Fully connected layer - 1
N = states_series.get_shape()[-1]
W = tf.Variable(np.random.rand(N, dense_size),dtype=tf.float32)
b = tf.Variable(np.zeros((1, dense_size)), dtype=tf.float32)
fc1 = tf.matmul(states_series, W) + b

#Fully connected layer - 2
N2 = fc1.get_shape()[-1]
W2 = tf.Variable(np.random.rand(N2, 1),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, 1)), dtype=tf.float32)
another_fc = tf.matmul(fc1, W2)  + b2

pred = tf.reshape(another_fc, [batch_size,]) #predictions
loss = tf.nn.l2_loss(pred - out) #loss
optimized = tf.train.AdamOptimizer(learning_rate).minimize(loss) #Applying Adam optimizer

num_steps = 1000
sess = tf.Session()
sess.run(tf.initialize_all_variables())

def run_network(shuffle = False):
    train_loss_log = []
    RMSE_log = []
    ME_log = []
    run_loss=0
    pred_log  = []
    true_log = []
    global_count = 0

    RMSE_min = 9999
    try:
        for i in range(num_steps):
            #Batch indices picked from the training indices
            training_batch_indices  = np.random.choice(index_train, size = batch_size)
            training_batch_indices = np.array(training_batch_indices)
            
            #input and output for training
            inp_arr =  []
            out_arr = []
            for x in training_batch_indices:
                inp_arr.append(histograms[x])
                out_arr.append(yields[x])
            inp_arr = np.array(inp_arr)
            out_arr = np.array(out_arr)
            
            #running the tensorflow network
            opt, run_loss = sess.run([optimized, loss], feed_dict = {inp:inp_arr, out:out_arr})
            
            #Calculating loss after a specified interval
            #Generating predictions on validation data as well
            if i%200 == 0:
                
                y_pred = []
                y_true = []
                steps = int(histogram_predict.shape[0] / batch_size)
                for j in range(steps):
                    temp_true = yield_predict[j * batch_size: (j + 1) * batch_size]
                    inp_arr = []
                    out_arr = []
                    for x in range(j * batch_size, (j + 1) * batch_size):
                        inp_arr.append(histogram_predict[x])
                        out_arr.append(yield_predict[x])
                        
                    inp_arr = np.array(inp_arr)
                    out_arr = np.array(out_arr)

                    #Shuffling the yield array randomly in case of hypothesis testing
                    if(shuffle):
                        np.random.shuffle(out_arr)
                    
                    temp_pred = sess.run(pred, feed_dict = {inp: inp_arr, out: out_arr})
                    y_pred.append(temp_pred)
                    y_true.append(temp_true)
                
                y_pred = np.array(y_pred)
                y_true = np.array(y_true)

                #Calculating loss
                RMSE = np.sqrt(np.mean(np.power((y_pred  - y_true), 2)))
                ME = np.mean(y_pred - y_true)
                
                if RMSE < RMSE_min:
                    RMSE_min = RMSE
                if(i >= num_steps*0.8):
                    pred_log = y_pred
                    true_log = y_true
                
                train_loss_log.append(run_loss)
                RMSE_log.append(RMSE)
                ME_log.append(ME)
                
    except KeyboardInterrupt:
        print ('Interrupted')

    return RMSE_log, ME_log, pred_log, true_log


RMSE_log, ME_log, pred_log, true_log = run_network()
print("Root mean squared error", RMSE_log)
print("Mean error", ME_log)

#permutation testing
multiple_RMSE = []
besterror = min(RMSE_log)
num_runs = 20

for i in range(num_runs):
    RMSE_log, ME_log, pred_log, true_log  = run_network(True)
    multiple_RMSE.append(RMSE_log[-1])
    print("Run number ", i)
    print("RMSE ", RMSE_log[-1])

print("RMSE for all runs: ", multiple_RMSE)

count = 0
for i in range(num_runs):
    if(multiple_RMSE[i] < besterror):
        count  = count + 1

print("Count", count)
print("p-val", count/num_runs)

plt.hist(x=multiple_RMSE, bins=20, range=[9.5, 12], color='orangered', alpha=0.75, rwidth=0.999, label='RMSE histogram')
plt.axvline(besterror, color='blue', label='Best error from LSTM')
plt.xticks(np.arange(min(multiple_RMSE), max(multiple_RMSE)+1, 0.5))
plt.legend()
plt.xlabel('RMSE Errors for permutation test')
plt.ylabel('Frequency')
plt.title('Histogram Plot for RMSE Errors of various permutations');
plt.savefig('permutation_test_lstm.png')
print(plt.show())