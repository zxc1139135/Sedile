from mpi4py import MPI
import numpy as np
import random
from array import array
import math
import time
import sys
import gc
import os
import csv
import phone



import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pylab as plt
import pickle as pickle

from utils.mpc_function import *
from utils.polyapprox_function import *

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print ("hello world from process", rank)

# system parameters
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) == 1:
    if rank == 0:
        print("ERROR: please input the number of workers")
    exit()
else:
    N = int(sys.argv[1])

K_ = int(np.floor((N - 1) / float(3))) + 1 - int(np.floor((N - 3) / float(6)))
T_ = int(np.floor((N - 3) / float(6)))

# learning parameters
max_iter = 3
layers = 18
# set the seed of the random number generator for consistency
np.random.seed(40)

# quantized dataset mod p and parameters
p = 2 ^ 26 - 5
q_bit_X = 1
q_bit_y = 0

# secure truncation protocol parameters
alpha_exp = 15
coeffs0_exp = 1
coeffs1_exp = 6
trunc_scale = alpha_exp + coeffs1_exp - q_bit_y
trunc_k, trunc_m = 24, trunc_scale

# dimension (dataset model parameter) d = 3037
d_number = 1
"""
dataset: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN, GISETTE
models: a two-layer MLP model; LeNet-5, AlexNet, VGG16, ResNet18
parameter: 
MNIST / MLP: 79,610 (26)
MNIST / LeNet-5: 61,000 (20)

Fashion-MNIST / MLP: 79,610 (26)
Fashion-MNIST / LeNet-5: 61,000 (20)

GISETTE / MLP: 79,610 (26)
GISETTE / LeNet-5: 61,000 (20)


CIFAR-10 / AlexNet: 1,074,986 (354)
CIFAR-10 / VGG16: 14,700,000 (4840)
CIFAR-10 / ResNet18: 11,219,328 (3694)

CIFAR-100 / AlexNet: 1,074,986 (354)
CIFAR-100 / VGG16: 14,700,000 (4840)
CIFAR-100 / ResNet18: 11,219,328 (3694)


SVHN / AlexNet: 6,700,000 (2206)
SVHN / VGG16: 10,400,000 (3424)
SVHN / ResNet18: 11,173,962 (3687)

"""

if rank == 0:
    print('Hi from crypto-service provider', 'rank', rank)

    filename = './datasets/SVHN.pickle'
    with open(filename, 'rb') as f:
        other = pickle.load(f)
        # print(other)
        train_data = other['train_dataset']
        test_data = other['test_dataset']
        # print(test_data)

        del other

    train_dataset = train_data['X']
    test_dataset = test_data['X']
    # print(train_dataset)
    # train_data = np.transpose(train_data, (3, 0, 1, 2))
    train_labels = train_data['y']
    test_labels = test_data['y']

    print(len(train_dataset))
    print(len(train_labels))
    print(train_dataset.shape)
    print(len(test_dataset))
    print(len(test_labels))
    print(test_dataset.shape)

    X_train = np.reshape(train_dataset, (train_dataset.shape[0], -1))
    print(X_train.shape)
    y_train = np.reshape(train_labels, (train_labels.shape[0], -1))
    print(y_train.shape)
    X_test = np.reshape(test_dataset, (test_dataset.shape[0], -1))
    print(X_test.shape)
    y_test = np.reshape(test_labels, (test_labels.shape[0], -1))
    print(y_test.shape)


    # start timer
    t0_read = time.time()


    # extract the first m rows
    X = X_train
    X_test = X_test

    # reshape row vector into a column vector
    m, d = X.shape
    m = m * 10
    y = np.reshape(y_train, (m, 1))
    y_test = np.reshape(y_test, (len(y_test)*10, 1))

    # release the memory
    X_train = None
    y_train = None
    # X_val = None
    # y_val = None

    # time spent in reading dataset
    t_read = time.time() - t0_read

    print('Time spent for reading dataset (sec):', t_read)
    print('Train data shape: ', X.shape)
    print('Train labels shape: ', y.shape)

    time_out = []
    # number of submatrices split
    K = K_
    T = T_
    print('(K,T)=', K, T)

    # remove extra data points so that m is divisible by k, i.e., put data suitable for HMC format
    m = X.shape[0] - (X.shape[0] % K)

    # extract the first m rows
    X = X[:m]
    # extract the first m elements
    y = y[:m]
    # reshape row vector into a column vector
    y = np.reshape(y, (m, 1))

    t0_offline = time.time()

    print('01.Data conversion: real to finite field')
    t0_q = time.time()
    # X_q: matrix with size (m by d)
    X_q = my_q(X, q_bit_X, p)

    # time spent in reading dataset
    t_q = time.time() - t0_q
    print('Time spent in reading dataset (s):', t_q)

    q_bit_y = 1
    y_scale = ((2 ** q_bit_y) * y).astype('int64')

    print('02. Secret Shares generation in finite field')
    t0 = time.time()

    # X_SS_T = HMC_encoding(X_q, N, K, T, p)
    X_SS_T = Harmonic_encoding(X_q, N, T, p)
    t_gen_X_SS_T = time.time() - t0
    print('Time spent in secret shares generation (s):', t_gen_X_SS_T)

    total_size_data_X_T = 0

    for j in range(1, N + 1):
        # send data in vector format
        data_X_T = np.reshape(X_SS_T[j - 1, :, :], d * m)
        # send number of rows =  number of training samples
        comm.send(m, dest=j)
        # send number of columns = number of features
        comm.send(d, dest=j)
        # sent data to worker j
        comm.Send(data_X_T, dest=j)
        total_size_data_X_T += len(data_X_T) * 8 / 1024 / 1024
    data_X_T, X_SS_T = None, None
    gc.collect()

    print('03. Random matrix and corresponding SS generation')
    r_mult1 = np.random.randint(p, size=(m, 1))
    r_mult1_SS_T = Harmonic_encoding(r_mult1, N, T, p)
    # r_mult1_SS_T = HMC_encoding(r_mult1, N, K, T, p)
    r_mult1_SS_2T = Harmonic_encoding(r_mult1, N, 2 * T, p)
    # r_mult1_SS_2T = HMC_encoding(r_mult1, N, K, 2*T, p)

    r_mult2 = np.random.randint(p, size=(d, 1))
    r_mult2_SS_T = Harmonic_encoding(r_mult2, N, T, p)
    # r_mult2_SS_T = HMC_encoding(r_mult2, N, K, T, p)
    r_mult2_SS_2T = Harmonic_encoding(r_mult2, N, 2 * T, p)
    # r_mult2_SS_2T = HMC_encoding(r_mult2, N, K, 2*T, p)

    r1 = np.random.randint(2 ** trunc_m, size=(d, 1))
    r2 = np.random.randint(2 ** (trunc_k - trunc_m), size=(d, 1))

    r1_Harmonic = Harmonic_encoding(r1, N, T, p)
    # r1_Harmonic = HMC_encoding(r1, N, K, T, p)
    r2_Harmonic = Harmonic_encoding(r2, N, T, p)
    # r2_Harmonic = HMC_encoding(r2, N, K, T, p)

    # initialize model parameters

    w = (1 / float(m)) * np.random.rand(d, d_number)
    # print(w.shape)
    # w = np.reshape(w, (d*3, 1))
    # print(w.shape)

    w_q_tmp = my_q(w, 0, p)
    w_SS_T = Harmonic_encoding(w_q_tmp, N, T, p)
    # w_SS_T = HMC_encoding(w_q_tmp, N, K, T, p)

    # random matrix for HMC encoding
    R_HMC = np.random.randint(p, size=(T, m // K, d))
    r_HMC = np.random.randint(p, size=(T, d, d_number))

    # generation Secret shares of the random matrix
    R_HMC_SS_T = np.empty((N, T, m // K, d), dtype='int64')
    for t in range(T):
        R_HMC_SS_T[:, t, :, :] = Harmonic_encoding(R_HMC[t, :, :], N, T, p)
        # R_HMC_SS_T[:,t,:,:] = HMC_encoding(R_HMC[t,:,:], N, K, T, p)

    r_HMC_SS_T = np.empty((N, T, d, d_number), dtype='int64')
    for t in range(T):
        r_HMC_SS_T[:, t, :, :] = Harmonic_encoding(r_HMC[t, :, :], N, T, p)
        # r_HMC_SS_T[:,t,:,:] = HMC_encoding(r_HMC[t,:,:], N, K, T, p)

    t0_CSP_send_SS = time.time()

    print('(m, d, K, T, m//K)=', m, d, K, T, m // K)

    total_size_data_y = 0
    total_size_data_w_T = 0
    total_size_data_R1_T = 0
    total_size_data_R1_2T = 0
    total_size_data_R2_T = 0
    total_size_data_R2_2T = 0
    total_size_data_r1_T = 0
    total_size_data_r2_T = 0
    total_size_data_R_HMC_T = 0
    total_size_data_r_HMC_T = 0

    # Sending data to workers in the preprocessing phase
    for j in range(1, N + 1):
        # print('Sending data to worker', j)
        # send data in vector format (np.reshape)
        data_y = np.reshape(y_scale, m)
        data_w_T = np.reshape(w_SS_T[j - 1, :, :], d * d_number)
        data_R1_T = np.reshape(r_mult1_SS_T[j - 1, :, :], m)
        data_R1_2T = np.reshape(r_mult1_SS_2T[j - 1, :, :], m)
        data_R2_T = np.reshape(r_mult2_SS_T[j - 1, :, :], d)
        data_R2_2T = np.reshape(r_mult2_SS_2T[j - 1, :, :], d)

        data_r1_T = np.reshape(r1_Harmonic[j - 1, :, :], d)
        data_r2_T = np.reshape(r2_Harmonic[j - 1, :, :], d)

        data_R_HMC_T = np.reshape(R_HMC_SS_T[j - 1, :, :, :], T * (m // K) * d)
        data_r_HMC_T = np.reshape(r_HMC_SS_T[j - 1, :, :, :], T * d * d_number)

        # sent data to worker j
        comm.Send(data_y, dest=j)
        comm.Send(data_w_T, dest=j)
        comm.Send(data_R1_T, dest=j)
        comm.Send(data_R1_2T, dest=j)
        comm.Send(data_R2_T, dest=j)
        comm.Send(data_R2_2T, dest=j)

        comm.Send(data_r1_T, dest=j)
        comm.Send(data_r2_T, dest=j)

        comm.Send(data_R_HMC_T, dest=j)
        comm.Send(data_r_HMC_T, dest=j)

        total_size_data_y += len(data_y) * 8 / 1024 / 1024
        total_size_data_w_T += len(data_w_T) * 8 / 1024 / 1024
        total_size_data_R1_T += len(data_R1_T) * 8 / 1024 / 1024
        total_size_data_R1_2T += len(data_R1_2T) * 8 / 1024 / 1024
        total_size_data_R2_T += len(data_R2_T) * 8 / 1024 / 1024
        total_size_data_R2_2T += len(data_R2_2T) * 8 / 1024 / 1024
        total_size_data_r1_T += len(data_r1_T) * 8 / 1024 / 1024
        total_size_data_r2_T += len(data_r2_T) * 8 / 1024 / 1024
        total_size_data_R_HMC_T += len(data_R_HMC_T) * 8 / 1024 / 1024
        total_size_data_r_HMC_T += len(data_r_HMC_T) * 8  / 1024 / 1024

    comm.Barrier()

    t_CSP_send_SS = time.time() - t0_CSP_send_SS
    t_offline = time.time() - t0_offline

    print('[crypto-service provider] sending X_SS_T & random SS is done')
    print('[crypto-service provider] Offline Time=', t_offline, ', sending SS in offline phase=', t_CSP_send_SS)

    data_y, y_scale, data_w_T, w_SS_T = None, None, None, None
    data_R1_T, data_R1_2T, data_R2_T = None, None, None
    data_R2_2T, data_r1_T, data_r2_T = None, None, None
    data_R_HMC_T, data_r_HMC_T, X_SS_T, data_X_T = None, None, None, None
    R_HMC_SS_T, r_HMC_SS_T, r1_Harmonic, r2_Harmonic = None, None, None, None

    print('start garbage collection')
    gc.collect()
    print('garbage collection is done')
    # test model
    # y_hat = np.dot(data_w_T, X_test)
    # test_function(y_hat, y_test)

    N_time_set = 110
    time_set_workers = np.empty((N, N_time_set), dtype='float')
    for j in range(1, N + 1):
        comm.Recv(time_set_workers[j - 1, :], source=j)

    N_size_set = 110
    size_set_workers = np.empty((N, N_size_set), dtype='float')
    for j in range(1, N + 1):
        comm.Recv(size_set_workers[j - 1, :], source=j)

    total_time = time.time() - t0_offline
    print('[crypto-service provider] Total time = ', total_time)

    time_set = {'K': K,
                'T': T,
                '[crypto-service provider] total_time': total_time,
                't_CSP_send_SS_time': t_CSP_send_SS,
                't_offline_time': t_offline,
                't_gen_X_SS_T_time': t_gen_X_SS_T,
                'time_set_workers_time': time_set_workers}

    T_workers = np.sum(time_set_workers, axis=0) / N
    S_workers = np.sum(size_set_workers, axis=0)

    print('gen HMC=', T_workers[0])
    print('each iteration')
    print('gen w_HMC=', T_workers[3])
    print('f_eval=', T_workers[4])
    print('gen f_eval_SS=', T_workers[5] + T_workers[6])
    print('multiplication=', T_workers[1] + T_workers[2])
    print('communication =', T_workers[6] + T_workers[7] + T_workers[8])
    print('Preprocessing in workers (from sum)=', T_workers[0] + T_workers[1] + T_workers[2])
    print('Main Loop total time (from sum)=', np.sum(T_workers[3:9]) - T_workers[7])
    print('From workers: preprocessing =', T_workers[9])
    print('From workers: Main Loop    =', T_workers[10] * layers)
    print('N,K,T', N, K, T)
    print('CTT =', T_workers[11] * layers)
    print('SAT =', T_workers[12] * layers)
    print('CUT =', T_workers[13] * layers)
    print('TTT =', T_workers[10] * layers)
    data = [["{:.4f}".format(T_workers[11] * layers), "{:.4f}".format(T_workers[12] * layers),
             "{:.4f}".format(T_workers[13] * layers), "{:.4f}".format(T_workers[10] * layers)]]

    # open result.csv and write result
    with open('result.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("main loop communication = ", (S_workers[1] + S_workers[2]) * layers)
    total_size = total_size_data_X_T + total_size_data_y + total_size_data_w_T + \
                 total_size_data_R1_T + total_size_data_R1_2T + total_size_data_R2_T + total_size_data_R2_2T + total_size_data_r1_T + \
                 total_size_data_r2_T + total_size_data_R_HMC_T + total_size_data_r_HMC_T + S_workers[0]
    print("offline total communication = ", total_size)
    print("preprocess total communication = ", S_workers[0])

    time_out.append(time_set)
    comm.Barrier()
    pickle.dump(time_out, open('./PPDML_SVHN_' + str(N), 'wb'), -1)

elif rank <= N:
    def MPI_TruncPr(in_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_m, T, p):
        t0 = time.time()
        a_SS_T = in_SS_T.astype('int64')
        trunc_size = np.prod(a_SS_T.shape)

        a_SS_T = np.reshape(a_SS_T, trunc_size)
        r1_SS_T = np.reshape(r1_SS_T, trunc_size)
        r2_SS_T = np.reshape(r2_SS_T, trunc_size)

        t1 = time.time()
        b_SS_T = np.mod(a_SS_T + 2 ** (trunc_k - 1), p)
        r_SS_T = np.mod((2 ** trunc_m) * r2_SS_T + r1_SS_T, p)
        c_SS_T = np.mod(b_SS_T + r_SS_T, p)
        # print ('rank=',rank, c_SS_T.shape)

        t2 = time.time()
        dec_input = np.empty((T + 1, trunc_size), dtype='int64')
        for j in range(1, T + 2):
            if rank == j:
                dec_input[j - 1, :] = c_SS_T
                # secret share q
                for j in list(range(1, rank)) + list(range(rank + 1, N + 1)):
                    data = c_SS_T
                    # sent data to worker j
                    comm.Send(data, dest=j)
            else:
                data = np.empty(trunc_size, dtype='int64')
                comm.Recv(data, source=j)
                # coefficients for the polynomial
                dec_input[j - 1, :] = data

        t3 = time.time()

        c_dec = Harmonic_decoding(dec_input, range(T + 1), p)
        # c_dec = HMC_decoding(dec_input, N, K, T, range(T+1), p)
        print('rank=', rank, 'c_dec is completed', c_dec.shape)

        t4 = time.time()
        c_prime = np.mod(np.reshape(c_dec, trunc_size), 2 ** trunc_m)
        a_prime_SS_T = np.mod(c_prime - r1_SS_T, p)
        d_SS_T = np.mod(a_SS_T - a_prime_SS_T, p)

        t5 = time.time()
        d_SS_T = divmod(d_SS_T, 2 ** trunc_m, p)
        d_SS_T = np.reshape(d_SS_T, in_SS_T.shape)

        t6 = time.time()
        time_set = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5])
        print('time info for trunc pr', time_set)
        return d_SS_T.astype('int64')


    # number of submatrices split
    K = K_
    T = T_

    # end of function definition
    print('Hi from worker,', 'rank', rank)

    # number of rows =  number of training samples
    m = comm.recv(source=0)
    # number of columns  = number of features
    d = comm.recv(source=0)

    # allocate space to receive the matrix
    data = np.empty(m * d, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    X_SS_T = np.reshape(data, (m, d))

    # allocate space to receive the matrix
    data = np.empty(m * 1, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    y_scale = np.reshape(data, (m, 1))

    # allocate space to receive the matrix
    data = np.empty(d * d_number, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    w_SS_T = np.reshape(data, (d, d_number))

    # allocate space to receive the matrix
    data = np.empty(m * 1, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r_SS_T = np.reshape(data, (m, 1))

    # allocate space to receive the matrix
    data = np.empty(m * 1, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r_SS_2T = np.reshape(data, (m, 1))

    # allocate space to receive the matrix
    data = np.empty(d * 1, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r_mult2_SS_T = np.reshape(data, (d, 1))

    # allocate space to receive the matrix
    data = np.empty(d * 1, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r_mult2_SS_2T = np.reshape(data, (d, 1))

    # allocate space to receive the matrix
    data = np.empty(d * d_number, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r1_SS_T = np.reshape(data, (d, d_number))

    # allocate space to receive the matrix
    data = np.empty(d * d_number, dtype='int64')
    comm.Recv(data, source=0)
    # coded matrix
    r2_SS_T = np.reshape(data, (d, d_number))

    data = np.empty(T * (m // K) * d, dtype='int64')
    comm.Recv(data, source=0)
    # random matrix for HMC encoding of X
    R_HMC_SS_T = np.reshape(data, (T, m // K, d))

    data = np.empty(T * d * d_number, dtype='int64')
    comm.Recv(data, source=0)
    # random matrix for HMC encoding of w
    r_HMC_SS_T = np.reshape(data, (T, d, d_number))

    print('data received! rank=', rank)
    comm.Barrier()

    # ------------------------------------------
    #       Preprocessing Starts Here.        -
    # ------------------------------------------

    # Group setting for HMC encoding & decoding
    # each group has (d-1) clients
    pre_total_size_data_x = 0
    if np.mod(N, T + 1) == 0:
        group_id = int(rank - 1) // int(T + 1)
        group_idx_set = range(group_id * (T + 1), (group_id + 1) * (T + 1))
    else:
        group_id = int(rank - 1) // int(T + 1)
        last_group_id = int(N) // int(T + 1)
        if (group_id == last_group_id) | (group_id == last_group_id - 1):
            group_idx_set = range((last_group_id - 1) * (T + 1), N)
        else:
            group_idx_set = range(group_id * (T + 1), (group_id + 1) * (T + 1))
    group_stt_idx = group_idx_set[0]
    group_idx_set_others = [idx for idx in group_idx_set if rank - 1 != idx]
    my_worker_idx = rank - 1
    # end of group setting

    # Preprocessing 1. HMC encoding of X
    # input  : X_SS_T (=secret share of X= [X]_i)
    # output : X_HMC (=\widetiled{X}_i)

    # 1.1. generate the secret share of encoded X
    t0_HMC_encoding_X = time.time()
    X_HMC_T = HMC_encoding_W_Random_partial(X_SS_T, R_HMC_SS_T, N, K, T, p, group_idx_set)
    t_HMC_encoding_X_onlyencoding = time.time() - t0_HMC_encoding_X

    # 1.2. sending the secret share of encoded X
    t0_comm_X_HMC = time.time()
    dec_input = np.empty((len(group_idx_set), (m // K) * d), dtype='int64')

    for j in group_idx_set:
        if my_worker_idx == j:
            dec_input[my_worker_idx - group_stt_idx, :] = np.reshape(X_HMC_T[my_worker_idx - group_stt_idx, :, :],
                                                                     (m // K) * d)
            for idx in group_idx_set_others:
                # print ('from',rank,' to ',idx+1)
                data = np.reshape(X_HMC_T[idx - group_stt_idx, :, :], (m // K) * d)
                # sent data to worker j
                comm.Send(data, dest=idx + 1)
                pre_total_size_data_x += len(data) * 8 / 1024 / 1024
        else:
            data = np.empty((m // K) * d, dtype='int64')
            comm.Recv(data, source=j + 1)
            # coefficients for the polynomial
            dec_input[j - group_stt_idx, :] = data
    # print ('dec_input info (af comm)=',dec_input[:,0])
    t_comm_X_HMC = time.time() - t0_comm_X_HMC

    # 1.3. reconstruct the secret : get X_HMC
    X_HMC_dec = Harmonic_decoding(dec_input, group_idx_set, p)
    # X_HMC_dec = HMC_decoding(dec_input, N, K, T, group_idx_set, p)
    X_HMC = np.reshape(X_HMC_dec, (m // K, d)).astype('int64')
    t_HMC_encoding_X = time.time() - t0_HMC_encoding_X

    print('time info for gen X_HMC', t_HMC_encoding_X_onlyencoding, t_comm_X_HMC, t_HMC_encoding_X)

    # Preprocessing 2. Calculate common terms
    t0_XTX = time.time()
    # XTX_HMC = np.random.randint(p,size=(d,d)).astype('int64')
    XTX_HMC = X_HMC.T.dot(X_HMC)
    t_XTX = time.time() - t0_XTX

    t0_XTy = time.time()
    c0_m_y = np.int64(2 ** (q_bit_y + coeffs1_exp - coeffs0_exp) - (2 ** coeffs1_exp) * y_scale)
    XTy_SS_T = X_SS_T.T.dot(c0_m_y)
    t_XTy = time.time() - t0_XTy

    t_preprocessing = time.time() - t0_HMC_encoding_X

    # -------------------------------------------
    #       Preprocessing Ends Here.           -
    # -------------------------------------------

    # -------------------------------------------
    #           Main Loop Starts Here.         -
    # -------------------------------------------

    # set parameters
    iter = 0
    hist_w_SS_T = np.empty((max_iter + 1, d * d_number), dtype='int64')
    hist_w_SS_T[0, :] = np.reshape(w_SS_T, d * d_number)

    t_HMC_encoding_w, t_f_eval, t_gen_f_SS, t_gen_grad_SS, t_comm_f_eval_SS, t_trunc, t_comm_w = 0, 0, 0, 0, 0, 0, 0
    CTT, SAT, CUT = 0, 0, 0
    t0_mainloop = time.time()
    main_total_size_data_w = 0
    main_total_size_data_f = 0

    while (iter < max_iter):
        folder_path = '/dev/shm'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        iter = iter + 1
        print('iter=', iter)

        CTT_0 = time.time()
        # 1. HMC encoding of w(t)
        # input  : w_SS_T (=secret share of w(t)= [w(t)]_i)
        # output : w_HMC (=\widetiled{w}^{(t)}_i)

        # 1.1 generate the secret share of encoded w
        t0_HMC_encoding_w = time.time()
        # w_rep: repeated vector with size ( d*K by 1 )
        w_rep_SS_T = np.transpose(np.tile(np.transpose(w_SS_T), K))
        print("w_rep_SS_T :", w_rep_SS_T.shape)
        print("r_HMC_SS_T: ", r_HMC_SS_T.shape)
        w_HMC_SS_T = HMC_encoding_w_Random_partial(w_rep_SS_T, r_HMC_SS_T, N, K, T, p, group_idx_set)
        print("w_HMC_SS_T: ", w_HMC_SS_T.shape)

        # print(type(w_HMC_SS_T[0,0,0]), np.max(w_HMC_SS_T))

        # 1.2. sending the secret share of encoded w
        dec_input = np.empty((len(group_idx_set), d * d_number), dtype='int64')
        t0_comm_w = time.time()
        for j in group_idx_set:
            if my_worker_idx == j:
                dec_input[my_worker_idx - group_stt_idx, :] = np.reshape(
                    w_HMC_SS_T[my_worker_idx - group_stt_idx, :, :], d * d_number)
                for idx in group_idx_set_others:
                    # print('from',rank,' to ',idx+1)
                    data = np.reshape(w_HMC_SS_T[idx - group_stt_idx, :, :], d * d_number)
                    # sent data to worker j
                    comm.Send(data, dest=idx + 1)
            else:
                data = np.empty(d * d_number, dtype='int64')
                comm.Recv(data, source=j + 1)
                # coefficients for the polynomial
                dec_input[j - group_stt_idx, :] = data
        t_comm_w += time.time() - t0_comm_w

        # 1.3. reconstruct the secret : get w_HMC
        w_HMC_dec = Harmonic_decoding(dec_input, group_idx_set, p)
        # w_HMC_dec = HMC_decoding(dec_input, N, K, T, group_idx_set, p)
        w_HMC = np.reshape(w_HMC_dec, (d, d_number)).astype('int64')

        t_HMC_encoding_w += time.time() - t0_HMC_encoding_w

        # 2. compute f over HMC_encoded inputs
        t0_f_eval = time.time()
        f_eval = np.dot(XTX_HMC, w_HMC)
        # print(f_eval)
        # quit()
        t_f_eval = + time.time() - t0_f_eval

        # 3. generate the secret shares of f_eval
        t0_gen_f_SS = time.time()
        f_eval_SS_T = Harmonic_encoding(f_eval, N, T, p)
        # f_eval_SS_T = HMC_encoding(f_eval, N, K, T, p)
        t_gen_f_SS = + time.time() - t0_gen_f_SS
        # print('f_eval:', f_eval.shape, f_eval_SS_T.shape)

        CTT += time.time() - CTT_0

        SAT_0 = time.time()

        # 4. HMC decoding f_eval  & calculate the gradient (over the secret share)
        t0_gen_grad_SS = time.time()

        # 4.1. send the secret shares of f_eval
        f_deg = 3
        RT = f_deg * (K + T - 1) + 1
        dec_input = np.empty((RT, d * d_number), dtype='int64')
        for j in range(1, RT + 1):
            if rank == j:
                dec_input[j - 1, :] = np.reshape(f_eval_SS_T[j - 1, :, :], d * d_number)
                # secret share q
                for j in list(range(1, rank)) + list(range(rank + 1, N + 1)):
                    data = np.reshape(f_eval_SS_T[j - 1, :, :], d * d_number)
                    # sent data to worker j
                    comm.Send(data, dest=j)
            else:
                data = np.empty(d * d_number, dtype='int64')
                comm.Recv(data, source=j)
                # coefficients for the polynomial
                dec_input[j - 1, :] = data
        t_comm_f_eval_SS += time.time() - t0_gen_grad_SS

        # 4.2. decode f_eval over the secret share
        dec_out = HMC_decoding(dec_input, N, K, T, range(RT), p)

        # 4.3. update the secret share of gradient
        f_SS_T = np.zeros((d, d_number), dtype='int64')
        for j in range(K):
            f_SS_T = np.mod(f_SS_T + np.reshape(dec_out[j, :], (d, d_number)), p)
        grad_SS_T = np.mod(f_SS_T + XTy_SS_T, p)

        t_gen_grad_SS += time.time() - t0_gen_grad_SS

        # 5. truncation gradient
        t0_trunc = time.time()
        grad_trunc_SS_T = MPI_TruncPr(grad_SS_T, r1_SS_T, r2_SS_T, trunc_k, trunc_scale, T, p)
        t_trunc += time.time() - t0_trunc

        SAT += time.time() - SAT_0

        CUT_0 = time.time()

        # 6. update the model
        w_SS_T = np.mod(w_SS_T - grad_trunc_SS_T, p)

        hist_w_SS_T[iter, :] = np.reshape(w_SS_T, d * d_number)

    t_mainloop = time.time() - t0_mainloop

    # send time_set to rank 0
    time_set = np.array(
        [t_HMC_encoding_X, t_XTX, t_XTy, t_HMC_encoding_w, t_f_eval, t_gen_f_SS, t_comm_w, t_comm_f_eval_SS,
         t_comm_X_HMC, t_preprocessing, t_mainloop, CTT, SAT, CUT])
    size_set = np.array([pre_total_size_data_x, main_total_size_data_w, main_total_size_data_f])

    comm.Send(time_set, dest=0)
    comm.Send(size_set, dest=0)

    comm.Barrier()