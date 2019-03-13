import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time

# load all the files into a dictionary for use
# pickle numpy arrays for faster loading, if they aren't there already
def load_files():
    print('loading data...')

    try:
        train_x = np.load('data/TrainDigitX.npy')
        train_y = np.load('data/TrainDigitY.npy')
        test_x = np.load('data/TestDigitX.npy')
        test_y = np.load('data/TestDigitY.npy')
        test_x2 = np.load('data/TestDigitX2.npy')
    except:
        train_x = np.loadtxt('data/TrainDigitX.csv.gz', ndmin=2, delimiter=',')
        bias = np.ones((train_x.shape[0], 1))
        train_x = np.append(train_x, bias, axis=1)

        train_y = np.loadtxt('data/TrainDigitY.csv', delimiter=',')
        train_y = y_to_one_hot(train_y.astype(int))

        test_x = np.loadtxt('data/TestDigitX.csv.gz', ndmin=2, delimiter=',')
        bias = np.ones((test_x.shape[0], 1))
        test_x = np.append(test_x, bias, axis=1)

        test_y = np.loadtxt('data/TestDigitY.csv', delimiter=',')
        test_y = y_to_one_hot(test_y.astype(int))

        test_x2 = np.loadtxt('data/TestDigitX2.csv.gz', ndmin=2, delimiter=',')
        bias = np.ones((test_x2.shape[0], 1))
        test_x2 = np.append(test_x2, bias, axis=1)

        np.save('data/TrainDigitX.npy', train_x)
        np.save('data/TrainDigitY.npy', train_y)
        np.save('data/TestDigitX.npy', test_x)
        np.save('data/TestDigitY.npy', test_y)
        np.save('data/TestDigitX2.npy', test_x2)

    print('data loaded.')

    return {'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y,
            'test_x2': test_x2,
           }


# some helper functions for math, all vectorized
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax_loss(x):
    return - np.log(np.exp(x)/np.sum(np.exp(x)))


# turn vector of y's into matrix of one-hot representation
def y_to_one_hot(y):
    one_hot = np.zeros((y.size, 10))
    one_hot[np.arange(y.size), y] = 1.
    return one_hot


# get an integer prediction from a probability vector
def prob_to_pred(y):
    return np.argmax(y, axis=1)


# initialize a neural net, represented as a tuple (weights, biases)
# weights is a list of weight matrices, biases is list of bias vectors
# i made the weights matrices so that numpy would vectorize it all. zooom.
# weights are initialized to the normal distribution around 0
# biases are initialized to zero
# a classmate told me that this speeds up convergence according to some paper
def init_nn():
    print('initialising neural net...')
    
    l1_weights = np.random.randn(HIDDEN_SIZE, INPUT_SIZE)

    hidden_weights = [np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(N_HIDDEN - 1)]
    hidden_biases = [np.random.randn(HIDDEN_SIZE) for _ in range(N_HIDDEN)]

    out_weights = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE)
    out_biases = np.zeros(OUTPUT_SIZE)

    weights = [l1_weights] + hidden_weights + [out_weights]
    biases = hidden_biases + [out_biases]

    print('neural net initialised.')

    return weights, biases


# do a forward pass of a datapoint through the nn
# nn is a tuple, a list of weight matrices, then a list of bias vectors.
# x is one datapoint. i'm doing this online, no batching at all.
# return activations list and prediction vector before softmax
def forward_pass(nn, x):
    weights, biases = nn
    a = x  # activations coming from previous layer
    a_s = [a]  # store activations as we go for backprop.

    for i in range(len(weights)):
        # as described in the slides, but vectorized across layers :o
        w, b = weights[i], biases[i]
        a = sigmoid(np.dot(w, a) + b)
        a_s.append(a)

    return a_s, a


# get sig'(x) from sig(x)
def d_sig(s_x):
    return s_x * (1 - s_x)


# do backpropagation given an nn, a list of activation arrays, 
# the final output of the nn, and an expected one-hot representation
def backpropagate(nn, a_s, output, y):
    weights, biases = nn
    
    # z_t headed into output layer
    a = a_s[-1]

    # delta_l for the final layer, from the slides
    d_l = (a - y) * np.dot(output, y) * d_sig(a)

    # now iterate through the layers one by one
    for i in range(len(weights) - 1, -1, -1):
        # reset activations to current layer's
        a = a_s[i]
        # update weights and biases
        weights[i] -= LEARNING_RATE * np.outer(d_l, a)
        biases[i] -= LEARNING_RATE * d_l
        # finally, update delta for the next layer
        d_l = np.dot(np.transpose(weights[i]), d_l) * d_sig(a)


# train a given nn on some data using SGD
def train_epoch(nn, train_x, train_y, n_ep=0):
    print('STARTING EPOCH {}'.format(n_ep + 1))
    N = train_x.shape[0]
    # shuffle examples for SGD
    shfl = np.random.permutation(N)
    for i in tqdm(range(N)):  # nice progress bar
        x, y = train_x[shfl[i]], train_y[shfl[i]]
        a_s, out = forward_pass(nn, x)
        backpropagate(nn, a_s, softmax_loss(out), y)
    print('FINISHED EPOCH')


# turns an nn and a set of x's into integer predictions
def make_predictions(nn, x_s):
    return prob_to_pred(np.array([forward_pass(nn, x)[1] for x in x_s]))


# calculates holdout error for a given nn and data
# returns percent of incorrect predictions
def calc_holdout_error(nn, hold_x, hold_y):
    hold_y, pred_y = prob_to_pred(hold_y), make_predictions(nn, hold_x)
    return 100 * (1 - np.sum(np.equal(hold_y, pred_y)) / hold_y.shape[0])


# gets nth holdout set and rest of data
# (train_x, train_y, hold_x, hold_y)
# n_th is modulo N_FOLD so u can just wrap around
def gen_holdouts(train_x, train_y, n_th):
    c_size = int(train_x.shape[0] / N_FOLD)  # chunk size
    c_bnd = [c_size * i for i in range(1, N_FOLD)]  # chunk bounds
    n_th = n_th % N_FOLD
    
    # do some disgusting array manipulation to break it into the right pieces
    if n_th == 0:
        return train_x[c_bnd[0]:], train_y[c_bnd[0]:],\
                train_x[:c_bnd[0]], train_y[:c_bnd[0]]
    if n_th == N_FOLD - 1:
        return train_x[:c_bnd[n_th - 1]], train_y[:c_bnd[n_th - 1]],\
            train_x[c_bnd[n_th - 1]:], train_y[c_bnd[n_th - 1]:]
    return np.concatenate((train_x[:c_bnd[n_th - 1]], train_x[c_bnd[n_th]:])),\
            np.concatenate((train_y[:c_bnd[n_th - 1]], train_y[c_bnd[n_th]:])),\
            train_x[c_bnd[n_th - 1]:c_bnd[n_th]], train_y[c_bnd[n_th - 1]:c_bnd[n_th]]


def main():
    # do some setup
    data = load_files()
    nn = init_nn()
    
    # i just run the nn a ton of epochs 
    # and use whatever got the best test error
    min_test_error = float('Inf') 
    min_nn = None
    min_epoch = 0
    
    for n_ep in range(MAX_N_EPOCHS):
        # get training and holdout sets
        t_x, t_y, h_x, h_y = \
            gen_holdouts(data['train_x'], data['train_y'], n_ep)

        # train one epoch
        train_epoch(nn, t_x, t_y, n_ep=n_ep)
       
        # calculate the holdout and test errors 
        holdout_error = calc_holdout_error(nn, h_x, h_y)
        test_error = calc_holdout_error(nn, data['test_x'], data['test_y'])
        
        print('HOLDOUT ERROR RATE:', holdout_error)
        print('TEST ERROR RATE:', test_error)
        
        # looks like we found a new best nn!
        # save it and keep going
        if test_error < min_test_error:
            min_test_error = test_error
            min_nn = deepcopy(nn)
            min_epoch = n_ep
    
    # uncomment for testing multiple parameters
    #return min_epoch, min_test_error
    
    # this section is to predict + write TestDigitY2.csv
    print('TRAINED. MIN ERROR RATE:', min_test_error)
    print('WRITING TestDigitX2.csv PREDICTIONS TO FILE')

    with open('TestDigitY.csv', 'w') as f:
        pred_y2 = make_predictions(min_nn, data['test_x'])
        for pred in pred_y2:
            f.write('{}\n'.format(pred))


# parameters I don't change
INPUT_SIZE = 785
OUTPUT_SIZE = 10
N_FOLD = 10  # for cross validation
MAX_N_EPOCHS = 50  # everything pretty much converges by 50

if __name__ == '__main__':
    N_HIDDEN = 1
    #LEARNING_RATE = 0.05
    HIDDEN_SIZE = 64
    #main() 
    
    # stuff to test different N_HIDDEN and HIDDEN_SIZE
    '''
    N_HIDDEN_OPTS = [i for i in range(1, 5)]
    HIDDEN_SIZE_OPTS = [32, 64, 128, 256]
    infolist = []
    for i in range(len(N_HIDDEN_OPTS)):
        for j in range(len(HIDDEN_SIZE_OPTS)):
            N_HIDDEN = N_HIDDEN_OPTS[i]
            HIDDEN_SIZE = HIDDEN_SIZE_OPTS[j]
            #print('NEW ROUND PARAMETERS:')
            #print('    N_HIDDEN    =', N_HIDDEN)
            #print('    HIDDEN_SIZE =', HIDDEN_SIZE)
            start = time.time()
            epochs, test_error = main()
            end = time.time()
            #print('    SECONDS RUNTIME =', end - start)
            infolist.append({'N_HIDDEN': N_HIDDEN,
                'HIDDEN_SIZE': HIDDEN_SIZE,
                'LEARNING_RATE': LEARNING_RATE,
                'EPOCHS_TO_CONVERGE': epochs,
                'TEST_ERROR': test_error,
                'SECONDS_RUNTIME': end - start,
                })
            print(infolist[-1])
    print(infolist)
    '''

    # stuff to test different LEARNING_RATE
    
    LEARNING_RATE_OPTS = [0.05, 0.1, 0.5, 1., 2.]
    infolist = []
    for i in range(len(LEARNING_RATE_OPTS)):
        LEARNING_RATE = LEARNING_RATE_OPTS[i]
        start = time.time()
        epochs, test_error = main()
        end = time.time()
        infolist.append({'N_HIDDEN': N_HIDDEN,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'EPOCHS_TO_CONVERGE': epochs,
            'TEST_ERROR': test_error,
            'SECONDS_RUNTIME': end - start,
            })
    print(infolist)
