from nn.layers import Conv2D
import numpy as np
from utils.check_grads_cnn import check_grads_layer
from nn.layers import Pool2D
from utils.tools import rel_error

from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPooling2D

import warnings
warnings.filterwarnings('ignore')


def conv_layer():

    batch = 10
    conv_params={
        'kernel_h': 5,
        'kernel_w': 5,
        'pad': 1,
        'stride': 2,
        'in_channel': 3,
        'out_channel': 8
    }
    in_height = 12
    in_width = 12
    out_height = 1+(in_height+conv_params['pad']-conv_params['kernel_h'])//conv_params['stride']
    out_width = 1+(in_width+conv_params['pad']-conv_params['kernel_w'])//conv_params['stride']

    input = np.random.uniform(size=(batch, conv_params['in_channel'], in_height, in_width))
    out_grad = np.random.uniform(size=(batch, conv_params['out_channel'], out_height, out_width))

    conv = Conv2D(conv_params)
    check_grads_layer(conv, input, out_grad)


def pool_layer():
    import numpy as np
    import warnings
    warnings.filterwarnings('ignore')

    from nn.layers import Pool2D
    from utils.tools import rel_error

    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import MaxPooling2D

    input = np.random.uniform(size=(10, 3, 30, 30))
    keras_input = input.transpose(0, 2, 3, 1)
    params = {
        'pool_type': 'max',
        'pool_height': 4,
        'pool_width': 4,
        'pad': 2,
        'stride': 2,
    }
    pool = Pool2D(params)
    out = pool.forward(input)

    keras_pool = Sequential([
        MaxPooling2D(pool_size=(params['pool_height'], params['pool_width']),
                     strides=params['stride'],
                     padding='same',
                     data_format='channels_last',
                     input_shape=keras_input.shape[1:])
    ])
    keras_out = keras_pool.predict(keras_input, batch_size=input.shape[0])

    print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out.transpose(0, 3, 1, 2)))


def rnn_forward():
    import numpy as np
    from tensorflow.keras import layers, Model
    from nn.layers import LSTMCell
    from utils.tools import rel_error
    import numpy
    import tensorflow.keras.backend as K

    N, D, H = 3, 10, 4
    x = np.random.uniform(size=(N, D))
    prev_h = np.random.uniform(size=(N, H))
    prev_c = np.random.uniform(size=(N, H))

    lstm_cell = LSTMCell(in_features=D, units=H)
    out, cell = lstm_cell.forward([x, prev_c, prev_h])

    # compare with the keras implementation
    keras_x = layers.Input(batch_shape=(N, None, D), name='x')
    hidden_states = K.variable(value=prev_h)
    cell_states = K.variable(value=prev_c)
    keras_rnn = layers.LSTM(units=H, use_bias=False, recurrent_activation='sigmoid', stateful=True)(keras_x)
    keras_model = Model(inputs=keras_x,
                        outputs=keras_rnn)
    keras_model.layers[1].states[0] = hidden_states
    keras_model.layers[1].states[1] = cell_states
    keras_model.layers[1].set_weights([lstm_cell.kernel,
                                       lstm_cell.recurrent_kernel])
    keras_out = keras_model.predict_on_batch([x[:, None, :]])

    print(keras_model.layers[1].get_weights)
    # print([a-lstm_cell.kernel,b-lstm_cell.recurrent_kernel])
    print('Relative error (<1e-5 will be fine): {}'.format(rel_error(keras_out, out)))


def rnn_backward():
    import numpy as np
    from nn.layers import BiRNN
    from utils.check_grads_rnn import check_grads_layer

    N, T, D, H = 2, 3, 4, 5
    x = np.random.uniform(size=(N, T, D))
    # padding x with a nan mask for testing if the input of backward rnn has been reversed correctly by using _reverse_temporal_data(self, x, mask).
    # mask used [[1, 1, 0],[1, 0, 0]]
    x[0, -1:, :] = np.nan
    x[1, -2:, :] = np.nan
    h0 = np.random.uniform(size=(H,))
    hr = np.random.uniform(size=(H,))

    brnn = BiRNN(in_features=D, units=H, h0=h0, hr=hr)
    out_grad = np.random.uniform(size=(N, T, H * 2))
    check_grads_layer(brnn, x, out_grad)


def train_mnist():
    from nn.optimizers import RMSprop, Adam
    import matplotlib.pyplot as plt
    import numpy as np

    from models.MNISTNet import MNISTNet
    from nn.loss import SoftmaxCrossEntropy, L2
    from nn.optimizers import Adam
    from data.datasets import MNIST
    np.random.seed(5242)

    mnist = MNIST()

    model = MNISTNet()
    loss = SoftmaxCrossEntropy(num_class=10)

    # define your learning rate sheduler
    def func(lr, iteration):
        if iteration % 1000 == 0:
            return lr * 0.5
        else:
            return lr

    adam = Adam(lr=0.001, decay=0, sheduler_func=None, bias_correction=True)
    l2 = L2(w=0.001)  # L2 regularization with lambda=0.001
    model.compile(optimizer=adam, loss=loss, regularization=l2)

    import time
    start = time.time()
    train_results, val_results, test_results = model.train(
        mnist,
        train_batch=50, val_batch=1000, test_batch=1000,
        epochs=2,
        val_intervals=-1, test_intervals=900, print_intervals=100)
    print('cost:', time.time() - start)


if __name__ == "__main__":
    # pool_layer()
    # rnn_forward()
    # rnn_backward()
    train_mnist()