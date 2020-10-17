from nn.layers import Conv2D
import numpy as np
from utils.check_grads_cnn import check_grads_layer

batch = 10
conv_params={
    'kernel_h': 2,
    'kernel_w': 2,
    'pad': 0,
    'stride': 1,
    'in_channel': 1,
    'out_channel': 1
}
in_height = 3
in_width = 3
out_height = 1+(in_height+2*conv_params['pad']-conv_params['kernel_h'])//conv_params['stride']
out_width = 1+(in_width+2*conv_params['pad']-conv_params['kernel_w'])//conv_params['stride']

input = np.array(
    [[1, 2, 3],
     [2, 3, 2],
     [1, 2, 2]], dtype=np.float
).reshape(1, 1, in_height, in_width)
out_grad = np.array(
    [[1, -1],
     [1, -1]], dtype=np.float
).reshape(1, 1, out_height, out_width)

conv = Conv2D(conv_params)

conv.weights = np.array(
    [[0, 1],
     [0, 1]], dtype=np.float
).reshape(1, 1, 2, 2)
check_grads_layer(conv, input, out_grad)