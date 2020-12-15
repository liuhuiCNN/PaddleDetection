import os
import sys
import pickle
import paddle
import paddle.fluid as fluid


class Load(paddle.nn.Layer):
    def __init__(self):
        super(Load, self).__init__()
    
    def forward(self, filename):
        weight = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self._helper.append_op(
            type='load',
            inputs={},
            outputs={'Out': [weight]},
            attrs={'file_path': filename})
        return weight



if __name__ == '__main__':
    p = 'output/st_weights/faster_rcnn_dcn_r50_vd_fpn_3x_server_side/res3a_branch2b_conv_offset.w_0'
    load = Load()
    weight = load(p)
    np_weight = weight.numpy()
    print(np_weight.shape)
    print(np_weight)
