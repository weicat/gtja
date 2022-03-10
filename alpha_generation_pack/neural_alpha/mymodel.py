import tensorflow as _tf
import tensorflow.keras.layers as _tfl
from tensorflow.keras.layers import Layer as _Layer
from tensorflow.keras.initializers import Initializer as _Initializer
from abc import ABC as _ABC
import keras
import warnings
from tensorflow_text import sliding_window
from itertools import combinations
from tensorflow.keras.layers import ReLU

if not "2.3.0" <= _tf.__version__:
    print(f"requires tensorflow version >= 2.3.0, "
          f"current version {_tf.__version__}")
    exit(1)

class _Log(_Layer):


    def call(self, inputs, *args, **kwargs):
        
        return _tf.math.log(inputs)
            
    
class _SlideWindowGenerating(_Layer):
    def __init__(self, stride = 1, 
                 ksize = 10, 
                 **kwargs):

            
        super(_SlideWindowGenerating, self).__init__(**kwargs)
        self.stride = stride
        self.ksize = ksize
        self.out_shape = None
        self.intermediate_shape = None
    
    def build(self, input_shape):
        (features,
          output_length) = __get_dimensions__(input_shape, 
                                              self.ksize,
                                              self.stride)
        self.out_shape = [-1, output_length, features]
        self.intermediate_shape = [-1, output_length, self.ksize, features] 
    
    
    def call(self, inputs, *args, **kwargs):
        
        rolled_inputs = sliding_window(inputs, self.ksize, axis = 1)
        rolled_inputs = rolled_inputs[:, -1::-self.stride, :, :][:, ::-1, :, :]
        
        return rolled_inputs
        
    def get_config(self):
        """获取参数，保存模型需要的函数."""
        config = super().get_config().copy()
        config.update({'stride': self.stride, 'step_size':self.ksize})
        return config    
    
    
    


class _CrossFeaturesGenerating(_Layer):
    '''
    以后再写成能传function的
    '''
    def __init__(self, cross_ignore_num = [], 
                 **kwargs):
    
            
        super(_CrossFeaturesGenerating, self).__init__(**kwargs)
        self.cross_ignore_num = cross_ignore_num
        
    def build(self, input_shape):
    
        feature_num = input_shape[-1]
        feature_shape = int(feature_num * (feature_num - 1) /2)
        
        self.out_shape = [-1, input_shape[1], feature_shape]
    
    def call(self, inputs, *args, **kwargs):
        feature_shape = inputs.shape[-1]
        if len(self.cross_ignore_num)!=0:
            t = list(range(feature_shape))
            z = []
            for i in t:
                if i not in self.cross_ignore_num:
                    z.append(i)
            mycombination = combinations(z, 2)
        else:
            mycombination = combinations(range(feature_shape), 2)
        temp = _tf.gather(inputs, list(mycombination), axis = -1)
        
        return temp[:, :, :, 0] - temp[:, :, :, 1]
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({'cross_ignore_num': self.cross_ignore_num})
                       
        return config  

class _StrideLayer(_Layer, _ABC):
    
    def __init__(self, allow_neg = False, 
                 normalized_method = 'l1',
                 **kargs):
        
        super(_StrideLayer, self).__init__(**kargs)
        self.allow_neg = allow_neg
        self.normalized_method = normalized_method
    

    def build(self, input_shape):
        
        output_length, ksize, features = input_shape[1], input_shape[2], input_shape[3]
        intermediate_shape = [-1, output_length, ksize, features] 
        
        initializer = _tf.keras.initializers.Constant(
            value = 1/ksize)
        
        if self.allow_neg:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32')
                )
        else:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32'),
                constraint = _tf.keras.constraints.NonNeg()
                )
        
    
    def getTransformedWeight(self):
        
        if self.normalized_method is None:
            return self.weight
        
        elif self.normalized_method == 'l1':
            return self.weight/_tf.expand_dims(_tf.reduce_sum(self.weight, axis = 1), axis = 1)
        
        
        elif self.normalized_method == 'l2':
            return self.weight/_tf.expand_dims(_tf.sqrt(_tf.reduce_sum(self.weight ** 2, axis = 1)), axis = 1)
        
    
    
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'allow_neg': self.allow_neg, 
                       'normalized_method':self.normalized_method})
        return config    
    
        
        
        
    


class Mean(_StrideLayer):
    """计算每个序列各stride的均值.

    Notes:
        这里怎么设计weight可以考虑，
        现在这一版本的weight是设计成
        每个feature 
        在每个时间window 
        在每个stride上的值都不一样
    """
    
    
    def call(self, inputs, *args, **kwargs):
        """函数主逻辑实现部分.

        Args:
            inputs (tensor): 输入dimension为(batch_size, time_steps, features)

        Returns:
            dimension 为(batch_size, time_steps / stride, features)

        """
            
        output = _tf.reduce_sum(inputs * self.getTransformedWeight(), axis = -2)
        return output





class Std(_StrideLayer):
    """计算每个序列各stride的标准差.
    
    Notes:
          input[0] : data
          input[1] : Mean_layer_output
    """
    
    
    def build(self, input_shapes):
        super(Std, self).build(input_shapes[0]) 
        
        
        
    
    

    def call(self, inputs, *args, **kwargs):
        data, mean = inputs
        # print(data)
        mean = _tf.expand_dims(mean, axis = 2)
        # compute standard deviations for each stride
        output = _tf.sqrt(_tf.reduce_sum(
            self.getTransformedWeight() * (data - mean)**2, axis = -2))

        return output





class ZScore(_Layer):
    """计算每个序列各stride的均值除以其标准差.

    Notes:
        并非严格意义上的z-score,
        计算公式为每个feature各个stride的mean除以各自的standard deviation

    """

    def call(self, inputs, *args, **kwargs):
        
        mean, std = inputs
        return _tf.math.divide_no_nan(mean, std)




class Return(_StrideLayer):

    def build(self, input_shape):
        
        output_length, ksize, features = input_shape[1], input_shape[2], input_shape[3]
        ksize = ksize - 1
        if ksize <= 0:
            raise ValueError("only one slide can not compute return")
        intermediate_shape = [-1, output_length, ksize, features] 
        
        initializer = _tf.keras.initializers.Constant(
            value = 1/ksize)
        
        if self.allow_neg:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32')
                )
        else:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32'),
                constraint = _tf.keras.constraints.NonNeg()
                )
        
    
    
    def call(self, inputs, *args, **kwargs):
        
        choose_lst = []
        ksize = inputs.shape[2]
        for i in range(ksize - 1):
            choose_lst.append( (i, i + 1) )
            
        temp = _tf.gather(inputs, choose_lst , axis = -2)
        ret = temp[:,:,:,1, :] - temp[:, :, :, 0, :]
        
        return _tf.reduce_sum(self.getTransformedWeight() * ret, axis = -2)
        





class _OuterProductLayer(_StrideLayer):

    

    def build(self, input_shape):
        """构建该层，计算维度信息."""
        
        output_length, ksize, features = input_shape[1], input_shape[2], input_shape[3]
        output_features = int(features * (features - 1) / 2)
        intermediate_shape = [-1, output_length, ksize, output_features] 
        self.lower_mask = _LowerNoDiagonalMask()( (features, features) )       
        initializer = _tf.keras.initializers.Constant(
            value = 1/ksize)
        
        if self.allow_neg:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32')
                )
        else:
            self.weight = _tf.Variable(
                initial_value = initializer(
                    shape = intermediate_shape[1:]
                    ,dtype = 'float32'),
                constraint = _tf.keras.constraints.NonNeg()
        )        
 
        
        
        
        


class Covariance(_OuterProductLayer):
    """计算每个stride各时间序列片段的covariance.

    Notes:
        计算每个stride每两个feature之间的covariance大小，
        输出feature数量为features * (features - 1) / 2

    """

    def build(self, input_shapes):
        super(Covariance, self).build(input_shapes[0]) 
        

    def call(self, inputs, *args, **kwargs):
        
        data, mean = inputs

        mean = _tf.expand_dims(mean, axis = 2)

        demeaned_data = data - mean
        
        covariance_matrix = _tf.einsum("ijkl,ijkm->ijklm",
                                       demeaned_data,
                                       demeaned_data)

        # get the lower part of the covariance matrix
        # without the diagonal elements
        covariances_matrix = _tf.boolean_mask(covariance_matrix,
                                       self.lower_mask,
                                       axis = 3)
        
        
        covariances = _tf.reduce_sum(self.getTransformedWeight() * covariances_matrix, axis = -2)
        return covariances


class Correlation(_Layer):
    
    def call(self, inputs, *args, **kargs):
        
        std, cov = inputs
        features = std.shape[-1]
        choosed = sorted(list(combinations(range(features), 2)), 
               key = lambda x: x[-1])
        
        temp = _tf.gather(std, choosed , axis = -1)
        std_prod = temp[:, :, :, 0] * temp[:, :, :, 1]
        
        output = _tf.math.divide_no_nan(cov, std_prod)
        
        return output
    
    


def getFeatureExpansionLayer(inputs):
    
    # inputs = keras.Input(shape= input_shape)
    layer1 = Mean()
    layer2 = Std()
    layer3 = ZScore()
    layer4 = Covariance()
    layer5 = Correlation()
    layer6 = Return()
    
    
    m = layer1(inputs)
    m = ReLU()(m)
    s = layer2([inputs, m])
    s = ReLU()(s)
    r = layer6(inputs)
    z = layer3([m, s])
    cov = layer4([inputs, m])
    cor = layer5([s, cov])
    concat = _tfl.Concatenate(axis=-1)
    outputs = concat([m, r, s, z, cov, cor])
    # outputs = mdel = 
    # outputs = concat([m, s, cov, cor])

    return outputs



def getProcessedFeatures(inputs, 
                         recurrent_unit,
                         hidden_units,
                         ):
    
     bn = _tfl.BatchNormalization()

     if recurrent_unit == "GRU":
         recurrent = _tfl.GRU(units=hidden_units)
     elif recurrent_unit == "LSTM":
         recurrent = _tfl.LSTM(units=hidden_units)
     else:
         raise ValueError("Unknown recurrent_unit")
     bn2 = _tfl.BatchNormalization()

     x = bn(inputs)
     x = recurrent(x)
     outputs = bn2(x)
     
     return outputs


def getConnectedLayers(
        inputs,
        classification,
        categories,
        l2,
        dropout,
        observation_layer_units
        ):
    
    
    dropout_layer = _tfl.Dropout(dropout)
    regularizer = _tf.keras.regularizers.l2(l2)
    
    intermediate_layer = _tfl.Dense(observation_layer_units,
                                      kernel_initializer="truncated_normal")
    
    if classification:
        if categories < 1:
            raise ValueError("categories should be at least 1")
        elif categories == 1:
            output_layer = _tfl.Dense(1, activation="sigmoid",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=regularizer)
        else:
            output_layer = _tfl.Dense(categories, activation="softmax",
                                      kernel_initializer="truncated_normal",
                                      kernel_regularizer=regularizer)
    else:
        output_layer = _tfl.Dense(1, activation="linear",
                                  kernel_initializer="truncated_normal",
                                  kernel_regularizer=regularizer)


    dropout = dropout_layer(inputs)
    dropout = intermediate_layer(dropout)
    outputs = output_layer(dropout)
    
    return outputs


def getModelV3(input_shape,
               cross_ignore_num = [],
               recurrent_unit = 'LSTM',
               hidden_units = 40,
               classification = False,
               categories = 0,
               l2 = 0.0001,
               dropout = 0,
               observation_layer_units = 4,
               ):
    inputs = keras.Input(shape= input_shape)
    x = _Log()(inputs)
    x = _CrossFeaturesGenerating(cross_ignore_num = cross_ignore_num)(x)    
    
    long_pred = _SlideWindowGenerating(stride = 10, ksize = 10)(x)
    short_pred = _SlideWindowGenerating(stride = 5, ksize = 5)(x)
    
    short_pred  = getFeatureExpansionLayer(short_pred)
    long_pred   = getFeatureExpansionLayer(long_pred)
    short_pred  = getProcessedFeatures(short_pred, 
                                       recurrent_unit = recurrent_unit,
                                       hidden_units = hidden_units)
    long_pred   = getProcessedFeatures(long_pred, 
                                       recurrent_unit = recurrent_unit,
                                       hidden_units = hidden_units)
    
    concat      = _tfl.Concatenate(axis=-1)([long_pred, short_pred])
    outputs = getConnectedLayers(concat,
                       classification = classification,
                       categories = categories,
                       l2 = l2,
                       dropout = dropout,
                       observation_layer_units = observation_layer_units)
    model = keras.Model(inputs = inputs, outputs = outputs)
    
    return model




class _LowerNoDiagonalMask(_Initializer):
    """获取不含对角元素的矩阵下三角mask.

    Notes:
        Provide a mask giving the lower triangular of a matrix
        without diagonal elements.

    """

    def __init__(self):
        super(_LowerNoDiagonalMask, self).__init__()

    def __call__(self, shape, **kwargs):
        """计算逻辑."""
        ones = _tf.ones(shape)
        mask_lower = _tf.linalg.band_part(ones, -1, 0)
        mask_diag = _tf.linalg.band_part(ones, 0, 0)
        # lower triangle removing the diagonal elements
        mask = _tf.cast(mask_lower - mask_diag, dtype=_tf.bool)
        return mask


def __get_dimensions__(input_shape, ksize, stride):
    if type(stride) is not int :
        raise ValueError("Illegal Argument: stride should be an integer "
                         "greater than 1")
    time_steps = input_shape[1]
    features = input_shape[2]
    output_length = (time_steps - ksize) // stride + 1

    if (time_steps - ksize)  % stride != 0:
        warnings.warn("time_steps - ksize 应该是 stride的整数倍")

    return features, output_length
