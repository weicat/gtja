"""训练中的辅助准确率信息.

该module包含涨跌精度信息的计算类。
使用到该类的模型在保存后重新读取时需要添加``custom_objects``参数，
或者使用``alphanet.load_model()``函数。

"""
import tensorflow as _tf
import tensorflow_probability as tfp

__all__ = ["UpDownAccuracy"]


class UpDownAccuracy(_tf.keras.metrics.Metric):
    """通过对return的预测来计算涨跌准确率."""

    def __init__(self, name='up_down_accuracy', **kwargs):
        """涨跌准确率."""
        super(UpDownAccuracy, self).__init__(name=name, **kwargs)
        self.up_down_correct_count = self.add_weight(name='ud_count',
                                                     initializer='zeros',
                                                     shape=(),
                                                     dtype=_tf.float32)
        self.length = self.add_weight(name='len',
                                      initializer='zeros',
                                      shape=(),
                                      dtype=_tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """加入新的预测更新精度值."""
        y_true = _tf.cast(y_true > 0.0, _tf.float32)
        y_pred = _tf.cast(y_pred > 0.0, _tf.float32)
        length = _tf.cast(len(y_true), _tf.float32)
        correct_count = length - _tf.reduce_sum(_tf.abs(y_true - y_pred))

        self.length.assign_add(length)
        self.up_down_correct_count.assign_add(correct_count)

    def result(self):
        """获取结果."""
        if self.length == 0.0:
            return 0.0
        return self.up_down_correct_count / self.length

    def reset_state(self):
        """在train、val的epoch末尾重置精度."""
        self.up_down_correct_count.assign(0.0)
        self.length.assign(0.0)


class IC(_tf.keras.metrics.Metric):

    def __init__(self, name='batchIc', **kwargs):
        """涨跌准确率."""
        super(IC, self).__init__(name=name, **kwargs)
        self.ic_sum = self.add_weight(name='ic_count',
                                                      initializer='zeros')
        self.length = self.add_weight(name='len',
                                      initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """加入新的预测更新精度值."""
        y_true = _tf.cast(y_true, _tf.float32)
        y_pred = _tf.cast(y_pred, _tf.float32)
        length = _tf.cast(len(y_true), _tf.float32)

        demean_true = y_true - _tf.reduce_mean(y_true)
        demean_pred = y_pred - _tf.reduce_mean(y_pred)
        
        sse = _tf.sqrt(_tf.reduce_sum(demean_true * demean_true))
        ssr = _tf.sqrt(_tf.reduce_sum(demean_pred * demean_pred))
        
        
        res = _tf.reduce_sum(demean_true * demean_pred) / (sse * ssr)
            
        self.ic_sum.assign_add(res)
        self.length.assign_add(1)

    def result(self):
        """获取结果."""
        if self.length == 0.0:
            return 0.0
        return self.ic_sum / self.length

    def reset_state(self):
        """在train、val的epoch末尾重置精度."""
        self.ic_sum.assign(0.0)
        self.length.assign(0.0)
        

class ClassificationIC(IC):
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        
        y_true = _tf.cast(y_true, _tf.float32)
        y_true = _tf.reduce_sum(y_true* _tf.constant([1., -1.]), axis = -1)
        sample_weight = _tf.cast(sample_weight, _tf.float32)        
        y_true = y_true * _tf.reshape(sample_weight, shape = (-1, ))
        y_pred = _tf.cast(y_pred, _tf.float32)[:, 0]

        demean_true = y_true - _tf.reduce_mean(y_true)
        demean_pred = y_pred - _tf.reduce_mean(y_pred)
        
        sse = _tf.sqrt(_tf.reduce_sum(demean_true * demean_true))
        ssr = _tf.sqrt(_tf.reduce_sum(demean_pred * demean_pred))
        
        
        res = _tf.reduce_sum(demean_true * demean_pred) / (sse * ssr)
            
        self.ic_sum.assign_add(res)
        self.length.assign_add(1)
        
        
    
    
    

        
        
class rankIC(_tf.keras.metrics.Metric):

    def __init__(self, name='batchRankIc', **kwargs):
        """涨跌准确率."""
        super(rankIC, self).__init__(name=name, **kwargs)
        self.rankic_sum = self.add_weight(name='rankic_count',
                                                      initializer='zeros')
        self.length = self.add_weight(name='len',
                                      initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """加入新的预测更新精度值."""
        y_true = _tf.cast(y_true, _tf.float32)
        y_pred = _tf.cast(y_pred, _tf.float32)
        # length = _tf.cast(len(y_true), _tf.float32)

        y_true = _tf.argsort(_tf.argsort(y_true,axis = 0),axis = 0)
        y_pred = _tf.argsort(_tf.argsort(y_pred,axis = 0),axis = 0)

        y_true = _tf.cast(y_true, _tf.float32)
        y_pred = _tf.cast(y_pred, _tf.float32)

        
        demean_true = y_true - _tf.reduce_mean(y_true)
        demean_pred = y_pred - _tf.reduce_mean(y_pred)
        
        sse = _tf.sqrt(_tf.reduce_sum(demean_true * demean_true))
        ssr = _tf.sqrt(_tf.reduce_sum(demean_pred * demean_pred))
        
        
        res = _tf.reduce_sum(demean_true * demean_pred) / (sse * ssr)
            
        self.rankic_sum.assign_add(res)
        self.length.assign_add(1.0)

    def result(self):
        """获取结果."""
        if self.length == 0.0:
            return 0.0
        return self.rankic_sum / self.length

    def reset_state(self):
        """在train、val的epoch末尾重置精度."""
        self.rankic_sum.assign(0.0)
        self.length.assign(0.0)
        

class ClassificationRankIC(rankIC):
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        
        y_true = _tf.cast(y_true, _tf.float32)
        y_true = _tf.reduce_sum(y_true* _tf.constant([1., -1.]), axis = -1)
        sample_weight = _tf.cast(sample_weight, _tf.float32)        
        y_true = y_true * _tf.reshape(sample_weight, shape = (-1, ))
        y_pred = _tf.cast(y_pred, _tf.float32)[:, 0]

        y_true = _tf.argsort(_tf.argsort(y_true,axis = 0),axis = 0)
        y_pred = _tf.argsort(_tf.argsort(y_pred,axis = 0),axis = 0)        
        
        
        y_true = _tf.cast(y_true, _tf.float32)
        y_pred = _tf.cast(y_pred, _tf.float32)
        
        demean_true = y_true - _tf.reduce_mean(y_true)
        demean_pred = y_pred - _tf.reduce_mean(y_pred)
        
        sse = _tf.sqrt(_tf.reduce_sum(demean_true * demean_true))
        ssr = _tf.sqrt(_tf.reduce_sum(demean_pred * demean_pred))
        res = _tf.reduce_sum(demean_true * demean_pred) / (sse * ssr)
            
        self.rankic_sum.assign_add(res)
        self.length.assign_add(1.0)
            
    

