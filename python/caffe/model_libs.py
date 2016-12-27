import os
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
  return os.path.exists(path)

def make_if_not_exist(path):
  if not os.path.exists(path):
    os.makedirs(path)

# helper function for common structures
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1, param=None, is_train=False, has_relu=True, bias_term=True, base_param_name=None):
  kwargs = {'num_output': nout, 'kernel_size': ks} 
  if (stride != 1): 
    kwargs['stride'] = stride
  if (group != 1): 
    kwargs['group'] = group
  if (pad != 0): 
    kwargs['pad'] = pad 
  if (bias_term==False):
    kwargs['bias_term'] = bias_term
  if (is_train):
    #kwargs['weight_filler'] = dict(type='gaussian', std=0.01)
    kwargs['weight_filler'] = dict(type='xavier')
    #kwargs['bias_filler'] = dict(type='constant', value=1)
  else:
    param = None
    base_param_name=None

  if (param!= None):
    #kwargs['param'] = [dict(name=param_name)]
    kwargs['param'] = param
  elif (base_param_name):
    if (bias_term):
      kwargs['param'] = [dict(name=base_param_name+'_conv_w'), dict(name=base_param_name+'_conv_b')]
    else:
      kwargs['param'] = [dict(name=base_param_name+'_conv_w')]

  conv = L.Convolution(bottom, **kwargs)
  if (has_relu):
    return conv, L.ReLU(conv, in_place=True)
  else:
    return conv


def conv_bn_scale(bottom, ks, nout, stride=1, pad=0, group=1, param=None, is_train=False, bn_param=None, scale_param=None, has_relu=False, base_param_name=None, bias_term=True):
  conv = conv_relu(bottom, ks, nout, stride=stride, pad=pad, group=group, param=param, is_train=is_train, has_relu=False, base_param_name=base_param_name, bias_term=bias_term)
  if bn_param:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True, param=bn_param)
  elif base_param_name and is_train==True:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True, param=[dict(name=base_param_name+'_bn1',lr_mult=0,decay_mult=0), dict(name=base_param_name+'_bn2',lr_mult=0,decay_mult=0), dict(name=base_param_name+'_bn3',lr_mult=0,decay_mult=0)])
  else:
    bn = L.BatchNorm(conv, use_global_stats=True, in_place=True)

  if scale_param:
    scale = L.Scale(bn, bias_term=True, in_place=True, param=scale_param)
  elif base_param_name and is_train==True:
    scale = L.Scale(bn, bias_term=True, in_place=True, param=[dict(name=base_param_name+'_scale1'), dict(name=base_param_name+'_scale2')])
  else:
    scale = L.Scale(bn, bias_term=True, in_place=True)

  if (has_relu):
    return conv, bn, scale, L.ReLU(scale, in_place=True)
  else:
    return conv, bn, scale

# yet another helper function
def max_pool(bottom, ks, stride=1):
  if (stride != 1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
  else:
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks)

def ave_pool(bottom, ks, stride=1):
  if (stride != 1):
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks, stride=stride)
  else:
    return L.Pooling(bottom, pool=P.Pooling.AVE, kernel_size=ks)

def fc_relu(bottom, nout, param=None, is_train=False, has_relu=True, base_param_name=None):
  kwargs = {'num_output': nout}
  if (is_train):
    kwargs['weight_filler'] = dict(type='gaussian', std=0.01)
    #kwargs['weight_filler'] = dict(type='xavier')
  else:
    param = None
  if (param != None):
    #kwargs['param'] = [dict(name=param_name)]
    kwargs['param'] = param
  elif (base_param_name): 
    kwargs['param'] = [dict(name=base_param_name+'_fc_w'), dict(name=base_param_name+'_fc_b')]
  
  fc = L.InnerProduct(bottom, **kwargs)
  if (has_relu):
    return fc, L.ReLU(fc, in_place=True)
  else:
    return fc
