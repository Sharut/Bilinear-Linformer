#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
'''Capsule in PyTorch
TBD
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from .bilinear_sparse_routing import BilinearSparseRouting, BilinearRouting, DynamicBilinearRouting
from .linformer import LinformerProjectionKernel, BilinearProjectionWithEmbeddings

#### Simple Backbone ####
class simple_backbone(nn.Module):
    def __init__(self, cl_input_channels,cl_num_filters,cl_filter_size, 
                                  cl_stride,cl_padding):
        super(simple_backbone, self).__init__()
        self.pre_caps = nn.Sequential(
                    nn.Conv2d(in_channels=cl_input_channels,
                                    out_channels=cl_num_filters,
                                    kernel_size=cl_filter_size, 
                                    stride=cl_stride,
                                    padding=cl_padding),
                    nn.ReLU(),
        )
    def forward(self, x):
        out = self.pre_caps(x) # x is an image
        return out 


#### ResNet Backbone ####
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class resnet_backbone_cifar(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters,
                 cl_stride):   
        super(resnet_backbone_cifar, self).__init__()
        self.in_planes = 64
        def _make_layer(block, planes, num_blocks, stride):
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        
        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1), # num_blocks=2 or 3
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=4, stride=cl_stride), # num_blocks=2 or 4
        )
    def forward(self, x):
        out = self.pre_caps(x) # x is an image
        return out 


#Imagenet backbone
class resnet_backbone_imagenet(nn.Module):
    def __init__(self, cl_input_channels, cl_num_filters,
                 cl_stride):   
        super(resnet_backbone_imagenet, self).__init__()
        self.in_planes = 64
        def _make_layer(block, planes, num_blocks, stride):
            # strides = [stride] + [1]*(num_blocks-1)
            strides = [stride]*3 + [1]*(num_blocks-1)

            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)
        
        self.pre_caps = nn.Sequential(
            nn.Conv2d(in_channels=cl_input_channels, 
                      out_channels=64, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            _make_layer(block=BasicBlock, planes=64, num_blocks=3, stride=1), # num_blocks=2 or 3
            # _make_layer(block=BasicBlock, planes=128, num_blocks=4, stride=cl_stride), # num_blocks=2 or 4
            _make_layer(block=BasicBlock, planes=cl_num_filters, num_blocks=4, stride=cl_stride), # num_blocks=2 or 4
            # _make_layer(block=BasicBlock, planes=512, num_blocks=2, stride=cl_stride), # num_blocks=2 or 4
        )
    def forward(self, x):
        out = self.pre_caps(x) # x is an image
        # print("Resnet backbone shape: ", out.shape)
        return out 


###
# Explained einsum

'''

https://stackoverflow.com/questions/26089893/understanding-numpys-einsum

torch.einsum('i,ij->i', A, B)

    1. A has one axis; we've labelled it i. And B has two axes; 
    we've labelled axis 0 as i and axis 1 as j.
    2. By repeating the label i in both input arrays, we are telling 
    einsum that these two axes should be multiplied together. 
    In other words, we're multiplying array A with each column of array B, 
    just like A[:, np.newaxis] * B does.
    3. Notice that j does not appear as a label in our desired output; 
    we've just used i (we want to end up with a 1D array). 
    By omitting the label, we're telling einsum to sum along this axis. 
    In other words, we're summing the rows of the products, just like .sum(axis=1) does.


'''








#### Capsule Layer ####
class CapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    
    '''
    Same as CapsuleConv
    except that kernal size=1 everywhere. 
    '''

    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp):
        super(CapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose
        
        # Matrix form of Hilton
        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules/(self.sqrt_d*in_n_capsules)) 
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))
        
        # Vector form of Hilton  
        else:
            self.weight_init_const = np.sqrt(out_n_capsules/(in_d_capsules*in_n_capsules)) 
            self.w = nn.Parameter(self.weight_init_const* \
                                          torch.randn(in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules))


        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            weight_init_const={}, dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.weight_init_const, self.dropout_rate
        )        
    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0,2,1)

        if self.matrix_pose:
            w = self.w # nxdm
            _input = input.view(input.shape[0], input.shape[1], self.sqrt_d, self.sqrt_d) # bnax
        else:
            w = self.w
            

        if next_capsule_value is None:
            # next_capsule_vale=None at 1st Iteration
            # query key == r_{i,j} (routing probabilities)
            query_key = torch.zeros(self.in_n_capsules, self.out_n_capsules).type_as(input)
            query_key = F.softmax(query_key, dim=1)

            if self.matrix_pose:
                # Einsum: computing multilinear expressions (i.e. sums of products) using the Einstein summation convention.
                next_capsule_value = torch.einsum('nm, bnax, nxdm->bmad', query_key, _input, w)
            else:
                next_capsule_value = torch.einsum('nm, bna, namd->bmd', query_key, input, w)
        else:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], 
                                       next_capsule_value.shape[1], self.sqrt_d, self.sqrt_d)
                # _query_key == agreement vector ( a_{i,j})
                _query_key = torch.einsum('bnax, nxdm, bmad->bnm', _input, w, next_capsule_value)
            else:
                _query_key = torch.einsum('bna, namd, bmd->bnm', input, w, next_capsule_value)
            

            # New routing probabilities
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=2)
            query_key = query_key / (torch.sum(query_key, dim=2, keepdim=True) + 1e-10)
            
            if self.matrix_pose:
                # Use new routing values, to update state of parent capsule
                next_capsule_value = torch.einsum('bnm, bnax, nxdm->bmad', query_key, _input, 
                                                  w)
            else:
                next_capsule_value = torch.einsum('bnm, bna, namd->bmd', query_key, input, 
                                                  w)

        # Apply dropout
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0], 
                                       next_capsule_value.shape[1], self.out_d_capsules)
                # Apply layer Norm
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value


# 
class CapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                 kernel_size, stride, matrix_pose, dp, coordinate_add=False):
        super(CapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add
        
        if matrix_pose:
            self.sqrt_d = int(np.sqrt(self.in_d_capsules))
            self.weight_init_const = np.sqrt(out_n_capsules/(self.sqrt_d*in_n_capsules*kernel_size*kernel_size)) 
            self.w = nn.Parameter(self.weight_init_const*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.sqrt_d, self.sqrt_d, out_n_capsules))

        else:
            self.weight_init_const = np.sqrt(out_n_capsules/(in_d_capsules*in_n_capsules*kernel_size*kernel_size)) 
            self.w = nn.Parameter(self.weight_init_const*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, in_d_capsules, out_n_capsules, 
                                                     out_d_capsules))
        
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={}, weight_init_const={}, \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, 
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, self.weight_init_const,
            self.dropout_rate
            )        
    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x capsule_dimension]
        
        # unfold(dimension, size, step) → Tensor: Unfold Extracts sliding local blocks along given dim
        # extracts kernel patches over complete height and width
        unfolded_input = input.unfold(2,size=self.kernel_size,step=self.stride).unfold(3,size=self.kernel_size,step=self.stride)
        unfolded_input = unfolded_input.permute([0,1,5,6,2,3,4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input
    
    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length 
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer

        # This converts (b,32,14,14,16) --> (b,32,3,3,7,7,16) (3X3 patches, 7 in number along both height and width)
        inputs = self.input_expansion(input)

        # print("Expansion: ",input.shape, inputs.shape)

        if self.matrix_pose:
            # W is pose of capsules of layer L
            # Input is capsule of layer L (p_{L})
            w = self.w # klnxdm

            # Converts (b,32,3,3,7,7,16) --> (b,32,3,3,7,7,4,4)
            _inputs = inputs.view(inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3],\
                                  inputs.shape[4], inputs.shape[5], self.sqrt_d, self.sqrt_d) # bnklmhax
            # print(_inputs.shape)

        else:
            w = self.w
            
        if next_capsule_value is None:
            # Routing probabilities in 1st iteration

            query_key = torch.zeros(self.in_n_capsules, self.kernel_size, self.kernel_size, 
                                                self.out_n_capsules).type_as(inputs) # nklm
            
            query_key = F.softmax(query_key, dim=3) # softmax on output number of capsules
            # print("Query :",query_key.shape)
            # print("w :",w.shape)
            # print("input :",_inputs.shape)
            

            if self.matrix_pose:
                # a,x are sqrt_d if matrix pose and not vector pose
                # This performs convolution as well and attention both 

                # next capsule shape is (b,32,7,7,16) just like original input 
                '''
                

                for all b:
                    for all m:
                        for all h:
                            for all w:
                                for all a:
                                    for all d:
                                        for all n: (summing over capsules of layer L)
                                            for all k: 
                                                for all l: (over all patches)
                                                    multiply input pose [(a,x)==(4,4)] with w [(x,d)==(4,4)] 
                '''
                next_capsule_value = torch.einsum('nklm, bnklhwax, klnxdm->bmhwad', query_key, 
                                              _inputs, w)
            

            else:
                # Vectorised implementation
                next_capsule_value = torch.einsum('nklm, bnklhwa, klnamd->bmhwd', query_key, 
                                              inputs, w)
        else:
            if self.matrix_pose:
                # break 16 to (4,4) pose
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],\
                                         next_capsule_value.shape[1], next_capsule_value.shape[2],\
                                         next_capsule_value.shape[3], self.sqrt_d, self.sqrt_d)
                
                # w=(3,3,32,4,4,m), _input=(b,32,3,3,7,7,4,4) , next_capsule_value= (b,m,7,7,4,4)
                _query_key = torch.einsum('bnklhwax, klnxdm, bmhwad->bnklmhw', _inputs, w, 
                                     next_capsule_value)

            else:    
                _query_key = torch.einsum('bnklhwa, klnamd, bmhwd->bnklmhw', inputs, w, 
                                     next_capsule_value)
            
            # Compute new routing probabilities
            _query_key.mul_(self.scale)
            query_key = F.softmax(_query_key, dim=4)
            query_key = query_key / (torch.sum(query_key, dim=4, keepdim=True) + 1e-10)
            
            if self.matrix_pose:
                # Update parent parent using new routing probabilties
                next_capsule_value = torch.einsum('bnklmhw, bnklhwax, klnxdm->bmhwad', query_key, 
                                              _inputs, w)
                # print("others iter : ", next_capsule_value.shape)    
            else:
                next_capsule_value = torch.einsum('bnklmhw, bnklhwa, klnamd->bmhwd', query_key, 
                                              inputs, w)     
        
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            if self.matrix_pose:
                # Correct size of parent capsule
                next_capsule_value = next_capsule_value.view(next_capsule_value.shape[0],\
                                         next_capsule_value.shape[1], next_capsule_value.shape[2],\
                                         next_capsule_value.shape[3], self.out_d_capsules)
                # Layer Norm 
                next_capsule_value = self.nonlinear_act(next_capsule_value)
            else:
                next_capsule_value = self.nonlinear_act(next_capsule_value)
                
        return next_capsule_value




#### Capsule Layers with the proposed bilinear sparse routing ####
class SACapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp):
        super(SACapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules # This is n_caps * h_in * w_in
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

        self.sinhkorn_caps_attn = BilinearSparseRouting(next_bucket_size=self.out_n_capsules, in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='FC', kernel_size=1,
                                                     temperature = 0.75,
                                                    non_permutative = True, sinkhorn_iter = 7, n_sortcut = 2, dropout = 0., current_bucket_size = self.in_n_capsules//8,
                                                    use_simple_sort_net = False)

    
    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.dropout_rate
        )        
    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        print("Input ", input.shape)
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0,2,1)

        print("Transformed ", input.shape)
        batch_size = input.shape[0]
        next_capsule_value = self.sinhkorn_caps_attn(current_pose=input, h_out=1, w_out=1, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value

class SACapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                 kernel_size, stride, matrix_pose, dp, padding=None, coordinate_add=False):
        super(SACapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add
        self.padding = padding
        
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)

        self.sinhkorn_caps_attn = BilinearSparseRouting(next_bucket_size=self.out_n_capsules, in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='conv', kernel_size=kernel_size,
                                                     temperature = 0.75,
                                                        non_permutative = True, sinkhorn_iter = 7, n_sortcut = 1, dropout = 0., current_bucket_size = self.in_n_capsules,
                                                        use_simple_sort_net = False)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={},  \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, 
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, 
            self.dropout_rate
            )       
             
    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        if self.padding:
            input = input.permute([0,1,4,2,3]) #For padding h,w
            if not self.padding%1:
                input = F.pad(input, [self.padding, self.padding, self.padding, self.padding]) #TODO: Padding to maintain same size, change so that caps dim not padded
            else:
                input = F.pad(input, [math.ceil(self.padding), math.floor(self.padding), math.ceil(self.padding), math.floor(self.padding)]) #TODO: Padding to maintain same size, change so that caps dim not padded
            input = input.permute([0,1,3,4,2])
        unfolded_input = input.unfold(2,size=self.kernel_size,step=self.stride).unfold(3,size=self.kernel_size,step=self.stride)
        unfolded_input = unfolded_input.permute([0,1,5,6,2,3,4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input
    
    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length 
        inputs = self.input_expansion(input)
        batch_size = inputs.shape[0]
        h_out = inputs.shape[4]
        w_out = inputs.shape[5] 
        next_capsule_value = self.sinhkorn_caps_attn(current_pose=inputs, h_out=h_out, w_out=w_out, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)                
        return next_capsule_value        







#### Capsule Layers with the proposed bilinear routing without sinkhorn - Ablation study ####
class BACapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, matrix_pose, dp):
        super(BACapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

        self.bilinear_attn = BilinearRouting(next_bucket_size=self.out_n_capsules, in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='FC', kernel_size=1,
                                                     temperature = 0.75,
                                                    non_permutative = True, sinkhorn_iter = 7, n_sortcut = 2, dropout = 0., current_bucket_size = self.in_n_capsules//8,
                                                    use_simple_sort_net = False)

    
    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.dropout_rate
        )        
    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0,2,1)


        batch_size = input.shape[0]
        next_capsule_value = self.bilinear_attn(current_pose=input, h_out=1, w_out=1, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value

class BACapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                 kernel_size, stride, matrix_pose, dp, padding=None, coordinate_add=False):
        super(BACapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add
        self.padding = padding
        
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)

        self.bilinear_attn = BilinearRouting(next_bucket_size=self.out_n_capsules, in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='conv', kernel_size=kernel_size,
                                                     temperature = 0.75,
                                                        non_permutative = True, sinkhorn_iter = 7, n_sortcut = 1, dropout = 0., current_bucket_size = self.in_n_capsules,
                                                        use_simple_sort_net = False)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={},  \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, 
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, 
            self.dropout_rate
            )       
             
    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        if self.padding:
            input = input.permute([0,1,4,2,3]) #For padding h,w
            if not self.padding%1:
                input = F.pad(input, [self.padding, self.padding, self.padding, self.padding]) #TODO: Padding to maintain same size, change so that caps dim not padded
            else:
                input = F.pad(input, [math.ceil(self.padding), math.floor(self.padding), math.ceil(self.padding), math.floor(self.padding)]) #TODO: Padding to maintain same size, change so that caps dim not padded
            input = input.permute([0,1,3,4,2])
        unfolded_input = input.unfold(2,size=self.kernel_size,step=self.stride).unfold(3,size=self.kernel_size,step=self.stride)
        unfolded_input = unfolded_input.permute([0,1,5,6,2,3,4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input
    
    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length 
        inputs = self.input_expansion(input)
        batch_size = inputs.shape[0]
        h_out = inputs.shape[4]
        w_out = inputs.shape[5] 
        next_capsule_value = self.bilinear_attn(current_pose=inputs, h_out=h_out, w_out=w_out, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)                
        return next_capsule_value 






#### Capsule Layers with the linformer projections and unfold operstions
class LACapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, input_img_size, output_img_size, hidden_dim, matrix_pose, dp):
        super(LACapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules # This is n_caps * h_in * w_in
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.input_img_size = input_img_size
        self.output_img_size = output_img_size
        self.matrix_pose = matrix_pose
        self.dropout_rate = dp
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.drop = nn.Dropout(self.dropout_rate)
        self.scale = 1. / (out_d_capsules ** 0.5)

        self.linformer_attention = LinformerProjectionKernel(in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='FC', input_img_size = input_img_size, output_img_size = output_img_size,
                                                    hidden_dim = hidden_dim, kernel_size=1, dropout = 0.)

    
    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, matrix_pose={}, \
            dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, self.matrix_pose,
            self.dropout_rate
        )        
    def forward(self, input, num_iter, next_capsule_value=None):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        # print("Input ", input.shape)
        if len(input.shape) == 5:
            input = input.permute(0, 4, 1, 2, 3)
            input = input.contiguous().view(input.shape[0], input.shape[1], -1)
            input = input.permute(0,2,1)

        # print("Transformed ", input.shape)
        batch_size = input.shape[0]
        next_capsule_value = self.linformer_attention(current_pose=input, h_out=1, w_out=1, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)
        return next_capsule_value

class LACapsuleCONV(nn.Module):
    r"""Applies as a capsule convolutional layer.
    TBD
    """
    def __init__(self, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                 kernel_size, stride, input_img_size, output_img_size, hidden_dim,matrix_pose, dp, padding=None, coordinate_add=False):
        super(LACapsuleCONV, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.input_img_size = input_img_size
        self.output_img_size = output_img_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.matrix_pose = matrix_pose
        self.coordinate_add = coordinate_add
        self.padding = padding
        
        self.nonlinear_act = nn.LayerNorm(out_d_capsules)
        self.dropout_rate = dp
        self.drop = nn.Dropout(self.dropout_rate)

        self.linformer_attention = LinformerProjectionKernel(in_n_capsules=in_n_capsules, in_d_capsules=in_d_capsules, out_n_capsules=out_n_capsules, 
                                                     out_d_capsules=out_d_capsules, matrix_pose=self.matrix_pose, layer_type='conv', input_img_size = input_img_size, output_img_size = output_img_size, 
                                                     hidden_dim=hidden_dim, kernel_size=kernel_size, dropout = 0.)

    def extra_repr(self):
        return 'in_n_capsules={}, in_d_capsules={}, out_n_capsules={}, out_d_capsules={}, \
                    kernel_size={}, stride={}, coordinate_add={}, matrix_pose={},  \
                    dropout_rate={}'.format(
            self.in_n_capsules, self.in_d_capsules, self.out_n_capsules, self.out_d_capsules, 
            self.kernel_size, self.stride, self.coordinate_add, self.matrix_pose, 
            self.dropout_rate
            )       
             
    def input_expansion(self, input):
        # input has size [batch x num_of_capsule x height x width x  x capsule_dimension]
        if self.padding:
            input = input.permute([0,1,4,2,3]) #For padding h,w
            if not self.padding%1:
                input = F.pad(input, [self.padding, self.padding, self.padding, self.padding]) #TODO: Padding to maintain same size, change so that caps dim not padded
            else:
                input = F.pad(input, [math.ceil(self.padding), math.floor(self.padding), math.ceil(self.padding), math.floor(self.padding)]) #TODO: Padding to maintain same size, change so that caps dim not padded
            input = input.permute([0,1,3,4,2])
        unfolded_input = input.unfold(2,size=self.kernel_size,step=self.stride).unfold(3,size=self.kernel_size,step=self.stride)
        unfolded_input = unfolded_input.permute([0,1,5,6,2,3,4])
        # output has size [batch x num_of_capsule x kernel_size x kernel_size x h_out x w_out x capsule_dimension]
        return unfolded_input
    
    def forward(self, input, num_iter, next_capsule_value=None):
        # k,l: kernel size
        # h,w: output width and length 
        inputs = self.input_expansion(input)
        batch_size = inputs.shape[0]
        h_out = inputs.shape[4]
        w_out = inputs.shape[5] 
        next_capsule_value = self.linformer_attention(current_pose=inputs, h_out=h_out, w_out=w_out, next_pose=next_capsule_value)
        next_capsule_value = self.drop(next_capsule_value)
        if not next_capsule_value.shape[-1] == 1:
            next_capsule_value = self.nonlinear_act(next_capsule_value)                
        return next_capsule_value   


