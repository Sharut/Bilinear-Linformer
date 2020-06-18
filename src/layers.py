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


def get_EF(input_size, dim):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    """
    EF = nn.Linear(input_size, dim)
    torch.nn.init.xavier_normal_(EF.weight)
    return EF

#### Bilinear Linformer Capsule Layer ####
class BilinearLinformerCapsuleFC(nn.Module):
    r"""Applies as a capsule fully-connected layer.
    TBD
    """

    def __init__(self, hidden_dim, input_size, in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, kernel_size, stride, padding, group_size, parameter_sharing, matrix_pose, dp):

        super(BilinearLinformerCapsuleFC, self).__init__()
        self.in_n_capsules = in_n_capsules
        self.in_d_capsules = in_d_capsules
        self.out_n_capsules = out_n_capsules
        self.out_d_capsules = out_d_capsules
        self.matrix_pose = matrix_pose
        self.hidden_dim =hidden_dim
        
        assert self.in_n_capsules % (input_size * input_size) == 0, "Something fishy"
        self.actual_channels = int(self.in_n_capsules/(input_size * input_size))

        self.current_grouped_conv = nn.Conv2d(in_channels=self.actual_channels*in_d_capsules,
                                     out_channels=self.actual_channels*in_d_capsules,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=group_size,
                                     bias=False)

        self.next_grouped_conv = nn.Conv2d(in_channels=out_n_capsules*out_d_capsules,
                                     out_channels=out_n_capsules*out_d_capsules,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=group_size,
                                     bias=False)
        
        
        BilinearOutImg_size = int((input_size - kernel_size + 2* padding)/stride)+1
        print("Helooooo", BilinearOutImg_size, " ", self.actual_channels)
        if parameter_sharing != "layerwise":
            self.E_proj = get_EF(BilinearOutImg_size*BilinearOutImg_size * self.actual_channels , hidden_dim)
            self.F_proj = get_EF(BilinearOutImg_size*BilinearOutImg_size * self.actual_channels, hidden_dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj
          
               

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
    def forward(self, current_pose, num_iter, next_pose = None, h_out=1, w_out=1):
        # b: batch size
        # n: num of capsules in current layer
        # a: dim of capsules in current layer
        # m: num of capsules in next layer
        # d: dim of capsules in next layer
        # current_pose shape: (b, num_Capsules, height, width, caps_dim), eg (b,32,16,16,256)
        input_height, input_width = current_pose.shape[2], current_pose.shape[3]
        batch_size = current_pose.shape[0]
        pose_dim=self.in_d_capsules
        # Applying grouped convolution across capsule dimension
        if len(current_pose.shape) == 5:
            current_pose = current_pose.permute(0,2,3,1,4)
            current_pose = current_pose.contiguous().view(current_pose.shape[0], current_pose.shape[1], current_pose.shape[2],-1)
            current_pose = current_pose.permute(0,3,1,2)

        # current_pose : (b, 32*256, 16, 16) --> (b, 32*256, a, b) --> (b,32*a*b,256) == (b,n,d) in sequences
        current_pose = self.current_grouped_conv(current_pose)
        
        current_pose = current_pose.reshape(current_pose.shape[0], self.in_d_capsules, -1)
        current_pose = current_pose.permute(0,2,1)
        # Headwise sharing across capsule types, key-value sharing, layer-wise sharing
        # shape : (b, hidden_dim, caps_dim)
        # print(current_pose.shape)
        
        transposed = torch.transpose(current_pose, 1, 2)
        # print(transposed.shape)
        k_val = self.E_proj(transposed)
        v_val = self.F_proj(transposed)

        # shape: (b, new_num_capsules, caps_dim)
        k_val, v_val =k_val.permute(0,2,1), v_val.permute(0,2,1)
        new_n_capsules = int(k_val.shape[1])
        # print('Perm shape: ', k_val.shape," ", v_val.shape)
        
        if next_pose is None:
            # print(batch_size,self.out_n_capsules,  pose_dim)
            dots = (torch.ones(batch_size*h_out*w_out, self.out_n_capsules, new_n_capsules)* (pose_dim ** -0.5)).type_as(k_val).to(k_val)
            dots = F.softmax(dots, dim=-2)
            # print(dots.shape, v_val.shape)
            # output shape: b, out_caps, caps_dim
            next_pose = torch.einsum('bji,bie->bje', dots, v_val)
            next_pose= next_pose.reshape(next_pose.shape[0],next_pose.shape[1]*next_pose.shape[2],1,1)
            next_pose = self.next_grouped_conv(next_pose)
        
            next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)
            return next_pose
        else:
            # next pose: (b,m,h_out,w_out,out_caps_dim)
            h_out = next_pose.shape[2]
            w_out = next_pose.shape[3]
            next_pose = next_pose.permute(0,2,3,1,4)
            next_pose = next_pose.contiguous().view(next_pose.shape[0], next_pose.shape[1], next_pose.shape[2],-1)
            next_pose = next_pose.permute(0,3,1,2)
            next_pose = self.next_grouped_conv(next_pose)
            next_pose = next_pose.reshape(next_pose.shape[0], self.out_n_capsules * next_pose.shape[2] * next_pose.shape[3], self.out_d_capsules)

            dots = torch.einsum('bje,bie->bji', next_pose, k_val) * (pose_dim ** -0.5) 
            dots = dots.softmax(dim=-2) 
            next_pose = torch.einsum('bji,bie->bje', dots, v_val)
            next_pose= next_pose.reshape(next_pose.shape[0],next_pose.shape[1]*next_pose.shape[2],1,1)
            next_pose = self.next_grouped_conv(next_pose)
            next_pose =next_pose.reshape(batch_size, self.out_n_capsules, h_out, w_out, pose_dim)


        # Apply dropout
        next_pose = self.drop(next_pose)
        if not next_pose.shape[-1] == 1:
            if self.matrix_pose:
                next_pose = next_pose.view(next_pose.shape[0], 
                                       next_pose.shape[1], self.out_d_capsules)
                next_pose = self.nonlinear_act(next_pose)
            else:
                next_pose = self.nonlinear_act(next_pose)
        return next_pose


# 


