#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from src import layers
import torch.nn as nn
import torch.nn.functional as F
import torch


'''
{'backbone': {'kernel_size': 3, 'output_dim': 128, 'input_dim': 3, 'stride': 2, 'padding': 1, 'out_img_size': 16}, 'primary_capsules': {'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'nu
m_caps': 32, 'padding': 0, 'out_img_size': 16}, 'capsules': [{'type': 'CONV', 'num_caps': 32, 'caps_dim': 16, 'kernel_size': 3, 'stride': 2, 'matrix_pose': True, 'out_img_size': 7}, {'type': 'CONV', 'num_
caps': 32, 'caps_dim': 16, 'kernel_size': 3, 'stride': 1, 'matrix_pose': True, 'out_img_size': 5}], 'class_capsules': {'num_caps': 10, 'caps_dim': 16, 'matrix_pose': True}}
{'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'num_caps': 32, 'padding': 0, 'out_img_size': 16}
'''





# Capsule model with bilinear routing and random projections
class CapsBilinearLinformerUnfoldModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):
        
        super(CapsBilinearLinformerUnfoldModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100':
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                
                output_img_size = params['capsules'][i]['out_img_size']                    
                input_img_size = params['primary_capsules']['out_img_size'] if i==0 else \
                                                               params['capsules'][i-1]['out_img_size']




                self.capsule_layers.append(
                    layers.LACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                input_img_size = input_img_size,
                                output_img_size = output_img_size,
                                hidden_dim= params['capsules'][i]['hidden_dim'],
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                output_img_size = 1
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                    input_img_size = params['primary_capsules']['out_img_size']

                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = 1

                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                    input_img_size = params['capsules'][i-1]['out_img_size']
                
                self.capsule_layers.append(
                    layers.LACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          input_img_size = input_img_size,
                          output_img_size = output_img_size,
                          hidden_dim= params['capsules'][i]['hidden_dim'],
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            output_img_size = 1
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = 1
                
            
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
                input_img_size = params['capsules'][-1]['out_img_size']
                
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
            input_img_size = params['primary_capsules']['out_img_size']
        self.capsule_layers.append(
            layers.LACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  input_img_size = input_img_size,
                  output_img_size = output_img_size,
                  hidden_dim= params['capsules'][i]['hidden_dim'],
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 











# Capsule model
class CapsModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):
        
        super(CapsModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100' or "NIST" in dataset:
              print("Using standard ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
        
        
        ## Primary Capsule Layer (a single CNN)
        # {'kernel_size': 1, 'stride': 1, 'input_dim': 128, 'caps_dim': 16, 'num_caps': 32, 'padding': 0, 'out_img_size': 16}
        print(params['primary_capsules'])
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                # num_in_capsules=32, in_cap_d=16, out_Cap=32, out_dim_cap=16
                # 3x3 kernel, stride 2 and output shape: 7x7
                self.capsule_layers.append(
                    layers.CapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                coordinate_add=False
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    # When there is no Conv layer after primary capsules
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                
                elif params['capsules'][i-1]['type'] == 'CONV':
                    # There are a total of 14X14X32 capsule outputs, each being 16 dimensional 
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.CapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
                )
                                                               
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        
        self.capsule_layers.append(
            layers.CapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))


    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        # Converts Input (b, 3, 14, 14)--> (b, 128, 14, 14)
        c = self.pre_caps(x)
        
        ## Primary Capsule Layer (a single CNN) (Ouput size: b, 512, 14, 14) (32 caps, 16 dim each)
        u = self.pc_layer(c)
        u = u.permute(0, 2, 3, 1) # b, 14, 14, 512
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # b, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # b, 32, 14, 14, 16
        
        # Layer norm
        init_capsule_value = self.nonlinear_act(u)  #capsule_utils.squash(u)

        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        

        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                
                # second to t iterations
                # perform the routing between the 2 capsule layers for some iterations 
                # till you move to next pair of layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        # Output capsule (last layer)
        out = capsule_values[-1]
        out = self.final_fc(out) # fixed classifier for all capsules
        out = out.squeeze() # fixed classifier for all capsules
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules        
        return out 




# Capsule model with bilinear sparse routing
class CapsSAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):
        
        super(CapsSAModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset:
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.SACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.SACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.SACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 



# Capsule model with bilinear routing without sinkhorn
class CapsBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):
        
        super(CapsBAModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset:
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.BACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.BACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.BACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
         
        ## Main Capsule Layers 
        # concurrent routing
        if not self.sequential_routing:
            # first iteration
            # perform initilialization for the capsule values as single forward passing
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                _val = self.capsule_layers[i].forward(_val, 0)
                capsule_values.append(_val) # get the capsule value for next layer
            
            # second to t iterations
            # perform the routing between capsule layers
            for n in range(self.num_routing-1):
                _capsule_values = [init_capsule_value]
                for i in range(len(self.capsule_layers)):
                    _val = self.capsule_layers[i].forward(capsule_values[i], n, 
                                    capsule_values[i+1])
                    _capsule_values.append(_val)
                capsule_values = _capsule_values
        # sequential routing
        else:
            capsule_values, _val = [init_capsule_value], init_capsule_value
            for i in range(len(self.capsule_layers)):
                # first iteration
                __val = self.capsule_layers[i].forward(_val, 0)
                # second to t iterations
                # perform the routing between capsule layers
                for n in range(self.num_routing-1):
                    __val = self.capsule_layers[i].forward(_val, n, __val)
                _val = __val
                capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 


# Capsule model with bilinear routing with dynamic routing
class CapsDBAModel(nn.Module):
    def __init__(self,
                 image_dim_size,
                 params,
                 dataset,
                 backbone,
                 dp,
                 num_routing,
                 sequential_routing=True):
        
        super(CapsDBAModel, self).__init__()
        #### Parameters
        self.sequential_routing = sequential_routing
        
        ## Primary Capsule Layer
        self.pc_num_caps = params['primary_capsules']['num_caps']
        self.pc_caps_dim = params['primary_capsules']['caps_dim']
        self.pc_output_dim = params['primary_capsules']['out_img_size']
        ## General
        self.num_routing = num_routing # >3 may cause slow converging
        
        #### Building Networks
        ## Backbone (before capsule)
        if backbone == 'simple':
            self.pre_caps = layers.simple_backbone(params['backbone']['input_dim'],
                                            params['backbone']['output_dim'],
                                            params['backbone']['kernel_size'], 
                                            params['backbone']['stride'],
                                            params['backbone']['padding'])
        elif backbone == 'resnet':
            # Ouputs 16 X 16 X 128 dim
            if dataset == 'CIFAR10' or dataset == 'CIFAR100'or "NIST" in dataset:
              print("Using CIFAR backbone")
              self.pre_caps = layers.resnet_backbone_cifar(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
            else:
              print("Using New ResNet Backbone")
              self.pre_caps = layers.resnet_backbone_imagenet(params['backbone']['input_dim'], 
                                             params['backbone']['output_dim'],
                                             params['backbone']['stride'])
        
        ## Primary Capsule Layer (a single CNN)
        self.pc_layer = nn.Conv2d(in_channels=params['primary_capsules']['input_dim'],
                                     out_channels=params['primary_capsules']['num_caps'] *\
                                                          params['primary_capsules']['caps_dim'],
                                     kernel_size=params['primary_capsules']['kernel_size'],
                                     stride=params['primary_capsules']['stride'],
                                     padding=params['primary_capsules']['padding'],
                                     bias=False)
        
        #self.pc_layer = nn.Sequential()     

        self.nonlinear_act = nn.LayerNorm(params['primary_capsules']['caps_dim'])
        
        ## Main Capsule Layers        
        self.capsule_layers = nn.ModuleList([])
        for i in range(len(params['capsules'])):
            if params['capsules'][i]['type'] == 'CONV':
                in_n_caps = params['primary_capsules']['num_caps'] if i==0 else \
                                                               params['capsules'][i-1]['num_caps']
                in_d_caps = params['primary_capsules']['caps_dim'] if i==0 else \
                                                               params['capsules'][i-1]['caps_dim']                                                               
                self.capsule_layers.append(
                    layers.DBACapsuleCONV(in_n_capsules=in_n_caps,
                                in_d_capsules=in_d_caps, 
                                out_n_capsules=params['capsules'][i]['num_caps'],
                                out_d_capsules=params['capsules'][i]['caps_dim'],
                                kernel_size=params['capsules'][i]['kernel_size'], 
                                stride=params['capsules'][i]['stride'], 
                                matrix_pose=params['capsules'][i]['matrix_pose'], 
                                dp=dp,
                                coordinate_add=False,
                                padding=params['capsules'][i].get('padding', None)
                            )
                )
            elif params['capsules'][i]['type'] == 'FC':
                if i == 0:
                    in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                                            params['primary_capsules']['out_img_size']
                    in_d_caps = params['primary_capsules']['caps_dim']
                elif params['capsules'][i-1]['type'] == 'FC':
                    in_n_caps = params['capsules'][i-1]['num_caps']
                    in_d_caps = params['capsules'][i-1]['caps_dim']                                           
                elif params['capsules'][i-1]['type'] == 'CONV':
                    in_n_caps = params['capsules'][i-1]['num_caps'] * params['capsules'][i-1]['out_img_size'] *\
                                                                                           params['capsules'][i-1]['out_img_size']  
                    in_d_caps = params['capsules'][i-1]['caps_dim']
                self.capsule_layers.append(
                    layers.DBACapsuleFC(in_n_capsules=in_n_caps, 
                          in_d_capsules=in_d_caps, 
                          out_n_capsules=params['capsules'][i]['num_caps'], 
                          out_d_capsules=params['capsules'][i]['caps_dim'], 
                          matrix_pose=params['capsules'][i]['matrix_pose'],
                          dp=dp
                          )
                )                                                   
        
        ## Class Capsule Layer
        if not len(params['capsules'])==0:
            if params['capsules'][-1]['type'] == 'FC':
                in_n_caps = params['capsules'][-1]['num_caps']
                in_d_caps = params['capsules'][-1]['caps_dim']
            elif params['capsules'][-1]['type'] == 'CONV':    
                in_n_caps = params['capsules'][-1]['num_caps'] * params['capsules'][-1]['out_img_size'] *\
                                                                                   params['capsules'][-1]['out_img_size']
                in_d_caps = params['capsules'][-1]['caps_dim']
        else:
            in_n_caps = params['primary_capsules']['num_caps'] * params['primary_capsules']['out_img_size'] *\
                                                                               params['primary_capsules']['out_img_size']
            in_d_caps = params['primary_capsules']['caps_dim']
        self.capsule_layers.append(
            layers.DBACapsuleFC(in_n_capsules=in_n_caps, 
                  in_d_capsules=in_d_caps, 
                  out_n_capsules=params['class_capsules']['num_caps'], 
                  out_d_capsules=params['class_capsules']['caps_dim'], 
                  matrix_pose=params['class_capsules']['matrix_pose'],
                  dp=dp
                  )
        )
        
        ## After Capsule
        # fixed classifier for all class capsules
        self.final_fc = nn.Linear(params['class_capsules']['caps_dim'], 1)
        # different classifier for different capsules
        #self.final_fc = nn.Parameter(torch.randn(params['class_capsules']['num_caps'], params['class_capsules']['caps_dim']))

    def forward(self, x, lbl_1=None, lbl_2=None):
        #### Forward Pass
        ## Backbone (before capsule)
        c = self.pre_caps(x)
        # print(c.shape)
        # print("Backbone: ", c.shape)
        ## Primary Capsule Layer (a single CNN)
        u = self.pc_layer(c) # torch.Size([100, 512, 14, 14])
        u = u.permute(0, 2, 3, 1) # 100, 14, 14, 512
        # print("Shape:", u.shape)
        u = u.view(u.shape[0], self.pc_output_dim, self.pc_output_dim, self.pc_num_caps, self.pc_caps_dim) # 100, 14, 14, 32, 16
        u = u.permute(0, 3, 1, 2, 4) # 100, 32, 14, 14, 16
        # init_capsule_value = self.nonlinear_act(u)#capsule_utils.squash(u)
        init_capsule_value = u
         
        ## Main Capsule Layers 
        # Sequetial routing only
        if self.sequential_routing:
          
          capsule_values, _val = [init_capsule_value], init_capsule_value
          for i in range(len(self.capsule_layers)):
              routing_coeff = None
              for n in range(self.num_routing):
                  # print("Routing num ", n)
                  new_coeff, __val = self.capsule_layers[i].forward(_val, n, routing_coeff)
                  routing_coeff = new_coeff
              _val = __val
              capsule_values.append(_val)
        
        ## After Capsule
        out = capsule_values[-1]
        # print("out shape, ", out.shape)
        out = self.final_fc(out) # fixed classifier for all capsules
        # print("classifier shape, ", out.shape)
        out = out.squeeze(1) # fixed classifier for all capsules
        out = out.squeeze(2)
        out = out.squeeze(1)
        #out = torch.einsum('bnd, nd->bn', out, self.final_fc) # different classifiers for distinct capsules
        # print("Final shape, ", out.shape)
        return out 

