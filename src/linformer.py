import math
import torch
from torch import nn
from operator import mul
from fractions import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
import numpy as np

########## Linformer projection on each kernel
class LinformerProjectionKernel(nn.Module):
    def __init__(self, 
                in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
                dropout = 0.):
        super().__init__()
  
        self.in_d_capsules = in_d_capsules
        self.out_d_capsules = out_d_capsules
        self.in_n_capsules = in_n_capsules
        self.out_n_capsules = out_n_capsules
        self.input_img_size=input_img_size
        self.output_img_size=output_img_size
        self.hidden_dim=hidden_dim
        self.pose_dim = in_d_capsules
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.matrix_pose = matrix_pose
        self.parameter_sharing = parameter_sharing

        if self.layer_type == 'FC':
            self.kernel_size=1

        if matrix_pose:
            # Random Initialisation of Two matrices
            self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
            
            # w_current =(3,3,32,4,4)
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
        else:
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.pose_dim, self.pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.pose_dim, self.pose_dim))

        
        max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
        heads = 1
        
        if parameter_sharing == "headwise":
            # print("Hello")
            self.E_proj = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, hidden_dim))
    
        else:
            assert (False),"Yet to write the non-headwise method"


        

        # Positional embeddings: (7,7,16)
        # self.rel_embedd = None
        self.dropout = nn.Dropout(dropout)
        print("You are using Bilinear routing with Linformer")


    def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
        
        # print('Using linformer kernels')
        # current pose: (b,32,3,3,7,7,16)
        # if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
        if next_pose is None:
            # ist iteration
            batch_size = current_pose.shape[0]
            if self.layer_type=='conv':
                # (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
                # (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                h_out = h_out
                w_out = w_out
            
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                #w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
            
            #
            # W_current is C_{L} and w_next is N_{L}
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)

            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
            
            if self.matrix_pose:
                # (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
                # print(current_pose.shape)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Multiplying p{L} by C_{L} to change to c_{L}
            # Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
            # Same matrix for the entire batch, output  = (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current) 


            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)
            
            ############## Linformer Projection
            current_pose = current_pose.permute(2,0,1) # (16,49b,288)
            E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

            current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
            current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)

    
            # R_{i,j} = (49b, m, 288)
            dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
            dots = dots.softmax(dim=-2)
            
 
            next_pose_candidates = current_pose  
            # Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
            # multiply with N_{L} 
            # next pose: (49b, m, 16) 
            next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
            
            ###################### Positional Embeddings
            # (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
            next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            # next_pose_candidates = next_pose_candidates + self.rel_embedd
            next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
            next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
            
            if self.matrix_pose:
                # Correct shapes: (49b, m, 4, 4)
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(2)
            
            # Found final pose of next layer by multiplying X with N_{L}
            # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

            # Reshape: (b, 7, 7, m, 16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            
        

            if self.layer_type == 'conv':
                # Reshape: (b,m,7,7,16) (just like original input, without expansion)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            
            elif self.layer_type == 'FC':
                # Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates
        

        else:
            # 2nd to T iterations
            batch_size = next_pose.shape[0]
            if self.layer_type=='conv':
                # Current_pose = (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                
                # next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
                next_pose = next_pose.permute([0,2,3,1,4])
                h_out = next_pose.shape[1]
                w_out = next_pose.shape[2]
           
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                # w_current = (288,4,4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
            
            # w_current = (1,288,4,4)
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)
            
            
            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 4, 4)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Tranformed currentlayer capsules to c_{L}
            # Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current)
            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)

            
            ############## Linformer Projection
            current_pose = current_pose.permute(2,0,1) # (16,49b,288)
            E_proj = self.E_proj.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.hidden_dim) # (288, hidden_dim)            

            current_pose = torch.matmul(current_pose, E_proj) # (16,49b,hidden_dim)
            current_pose = current_pose.permute(1,2,0) # (49b, hidden_Dim, 16)


            ###################### Positonal Embeddings
            # Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
            # print("original ", next_pose.shape)
            next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            # print(next_pose.shape, self.rel_embedd.shape)
            # next_pose = next_pose + self.rel_embedd
                

            # next_pose = (b,m,7,7,16) --> (49b,m,16)   
            next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.matrix_pose:
                # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
                next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose = next_pose.unsqueeze(3)
            
            # Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
            next_pose = torch.matmul(w_next, next_pose)
            

            if self.matrix_pose:
                # next_pose = (49b,m,16)
                next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
            else:
                next_pose = next_pose.squeeze(3)
    
            # Finding scaled alignment scores between updated buckets
            # dots = (49b, m ,288)
            dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
            

            # attention routing along dim=-2 (next layer buckets)
            # Dim=-1 if you wanna invert the inverted attention
            dots = dots.softmax(dim=-2) 
            next_pose_candidates = current_pose

            # Yet to multiply with N_{L} (next_w)
            next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
            
            if self.matrix_pose:
                # next pose: 49b,m,16 --> 49b,m,4,4
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(3)
            
            # Multiplied with N_{j} to get final pose
            # w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
            
            # next_pose_candidates = (b,7,7,m,16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            

            if self.layer_type == 'conv':
                # next_pose_candidates = (b,m,7,7,16)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            elif self.layer_type == 'FC':
                # next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates


















#Capsules Linformer projections but with convolution capsules too

class LinformerProjectionEntireOutImg(nn.Module):
    def __init__(self, 
                in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
                dropout = 0.):
        super().__init__()
  
        self.in_d_capsules = in_d_capsules
        self.out_d_capsules = out_d_capsules
        self.in_n_capsules = in_n_capsules
        self.out_n_capsules = out_n_capsules
        self.input_img_size=input_img_size
        self.output_img_size=output_img_size
        self.hidden_dim=hidden_dim

        self.pose_dim = in_d_capsules
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.matrix_pose = matrix_pose
        self.parameter_sharing = parameter_sharing

        if self.layer_type == 'FC':
            self.kernel_size=1

        if matrix_pose:
            # Random Initialisation of Two matrices
            self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
            
            # w_current =(3,3,32,4,4)
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
        else:
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.pose_dim, self.pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.pose_dim, self.pose_dim))

        
        max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
        heads = 1
        
        if parameter_sharing == "headwise":
            # print("Hello")
            if self.layer_type =='conv':
                self.E_proj = nn.Parameter(0.02*torch.randn(self.in_n_capsules, output_img_size * output_img_size, hidden_dim))
            else:
                self.E_proj = nn.Parameter(0.02*torch.randn(int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), input_img_size * input_img_size, hidden_dim))
    
        else:
            assert (False),"Yet to write the non-headwise method"

        # Positional embeddings: (7,7,16)
        self.rel_embedd = nn.Parameter(torch.randn(output_img_size, output_img_size, self.out_d_capsules), requires_grad=True)
        # self.rel_embedd = None
        self.dropout = nn.Dropout(dropout)
        print("You are using Bilinear routing with Linformer")


    def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
        
        # current pose: (b,32,3,3,7,7,16)
        # if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
        if next_pose is None:
            # ist iteration
            batch_size = current_pose.shape[0]
            if self.layer_type=='conv':
                # (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
                # (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                h_out = h_out
                w_out = w_out
            
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                #w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
            
            #
            # W_current is C_{L} and w_next is N_{L}
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)

            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
            
            if self.matrix_pose:
                # (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
                # print(current_pose.shape)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Multiplying p{L} by C_{L} to change to c_{L}
            # Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
            # Same matrix for the entire batch, output  = (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current) 
            
            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)
            
            

            # Linformer projection
            # (b,3,3,16,32,1,7*7) X (32,49,hidden_dim) --> (b,3,3,16,32,1,hidden_dim)
            if self.layer_type=='conv':
                current_pose = current_pose.reshape(batch_size, self.kernel_size, self.kernel_size , self.pose_dim, self.in_n_capsules, 1, h_out*w_out)
                # print("Input shape: ", current_pose.shape, self.E_proj.shape)
                current_pose = torch.matmul(current_pose, self.E_proj).squeeze(5)
                current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[1]*current_pose.shape[2]*current_pose.shape[4]*current_pose.shape[5], current_pose.shape[3])
                dots=(torch.ones(batch_size, self.out_n_capsules*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules * self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)

            else:
                # Input is (b, num_Caps*input_size*input_size,16) ==(b,5*5*32,16) -> (b,16,32,1,5*5) X (32,25,hidden_dim) --> (b,16,32,1,hidden_dim) -> (b,32*hidden_dim, 16)
                current_pose = current_pose.reshape(batch_size, self.pose_dim, int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), 1, self.input_img_size * self.input_img_size)  
                # print("Input shape: ", current_pose.shape, self.E_proj.shape)
                current_pose = torch.matmul(current_pose, self.E_proj).squeeze(3)
                current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[2]*current_pose.shape[3], current_pose.shape[1])
                dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size* int(self.in_n_capsules/(self.input_img_size * self.input_img_size)) * self.hidden_dim)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
                # print("Input shape: ", current_pose.shape, dots.shape)


            # R_{i,j} = (b, m*7*7, 3*3*32*hidden_dim)
            dots = dots.softmax(dim=-2)
            
 
            next_pose_candidates = current_pose  
            # Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
            # multiply with N_{L} 
            # next pose: (49b, m, 16) 
            next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
            # (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
            next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            
            next_pose_candidates = next_pose_candidates + self.rel_embedd
            next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
            next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
            if self.matrix_pose:
                # Correct shapes: (49b, m, 4, 4)
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(2)
            
            # Found final pose of next layer by multiplying X with N_{L}
            # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

            # Reshape: (b, 7, 7, m, 16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.layer_type == 'conv':
                # Reshape: (b,m,7,7,16) (just like original input, without expansion)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            
            elif self.layer_type == 'FC':
                # Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates
        

        else:
            # 2nd to T iterations
            batch_size = next_pose.shape[0]
            if self.layer_type=='conv':
                # Current_pose = (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                
                # next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
                next_pose = next_pose.permute([0,2,3,1,4])
                h_out = next_pose.shape[1]
                w_out = next_pose.shape[2]
           
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                # w_current = (288,4,4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
            
            # w_current = (1,288,4,4)
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)
            
            
            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 4, 4)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Tranformed currentlayer capsules to c_{L}
            # Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current)
            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)



            # Linformer projection
            # (b,3,3,16,32,1,7*7) X (32,49,hidden_dim) --> (b,3,3,16,32,1,hidden_dim)
            if self.layer_type=='conv':
                # print(current_pose.shape)
                current_pose = current_pose.reshape(batch_size, self.kernel_size, self.kernel_size , self.pose_dim, self.in_n_capsules, 1, h_out*w_out)
                # print("Input shape: ", current_pose.shape, self.E_proj.shape)
                current_pose = torch.matmul(current_pose, self.E_proj).squeeze(5)
                current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[1]*current_pose.shape[2]*current_pose.shape[4]*current_pose.shape[5], current_pose.shape[3])

            else:
                # Input is (b, num_Caps*input_size*input_size,16) ==(b,5*5*32,16) -> (b,16,32,1,5*5) X (32,25,hidden_dim) --> (b,16,32,1,hidden_dim) -> (b,32*hidden_dim, 16)
                current_pose = current_pose.reshape(batch_size, self.pose_dim, int(self.in_n_capsules/(self.input_img_size * self.input_img_size)), 1, self.input_img_size * self.input_img_size)  
                # print("Input shape: ", current_pose.shape, self.E_proj.shape)
                current_pose = torch.matmul(current_pose, self.E_proj).squeeze(3)
                current_pose = current_pose.reshape(current_pose.shape[0], current_pose.shape[2]*current_pose.shape[3], current_pose.shape[1])



            
            # Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
            # print("original ", next_pose.shape)
            next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            # print(next_pose.shape, self.rel_embedd.shape)
            next_pose = next_pose + self.rel_embedd
                


            # next_pose = (b,m,7,7,16) --> (49b,m,16)   
            next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.matrix_pose:
                # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
                next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose = next_pose.unsqueeze(3)
            
            # Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)

            next_pose = torch.matmul(w_next, next_pose)
            

            if self.matrix_pose:
                # next_pose = (b,49m,16)
                if self.layer_type=='conv':
                    next_pose = next_pose.view(batch_size, self.out_n_capsules*h_out*w_out,  self.pose_dim)
                else:
                    next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
                   
            else:
                next_pose = next_pose.squeeze(3)
    
            
            # Finding scaled alignment scores between updated buckets
            # dots = (49b, m ,288*hidden_dim)
            
            dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
            # print("dots time shape: ", current_pose.shape, next_pose.shape, dots.shape)
            

            # attention routing along dim=-2 (next layer buckets)
            # Dim=-1 if you wanna invert the inverted attention
            dots = dots.softmax(dim=-2) 
            next_pose_candidates = current_pose

            # Yet to multiply with N_{L} (next_w)

            next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
            # print("Netx canditate: ", next_pose_candidates.shape)

            if self.matrix_pose:
                # next pose: 49b,m,16 --> 49b,m,4,4
                next_pose_candidates=next_pose_candidates.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(3)
            
            # Multiplied with N_{j} to get final pose
            # w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
            
            # next_pose_candidates = (b,7,7,m,16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.layer_type == 'conv':
                # next_pose_candidates = (b,m,7,7,16)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            elif self.layer_type == 'FC':
                # next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates








class BilinearProjectionWithEmbeddings(nn.Module):
    def __init__(self, 
                in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                matrix_pose, layer_type, input_img_size, output_img_size, hidden_dim=None, kernel_size=None, parameter_sharing='headwise',
                dropout = 0.):
        super().__init__()
  
        self.in_d_capsules = in_d_capsules
        self.out_d_capsules = out_d_capsules
        self.in_n_capsules = in_n_capsules
        self.out_n_capsules = out_n_capsules
        
        self.pose_dim = in_d_capsules
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.matrix_pose = matrix_pose
        self.parameter_sharing = parameter_sharing

        if self.layer_type == 'FC':
            self.kernel_size=1

        if matrix_pose:
            # Random Initialisation of Two matrices
            self.matrix_pose_dim = int(np.sqrt(self.in_d_capsules))
            
            # w_current =(3,3,32,4,4)
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim))
        else:
            self.w_current = nn.Parameter(0.02*torch.randn(kernel_size, kernel_size,
                                                     in_n_capsules, self.pose_dim, self.pose_dim))
            self.w_next = nn.Parameter(0.02*torch.randn(
                                                     out_n_capsules, self.pose_dim, self.pose_dim))

        
        max_seq_len = self.kernel_size*self.kernel_size*self.in_n_capsules
        heads = 1
        

        # Positional embeddings: 2 embeddings (1,7,1,8) and (1,1,7,8)
        self.rel_embedd_h = nn.Parameter(torch.randn(1, output_img_size,1, self.out_d_capsules //2), requires_grad=True)
        self.rel_embedd_w = nn.Parameter(torch.randn(1, 1, output_img_size, self.out_d_capsules //2), requires_grad=True)

        # self.rel_embedd = None
        self.dropout = nn.Dropout(dropout)
        print("You are using Bilinear routing with Linformer")


    def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
        
        # current pose: (b,32,3,3,7,7,16)
        # if FC current pose is (b, numcaps*h_in*w_in, caps_dim)
        if next_pose is None:
            # ist iteration
            batch_size = current_pose.shape[0]
            if self.layer_type=='conv':
                # (b, h_out, w_out, num_capsules, kernel_size, kernel_size, capsule_dim)
                # (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                h_out = h_out
                w_out = w_out
            
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                #w_current =(3,3,32,4,4) --> (3*3*32, 4, 4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
            
            #
            # W_current is C_{L} and w_next is N_{L}
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)

            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)#view error
            
            if self.matrix_pose:
                # (b*7*7, 3*3*32, 4, 4) = (49b, 288, 4, 4)
                # print(current_pose.shape)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Multiplying p{L} by C_{L} to change to c_{L}
            # Current pose: (49b, 288, 4, 4), w_current = (1, 288, 4, 4)
            # Same matrix for the entire batch, output  = (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current) 
            

            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)
            
            # R_{i,j} = (49b, m, 288)
            dots=(torch.ones(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size*self.in_n_capsules)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
            dots = dots.softmax(dim=-2)
            
 
            next_pose_candidates = current_pose  
            # Multiplies r_{i,j} with c_{L} ( no sorting in the 1st iteration) to give X. Still have to
            # multiply with N_{L} 
            # next pose: (49b, m, 16) 
            next_pose_candidates = torch.einsum('bij,bje->bie', dots, next_pose_candidates)
            
            ###################### Positional Embeddings
            # (49b,m,16) --> (b,m,7,7,16) + rel_embedding (7,7,16) and then reshaped to (49b,m,16)
            next_pose_candidates = next_pose_candidates.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            # next_pose_candidates = next_pose_candidates + self.rel_embedd
            next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)
            next_pose_candidates = next_pose_candidates.reshape(-1,next_pose_candidates.shape[3], next_pose_candidates.shape[4])
            
            if self.matrix_pose:
                # Correct shapes: (49b, m, 4, 4)
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1], self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(2)
            
            # Found final pose of next layer by multiplying X with N_{L}
            # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)

            # Reshape: (b, 7, 7, m, 16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            
            
            ###################### Positional Embeddings in the end
            next_pose_candidates = next_pose_candidates.permute(0,3,1,2,4) #(b,m,7,7,16) 
            next_pose_candidates_h, next_pose_candidates_w = next_pose_candidates.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
            # adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,8)
            next_pose_candidates = torch.cat((next_pose_candidates_h + self.rel_embedd_h, next_pose_candidates_w + self.rel_embedd_w), dim=4)
            # next_pose_candidates = next_pose_candidates+self.rel_embedd
            next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)

            

            if self.layer_type == 'conv':
                # Reshape: (b,m,7,7,16) (just like original input, without expansion)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            
            elif self.layer_type == 'FC':
                # Reshape: (b, 1, 1, m, 16) --> (b, 1, m, 16) (h_out, w_out ==1)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates
        

        else:
            # 2nd to T iterations
            batch_size = next_pose.shape[0]
            if self.layer_type=='conv':
                # Current_pose = (b,7,7,32,3,3,16)
                current_pose = current_pose.permute([0,4,5,1,2,3,6])
                
                # next_pose = (b,m,7,7,16) --> (b,7,7,m,16)
                next_pose = next_pose.permute([0,2,3,1,4])
                h_out = next_pose.shape[1]
                w_out = next_pose.shape[2]
           
            elif self.layer_type=='FC':
                h_out = 1
                w_out = 1
            
            pose_dim = self.pose_dim
            w_current = self.w_current
            w_next = self.w_next
            if self.matrix_pose:
                # w_current = (288,4,4)
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                w_current = w_current.view(self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim) 
            
            # w_current = (1,288,4,4)
            w_current = w_current.unsqueeze(0)  
            w_next = w_next.unsqueeze(0)
            
            
            current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 4, 4)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)#replace the 2 reshapes
            else:
                current_pose = current_pose.unsqueeze(2)
            
            # Tranformed currentlayer capsules to c_{L}
            # Multiply (49b, 288, 4, 4) with (1,288,4,4) --> (49b, 288, 4, 4)
            current_pose = torch.matmul(current_pose, w_current)
            
            if self.matrix_pose:
                # Current_pose = (49b, 288, 16)
                current_pose = current_pose.reshape(batch_size*h_out*w_out, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                current_pose = current_pose.squeeze(2)

            
            ###################### Positonal Embeddings
            # Adding positional embeddings to next pose: (b,7,7,m,16) -->(b,m,7,7,16)+(7,7,16)
            # print("original ", next_pose.shape)
            next_pose = next_pose.reshape(batch_size,self.out_n_capsules, h_out, w_out,  self.pose_dim)
            # print(next_pose.shape, self.rel_embedd.shape)
            # next_pose = next_pose + self.rel_embedd
                

            # next_pose = (b,m,7,7,16) --> (49b,m,16)   
            next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.matrix_pose:
                # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
                next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose = next_pose.unsqueeze(3)
            
            # Tranform next pose using N_{L}: w_next = (49b,m,4,4) * (1,m,4,4)
            next_pose = torch.matmul(w_next, next_pose)
            

            if self.matrix_pose:
                # next_pose = (49b,m,16)
                next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
            else:
                next_pose = next_pose.squeeze(3)
    
            # Finding scaled alignment scores between updated buckets
            # dots = (49b, m ,288)
            dots = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
            

            # attention routing along dim=-2 (next layer buckets)
            # Dim=-1 if you wanna invert the inverted attention
            dots = dots.softmax(dim=-2) 
            next_pose_candidates = current_pose

            # Yet to multiply with N_{L} (next_w)
            next_pose_candidates = torch.einsum('bji,bie->bje', dots, next_pose_candidates)
            
            if self.matrix_pose:
                # next pose: 49b,m,16 --> 49b,m,4,4
                next_pose_candidates = next_pose_candidates.view(next_pose_candidates.shape[0], next_pose_candidates.shape[1],self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                next_pose_candidates = next_pose_candidates.unsqueeze(3)
            
            # Multiplied with N_{j} to get final pose
            # w_next: (49b,m,4,4); b_next_pose_candidates: (49b,m , 4, 4)
            next_pose_candidates = torch.matmul(next_pose_candidates, w_next)
            
            # next_pose_candidates = (b,7,7,m,16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            

            ###################### Positional Embeddings in the end
            next_pose_candidates = next_pose_candidates.permute(0,3,1,2,4) #(b,m,7,7,16) 
            next_pose_candidates_h, next_pose_candidates_w = next_pose_candidates.split(self.pose_dim // 2, dim=4) # (b,m,7,7,8) and (b,m,7,7,8)
            # adding and concatenating (1,7,1,8) and (1,1,7,8) to (b,m,7,7,8)
            next_pose_candidates = torch.cat((next_pose_candidates_h + self.rel_embedd_h, next_pose_candidates_w + self.rel_embedd_w), dim=4)
            # next_pose_candidates = next_pose_candidates+self.rel_embedd
            next_pose_candidates = next_pose_candidates.permute(0,2,3,1,4)

            if self.layer_type == 'conv':
                # next_pose_candidates = (b,m,7,7,16)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            elif self.layer_type == 'FC':
                # next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates




# temp = torch.randn((2, 3, 32, 32))
# conv = AttentionConv(3, 16, kernel_size=3, padding=1)
# print(conv(temp).size())

