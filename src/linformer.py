import math
import torch
from torch import nn
from operator import mul
from fractions import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
import numpy as np


#Capsules WITHOUT Sinkhorn
class BilinearRouting(nn.Module):
    def __init__(self, next_bucket_size, 
                in_n_capsules, in_d_capsules, out_n_capsules, out_d_capsules, 
                matrix_pose, layer_type, kernel_size=None,
                temperature = 0.75,
        non_permutative = True, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., current_bucket_size = None,
        use_simple_sort_net = False):
        super().__init__()
        self.next_bucket_size = next_bucket_size
        self.current_bucket_size = default(current_bucket_size, next_bucket_size)
        assert not (self.next_bucket_size != self.current_bucket_size and n_sortcut == 0), 'sortcut must be used if the query buckets do not equal the key/value buckets'

        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut
        self.in_d_capsules = in_d_capsules
        self.out_d_capsules = out_d_capsules
        self.in_n_capsules = in_n_capsules
        self.out_n_capsules = out_n_capsules
        
        self.pose_dim = in_d_capsules
        self.layer_type = layer_type
        self.kernel_size = kernel_size
        self.matrix_pose = matrix_pose

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
        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, self.current_bucket_size, max_seq_len // self.current_bucket_size, self.in_d_capsules * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, self.next_bucket_size, self.current_bucket_size, self.in_d_capsules, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)

        self.dropout = nn.Dropout(dropout)
        print("You are using Bilinear routing without sinkhorn")


    def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
        
        # current pose: (b,32,3,3,7,7,16)
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





