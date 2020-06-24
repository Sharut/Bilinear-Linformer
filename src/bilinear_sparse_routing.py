import math
import torch
from torch import nn
from operator import mul
from fractions import gcd
import torch.nn.functional as F
from inspect import isfunction
from functools import partial, wraps, reduce
import numpy as np

# helper functions

def identity(x, *args, **kwargs): return x

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def divisible_by(num, divisor):
    return num % divisor == 0

def lcm(*numbers):
    return int(reduce(lambda x, y: (x * y) / gcd(x, y), numbers, 1))

def all_none(*arr):
    return all(el is None for el in arr)

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def rotate_left(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(n, None))
    r = (*pre_slices, slice(0, n))
    return torch.cat((t[l], t[r]), dim=dim)

def rotate_right(t, n, dim=0):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(-n, None))
    r = (*pre_slices, slice(None, -n))
    return torch.cat((t[l], t[r]), dim=dim)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    '''
    The reduce(fun,seq) function is used to apply a particular function passed in its 
    argument to all of the list elements mentioned in the sequence passed along.

    Working : 

    1. At first step, first two elements of sequence are picked and the result is obtained.
    2. Next step is to apply the same function to the previously attained result and the 
       number just succeeding the second element and the result is again stored.
    3. This process continues till no more elements are left in the container.
    4. The final returned result is returned and printed on console.

    '''
    # operator.mul is simple multiplication
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def merge_heads(h, v):
    b, t, d = v.shape
    return v.view(b, t, h, -1).transpose(1, 2).reshape(b, h, t, -1)

def split_heads(h, v):
    *_, t, d = v.shape
    return v.view(-1, h, t, d).transpose(1, 2).reshape(-1, t, d * h)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def bucket(buckets, t, dim=1):
    # T is (-1,t) size, shape is [1,t]
    shape = list(t.shape)
    # Now shape = [1,buckets, -1]
    shape[dim:dim+1] = [buckets, -1]
    # t is reshapes into buckets
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    # Sinkhorn sorting normalization
    # r is in log domain
    n = r.shape[1]
    for _ in range(n_iters):
        # row and column scaling
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    
    # Change to log domain
    r = log(r)

    # Sample gumble noise
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    
    #
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def reorder_buckets(t, r):
    # r is reordering matrix (b, num_bucket_query, num_bucket_key)
    # t is (bucketed query/key): (b, num_buckets, bucket_size, dimension_embedding)
    # Simple matrix multiplication along "v" (last dimension)
    return torch.einsum('buv,bvtd->butd', r, t)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def cumavg(t, dim):
    r = torch.arange(1, t.shape[dim] + 1, device=t.device, dtype=t.dtype)
    expand_slice = [None] * len(t.shape)
    expand_slice[dim] = slice(None, None)
    return t.cumsum(dim=dim) / r[tuple(expand_slice)]

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def expand_dim(t, dim, k):
    '''
    This fucntion tiles/expands a function expands a given tensor (t)
    by an amount k
    eg if t=(10,8), k=3;  then if dim=0, t=(30,8) ;  if dim=1, t=(10,24)
    '''
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k

    '''
    expand: Returns a new view of the self tensor with singleton 
    dimensions expanded to a larger size. Like tile, it just
    makes multiple copies of current tensor and concatenates in desired shape
    '''
    return t.expand(*expand_shape)

def expand_batch_and_merge_head(b, t):
    
    # Shape of t: (1, heads, dim, max_buckets)
    # This variable shape = (heads, dim, max_buckets)
    shape = list(t.squeeze(0).shape)
    
    # This function is tiling/copying t across dimension 0, "b" (dim_head) times
    # New shape of t is (b,heads,dim,max_buckets)
    t = expand_dim(t, 0, b)

    # shape[0] = num_heads, so make shape[0]=num_heads * dim_head
    shape[0] = shape[0] * b

    # Reshaped into (num_heads * dim_head, dim, max_buckets)
    return t.reshape(*shape)

def differentiable_topk(x, k, temperature=0.75):
    # x = (bh, num_buckets, max_buckets)
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        # softmax across max_buckets
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

# helper classes

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x

# positional embeddings
class AxialPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len, axial_shape = ()):
        super().__init__()
        assert reduce(mul, axial_shape, 1) == max_seq_len, 'axial position shape must multiply up to max sequence length'

        self.dim = dim
        self.seq_len = max_seq_len
        self.shape = axial_shape

        self.weights = ParameterList(self, 'weights', len(axial_shape))

        for ind, shape in enumerate(self.shape):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    def forward(self, x):
        b, t, e = x.shape
        embs = []

        for ax_emb in self.weights.to_list():
            expand_shape = (b, *self.shape, self.dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.seq_len, self.dim)
            embs.append(emb)

        pos_emb = sum(embs)
        return pos_emb[:, :t].to(x)

class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

# Local attention
class LocalAttention(nn.Module):
    def __init__(self, bucket_size, causal = False, look_backward = 1, look_forward = 0, dropout = 0.):
        super().__init__()
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        bucket_size, causal, look_backward, look_forward = self.bucket_size, self.causal, self.look_backward, self.look_forward
        # Dimensions of q: b x h x t x e
        # t is Length of Sequence / Number of Tokens
        b, h, t, e, device, dtype = *q.shape, q.device, q.dtype
        
        # Number of buckets
        buckets = t // bucket_size

        # Ticker is tensor [[0,1,2,3.... t-1]] (-1, t) dimensional
        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        

        '''
        If say t=15, bucket size=3 then
        b_t = torch.Tensor([[[ 0,  1,  2],
                     [ 3,  4,  5],
                     [ 6,  7,  8],
                     [ 9, 10, 11],
                     [12, 13, 14]]])
        '''
        b_t = bucket(buckets, ticker)

        
        # Basically set ind_from=0, ind_to=1 in merge_dims() function 
        merge_batch_and_heads = partial(merge_dims, 0, 1)

        # passes each q,k,v as tensors to the function merge_batch_and_heads
        # This merge first 2 dimensions of a tensor and reshapes it eg, q=(8,6,4,9), output = (48,4,9)
        q, k, v = map(merge_batch_and_heads, (q, k, v))

        
        bucket_fn = partial(bucket, buckets)
        # bq = bucket(buckets, q) ; bk = bucket(buckets, k) ; bv = bucket(buckets, v) 
        bq, bk, bv = map(bucket_fn, (q, k, v))


        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)
        mask_value = max_neg_value(dots)

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = map(bucket_fn, (q_mask, kv_mask))

            mk = look_around(mk, pad_value = False, **look_around_kwargs)
            mask = (mq[:, None, :, :, None] * mk[:, None, :, None, :])
            mask = merge_batch_and_heads(mask.expand(-1, h, -1, -1, -1))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(b, h, t, e)
        return out




# non-causal sortnet and sinkhorn attention
class SimpleSortNet(nn.Module):
    def __init__(self, heads, bucket_size, max_buckets, dim, non_permutative, temperature, sinkhorn_iter):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.max_buckets = max_buckets
        self.bucket_size = bucket_size
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        self.linear = nn.Parameter(torch.randn(1, heads, dim, max_buckets))
        self.act = nn.ReLU()

    def forward(self, q, k, topk=1):

        # Given query (q) and key (k)
        # let q = (48, 15, 11), bucket size=3
        # t is length of tokens in q
        # _ refers to length of embedding, also referred to as "dim"
        bh, t, _ = q.shape

        # self.heads= number of heads (8); b=6
        b = bh // self.heads
        buckets = t // self.bucket_size # num_buckets=5

        # let q.shape = (a,t,b), then b_q shape = (a,num_buckets,bucket_size,b) == (48,5,3,11)
        b_q, b_k = bucket(buckets, q), bucket(buckets, k)

        # Sum all elements inside a bucket for both query (48,5,11) and key (48,5,11)
        # and concatenate along (embedding dimension) to give x of shape(48,5,22) 
        
        '''
        Concatenating key and query vectors to extend the method to encoder-decoder attention 
        Paper only has self attention incorporated.
        '''
        x = torch.cat((b_q.sum(dim=2), b_k.sum(dim=2)), dim=-1)

        
        # b = 6 (size of each head), W shape is (1,8,6,max_buckets) --> (48,dim,max_buckets)
        W = expand_batch_and_merge_head(b, self.linear)
        
        # (@) operator : The matrix multiplication(s) are done between the last two dimensions
        # Multiplication indepedent of head.
        # x=(bh, num_buckets, dim), W=(bh, dim, max_buckets); R=(bh, num_buckets, max_buckets)
        # Simple feed forward net (Relu(W.X))
        R = self.act(x @ W)

        # Normlaize R to make to doubly stochastic/ permutation matrix
        return differentiable_topk(R, k=topk) if self.non_permutative else gumbel_sinkhorn(R, self.sinkhorn_iter, self.temperature)



class AttentionSortNet(nn.Module):
    def __init__(self, heads, bucket_size, kv_bucket_size, dim, non_permutative, temperature, sinkhorn_iter, n_sortcut = 0):
        super().__init__()
        self.heads = heads
        self.bucket_size = bucket_size
        self.kv_bucket_size = kv_bucket_size
        self.dim = dim
        self.non_permutative = non_permutative
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter
        # n_sortcut is the budget parameter/ top n chosen buckets
        self.n_sortcut = n_sortcut

    def forward(self, q, k, topk=1):
        bh, *_, bucket_size, kv_bucket_size, device, dtype, dim = *q.shape, self.bucket_size, self.kv_bucket_size, q.device, q.dtype, self.dim
        b = bh // self.heads

        # b is bucket index, h is head index, t is length bucket, d is dimension of embedding inside bucket
        #let q=(48,15,11), buckets=5, bucket_size=3, 11=embedding dimension
        
        # Separate bucket sizes for query and keys;
        buckets = q.shape[1] // bucket_size
        kv_buckets = k.shape[1] // kv_bucket_size

        # b_q = (48,5,3,11)
        b_q = bucket(buckets, q) if self.n_sortcut == 0 else bucket(1, q)
        b_k = bucket(kv_buckets, k)

        #  Mean across all tokens in a bucket for query and key; sq = (48,5,11)
        sq = b_q.mean(dim=2)
        sk = b_k.mean(dim=2)

        # Agreement
        # Don't initiase R using a simple feed forward net
        # Make R as the attention matrix itself to incorporate decoder-encoder attention
        # R_{i,j} gives agreement score between ith query and jth Key
        
        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)
        if self.non_permutative:
            k = topk if self.n_sortcut == 0 else self.n_sortcut
            return differentiable_topk(R, k=k)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)


class SinkhornAttention(nn.Module):
    '''
    This code assumes that bucket_size for
    both query/key-value is fixed. Although
    the num_buckets might vary
    '''
    def __init__(self, bucket_size, dim, dim_heads, heads, max_seq_len, temperature = 0.75, non_permutative = True, sinkhorn_iter = 7, n_sortcut = 0, dropout = 0., kv_bucket_size = None, use_simple_sort_net = False, n_top_buckets = 1):
        super().__init__()
        self.bucket_size = bucket_size
        self.kv_bucket_size = default(kv_bucket_size, bucket_size)

        self.dim = dim
        self.heads = heads
        self.temperature = temperature
        self.non_permutative = non_permutative
        self.sinkhorn_iter = sinkhorn_iter
        self.n_sortcut = n_sortcut

        '''
        "SimpleSortNet" uses Feed forward net to intialise R;
        "AttentionSortNet" uses attention matrix to initialise R (encoder-decoder attention)
        as proposed by Vaswani, 2017  - Tranformer Networks
        
        '''

        if use_simple_sort_net:
            self.sort_net = SimpleSortNet(heads, self.kv_bucket_size, max_seq_len // self.kv_bucket_size, dim_heads * 2, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter)
        else:
            self.sort_net = AttentionSortNet(heads, self.bucket_size, self.kv_bucket_size, dim_heads, non_permutative = non_permutative, temperature = temperature, sinkhorn_iter = sinkhorn_iter, n_sortcut = n_sortcut)

        self.n_top_buckets = n_top_buckets
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, q_mask = None, kv_mask = None):
        b, h, t, d_h, n_top, d, heads, temperature, bucket_size, kv_bucket_size, device = *q.shape, self.n_top_buckets, self.dim, self.heads, self.temperature, self.bucket_size, self.kv_bucket_size, q.device

        # d_h is embedding dimension, t is length of sequence, h is head index, b is batch size
        bh = b * h
        buckets = q.shape[2] // bucket_size
        kv_buckets = k.shape[2] // kv_bucket_size
        n_top = min(n_top, kv_buckets)

        # This merge first 2 dimensions of a tensor and reshapes it 
        #eg, q=(6,8,15,11), output = (48,15,11). So it basically merges the head ( num_heads * head_dim)
        # In general, it merges all dimensions from start_ind to end_ind
        merge_batch_head = partial(merge_dims, 0, 1)
        
        q, k, v = map(merge_batch_head, (q, k, v))

        # bucket query, key, values
        # Different buckets for (k,v) and query; b_k/b_v=(48, 5(bucket_index), 3(bucket_size), 11)
        # let b_q = (48, 10, 3, 11) , b_k = (49, 5, 3, 11)
        b_q = bucket(buckets, q) 
        b_k, b_v = map(partial(bucket, kv_buckets), (k, v))

        
        bsz = b_k.shape[2]

        # calculate reordering matrix R with SortNet; R = (b, 10(num_buckets_query), 5(num_buckets_key))
        R = self.sort_net(q, k, topk=n_top)
        # Typeas: Returns this tensor cast to the type of "q"
        # to: Returns a Tensor with same torch.dtype and torch.device as "q" 
        R = R.type_as(q).to(q)

        # Reordered buckets for both keys and values, not queries, result = (48,10,3,11)
        # num_query_buckets = buckets (as stated above)
        b_k_r = reorder_buckets(b_k, R)
        b_v_r = reorder_buckets(b_v, R)


        # choose the top n ranked buckets for all query buckets
        # n_sortcut is top n out of total buckets
        if self.n_sortcut > 0:
            # output = (48, top_k, 3, 11) --> (48, 1, topk*3, 11)
            b_k_r = b_k_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h) 
            b_v_r = b_v_r[:, 0:self.n_sortcut].reshape(bh, 1, -1, d_h)
            
            # This expands tensor b_k_r along dimension 1 , number of buckets times. 
            # (48, 1, topk*3, 11) --> (48, 10,topk*3,11)
            b_k_r = expand_dim(b_k_r, 1, buckets)
            b_v_r = expand_dim(b_v_r, 1, buckets)
        else:
            # Reshape to confirm dimensions: (48,10,3,11)
            b_k_r = b_k_r.reshape(bh, buckets, -1, d_h)
            b_v_r = b_k_r.reshape(bh, buckets, -1, d_h)

        # If Num_buckets(q) == Num_buckets(K/V), then
        # Concatenate the sorted buckets, with unsorted ones
        # Concatenate to have contextual key/value pair and not just self-attention
        b_k = torch.cat((b_k_r, b_k), dim=2) if buckets == kv_buckets else b_k_r
        b_v = torch.cat((b_v_r, b_v), dim=2) if buckets == kv_buckets else b_v_r

        # Alignment correlation value
        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (d_h ** -0.5)

        # mask 
        mask_value = max_neg_value(dots)


        # ?????????????????
        if not all_none(q_mask, kv_mask):
            q_mask = default(q_mask, lambda: torch.ones((b, t), device=device).bool())
            kv_mask = default(kv_mask, q_mask)
            mq, mk = bucket(buckets, q_mask), bucket(kv_buckets, kv_mask)
            expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, h))
            mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))

            mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))

            if self.n_sortcut > 0:
                mk_r = mk_r[:, 0:self.n_sortcut].reshape(-1, 1, bsz * self.n_sortcut)
                mk_r = expand_dim(mk_r, 1, buckets)
            else:
                mk_r = mk_r.reshape(bh, buckets, -1)

            mk = torch.cat((mk_r, mk), dim=2) if buckets == kv_buckets else mk_r
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            dots.masked_fill_(~mask, mask_value)
            del mask            

        # ?????????????????
        

        # attention
        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        # Attention bucketed output
        out = torch.einsum('buij,buje->buie', dots, b_v)
        out = unbucket(out)

        out = out.reshape(b, h, t, d_h)
        return out



#Capsules
class BilinearSparseRouting(nn.Module):
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
        print("HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO !!!!!!!!!!!!!!!!!!")


    def forward(self, current_pose, h_out=1, w_out=1, next_pose=None):
        
        if self.layer_type =='FC':
            print("Current pose ", current_pose.shape)
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
                # This is input shape
            
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
            if self.layer_type =='FC':
                print("Current pose1 ", current_pose.shape)
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
            if self.layer_type =='FC':
                print("Current pose2 ", current_pose.shape)
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
            
            # Now we have transformed both P_{L} and P_{L+1} to c_{L} and n_{L+1}
            temperature = self.temperature
            next_bucket_size = self.next_bucket_size
            current_bucket_size = self.current_bucket_size
            device = next_pose.device

            # Number of capsules/bucket size=number of buckets
            # Bucket the capsules
            next_buckets = next_pose.shape[1] // next_bucket_size
            current_buckets = current_pose.shape[1] // current_bucket_size
            
            # Make buckets for each of the poses
            # Current_pose = (49b,288,16) ; next_pose = (49b,m,16)
            b_next_pose = bucket(next_buckets, next_pose)
            b_current_pose = bucket(current_buckets, current_pose)
            

            if self.matrix_pose:
                # w_current = (1,288,16)
                w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
                w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
            else:
                w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
                w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim*self.pose_dim)
            
            b_w_current = bucket(current_buckets, w_current)
            bsz = b_current_pose.shape[2]

            #
            R = self.sort_net(next_pose, current_pose, topk=1)
            R = R.type_as(next_pose).to(next_pose)

            # Reorder current pose buckets
            b_current_pose_r = reorder_buckets(b_current_pose, R)
            
            '''
            Expand: Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
            Passing -1 as the size for a dimension means not changing the size of that dimension.
            b_w_current = (1,num_buckets, bucket_size, 16) --> (49b, num_buckets, bucket_size, 16 )
            Made same dimensions as b_current_pose
            '''
            b_w_current = b_w_current.expand(batch_size*h_out*w_out, -1, -1, -1)
            b_w_current_r = reorder_buckets(b_w_current, R)

            if self.n_sortcut > 0:
                # b_current_pose_r = (49b, num_buckets_next, bucket_size, 16) --> (49b, 1 , top_k * bucket_size, 16)
                b_current_pose_r = b_current_pose_r[:, 0:self.n_sortcut].reshape(batch_size*h_out*w_out, 1, -1, self.pose_dim)
                b_w_current_r = b_w_current_r[:, 0:self.n_sortcut].reshape(batch_size*h_out*w_out, 1, -1, self.pose_dim)
                
                # b_current_pose_r = (49b, next_buckets, top_k * bucket_size, 16)
                b_current_pose_r = expand_dim(b_current_pose_r, 1, next_buckets)
                b_w_current_r = expand_dim(b_w_current_r, 1, next_buckets)

            c = b_current_pose_r
            b_w_current = b_w_current_r

            b_current_pose = torch.cat((b_current_pose_r, b_current_pose), dim=2) if next_buckets == current_buckets else b_current_pose_r
            b_w_current = torch.cat((b_w_current_r, b_w_current), dim=2) if next_buckets == current_buckets else b_w_current_r
            
            # Finding scaled alignment scores between updated buckets
            dots = torch.einsum('buie,buje->buij', b_next_pose, b_current_pose) * (pose_dim ** -0.5) 
            

            # attention routing along dim=-2 (next layer buckets)
            # Dim=-1 if you wanna invert the inverted attention
            dots = dots.softmax(dim=-2) 
            b_next_pose_candidates = b_current_pose

            # Yet to multiply with N_{L} (next_w)
            b_next_pose_candidates = torch.einsum('buij,buje->buie', dots, b_next_pose_candidates)
            
            if self.matrix_pose:
                b_next_pose_candidates = b_next_pose_candidates.view(b_next_pose_candidates.shape[0], b_next_pose_candidates.shape[1], b_next_pose_candidates.shape[2], self.matrix_pose_dim, self.matrix_pose_dim)
            else:
                b_next_pose_candidates = b_next_pose_candidates.unsqueeze(3)
            
            # Multiplied with N_{j} to get final pose
            # w_next: (49b,m,4,4); b_next_pose_candidates: (49b, bucket, bucket_size, 4, 4)
            b_next_pose_candidates = torch.matmul(b_next_pose_candidates, w_next)
            next_pose_candidates = unbucket(b_next_pose_candidates)
            
            # next_pose_candidates = (b,7,7,m,16)
            next_pose_candidates = next_pose_candidates.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
            
            if self.layer_type == 'conv':
                # next_pose_candidates = (b,m,7,7,16)
                next_pose_candidates = next_pose_candidates.permute([0,3,1,2,4])
            elif self.layer_type == 'FC':
                # next_pose_candidates = (b,1,1,m,16) --> (b,1,m,16)
                next_pose_candidates = next_pose_candidates.squeeze(1)
            return next_pose_candidates



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







def squash(input_tensor):
    '''
    Squashes an input Tensor so it has a magnitude between 0-1.
       param input_tensor: a stack of capsule inputs, s_j
       return: a stack of normalized, capsule output vectors, v_j
    '''
    # same squash function as before
    squared_norm = (input_tensor ** 2).sum(dim=-2, keepdim=True)
    scale = squared_norm / (1 + squared_norm) # normalization coeff
    output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
    return output_tensor



# DYANMIC BILINEAR ROUTING
class DynamicBilinearRouting(nn.Module):
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


    def forward(self, current_pose, dots, h_out=1, w_out=1):
        # print("Using Dynamic routing with Bilinear attention")
        # current pose: (b,32,3,3,7,7,16)
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
        # print("Current img ", current_pose.shape)
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
        if dots==None:
            dots=(torch.zeros(batch_size*h_out*w_out, self.out_n_capsules, self.kernel_size*self.kernel_size*self.in_n_capsules)* (pose_dim ** -0.5)).type_as(current_pose).to(current_pose)
        
        dots = dots.softmax(dim=-2)
        next_pose = current_pose 

        # multiply with N_{L} 
        # next pose: (49b, m, 16) 
        # print(dots.shape, next_pose.shape)
        next_pose = torch.einsum('bij,bje->bie', dots, next_pose)
        
        # Squash the output to make unit length
        # print("Current shape ", next_pose.shape )
        next_pose = squash(next_pose)
        # print("Squashed shape ", next_pose.shape )



        # next_pose = (b,m,7,7,16) --> (49b,m,16)   
        next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
        
        if self.matrix_pose:
            # next_pose = (49b,m,16)  -->  (49b,m,4,4) 
            next_pose = next_pose.reshape(batch_size*h_out*w_out, self.out_n_capsules,  self.matrix_pose_dim, self.matrix_pose_dim)
        else:
            next_pose = next_pose.unsqueeze(3)

        # Found final pose of next layer by multiplying X with N_{L}
        # Multiply (49b, m, 4, 4) with (1, m, 4, 4) == (49b, m , 4, 4)
        next_pose = torch.matmul(w_next, next_pose)
        # Reshape: (b, 7, 7, m, 16)
    
        
        #****************
        
        if self.matrix_pose:
            # next_pose = (49b,m,16)
            next_pose = next_pose.view(batch_size*h_out*w_out, self.out_n_capsules,  self.pose_dim)
        else:
            next_pose = next_pose.squeeze(3)
    
        #****************
        




        ##########################################################
        # # Now we have transformed both P_{L} and P_{L+1} to c_{L} and n_{L+1}
        # temperature = self.temperature
        # next_bucket_size = self.next_bucket_size
        # current_bucket_size = self.current_bucket_size
        # device = next_pose.device

        # # Number of capsules/bucket size=number of buckets
        # # Bucket the capsules
        # next_buckets = next_pose.shape[1] // next_bucket_size
        # current_buckets = current_pose.shape[1] // current_bucket_size
        
        # # Make buckets for each of the poses
        # # Current_pose = (49b,288,16) ; next_pose = (49b,m,16)
        # b_next_pose = bucket(next_buckets, next_pose)
        # b_current_pose = bucket(current_buckets, current_pose)
        

        # if self.matrix_pose:
        #     # w_current = (1,288,16)
        #     w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.matrix_pose_dim, self.matrix_pose_dim)
        #     w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim)
        # else:
        #     w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim, self.pose_dim)
        #     w_current = w_current.view(1, self.kernel_size*self.kernel_size*self.in_n_capsules, self.pose_dim*self.pose_dim)
        
        # b_w_current = bucket(current_buckets, w_current)
        # bsz = b_current_pose.shape[2]

        # #
        # R = self.sort_net(next_pose, current_pose, topk=1)
        # R = R.type_as(next_pose).to(next_pose)

        # # Reorder current pose buckets
        # b_current_pose_r = reorder_buckets(b_current_pose, R)
        
        # '''
        # Expand: Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
        # Passing -1 as the size for a dimension means not changing the size of that dimension.
        # b_w_current = (1,num_buckets, bucket_size, 16) --> (49b, num_buckets, bucket_size, 16 )
        # Made same dimensions as b_current_pose
        # '''
        # b_w_current = b_w_current.expand(batch_size*h_out*w_out, -1, -1, -1)
        # b_w_current_r = reorder_buckets(b_w_current, R)

        # if self.n_sortcut > 0:
        #     # b_current_pose_r = (49b, num_buckets_next, bucket_size, 16) --> (49b, 1 , top_k * bucket_size, 16)
        #     b_current_pose_r = b_current_pose_r[:, 0:self.n_sortcut].reshape(batch_size*h_out*w_out, 1, -1, self.pose_dim)
        #     b_w_current_r = b_w_current_r[:, 0:self.n_sortcut].reshape(batch_size*h_out*w_out, 1, -1, self.pose_dim)
            
        #     # b_current_pose_r = (49b, next_buckets, top_k * bucket_size, 16)
        #     b_current_pose_r = expand_dim(b_current_pose_r, 1, next_buckets)
        #     b_w_current_r = expand_dim(b_w_current_r, 1, next_buckets)

        # c = b_current_pose_r
        # b_w_current = b_w_current_r

        # b_current_pose = torch.cat((b_current_pose_r, b_current_pose), dim=2) if next_buckets == current_buckets else b_current_pose_r
        # b_w_current = torch.cat((b_w_current_r, b_w_current), dim=2) if next_buckets == current_buckets else b_w_current_r
        
        ###################################################################



        # Finding scaled alignment scores between updated buckets
        # print("error: ",next_pose.shape, current_pose.shape)
        dr_agreement = torch.einsum('bje,bie->bji', next_pose, current_pose) * (pose_dim ** -0.5) 
        dots= dots + dr_agreement


        # next_pose_candidates = (b,7,7,m,16)
        next_pose = next_pose.view(batch_size, h_out, w_out, self.out_n_capsules,  self.pose_dim)
        
        if self.layer_type == 'conv':
            # next_pose = (b,m,7,7,16)
            next_pose = next_pose.permute([0,3,1,2,4])
        elif self.layer_type == 'FC':
            # next_pose = (b,1,1,m,16) --> (b,1,m,16)
            next_pose = next_pose.squeeze(1)
        
        return dots, next_pose
