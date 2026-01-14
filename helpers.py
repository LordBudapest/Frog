import math
import torch
import numpy as np
from collections import deque
def primes(n : int) -> list[int]:
    '''
    Generate all primes less than or equal to n
    '''
    primes = [True]*(n+1)
    primes[0] = primes[1] = False
    for i in range(2, int(math.sqrt(n))+1):
        if primes[i]:
            for j in range(i*i, n+1, i):
                primes[j] = False
    return list(filter(lambda x: primes[x], range(n+1)))
primes = primes(3000000)
def cayley_graph_size(n: int) -> int:
    '''
    Compute the number of nodes in the Cayley graph of S_n
    '''
    m = n
    ans = n**3
    i = 0
    j = primes[0]
    l = len(primes)
    while j <= m:
        if m%j == 0:
            ans = (ans // (j * j)) * (j*j - 1)
            while(not m%j):
                m //= j
        i += 1
        if i >= l:
            print('increase the prime list size')
            break
        j = primes[i]
    return ans
def get_cayley_n(m: int) -> int:
    '''
    Input: number of nodes in the graph
    Output: n: get_cayley_n(m) returns n such that the Cayley graph of S_n has at least m nodes
    '''
    #Notice that, \prod{p|n} (1 - 1/p^2) >= 6/pi^2
    #Thus, n^3 * 6/pi^2 <= cayley_graph_size(n) <= n^3
    #Thus, (m * pi^2 / 6)^(1/3) >= n >= m^(1/3)
    if m <= 2:
        return 2
    low = math.ceil(m ** (1/3))
    high = math.ceil((m * (math.pi**2) / 6) ** (1/3))
    _min = cayley_graph_size(high)
    n = high
    for i in range(low, high):
        cur_min = cayley_graph_size(i)
        if cur_min >= m and cur_min < _min:
            _min = cur_min
            n = i
    return n
def get_cayley_graph(n):
    '''
    Get the edge index of the Cayley graph of S_n
    '''
    generators = np.array([
        [[1,1], [0,1]],
        [[1, n-1], [0, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [n-1, 1]] 
    ])
    ind = 1
    queue = deque([np.array([[1,0],[0,1]])])
    nodes = {(1,0,0,1): 0}

    senders = []
    receivers = [] 

    while queue:
        x = queue.pop()
        x_flat = (x[0,0], x[0,1], x[1,0], x[1,1])
        assert x_flat in nodes
        ind_x = nodes[x_flat]
        for i in range(4):
            tx = np.matmul(x, generators[i]) % n
            tx_flat = (tx[0,0], tx[0,1], tx[1,0], tx[1,1])
            if tx_flat not in nodes:
                nodes[tx_flat] = ind
                queue.append(tx)
                ind += 1
            ind_tx = nodes[tx_flat]
            senders.append(ind_x)
            receivers.append(ind_tx)
    return torch.tensor([senders, receivers], dtype=torch.long)