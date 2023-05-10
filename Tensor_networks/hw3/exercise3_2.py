import numpy as np
import sys
import copy
import random
from itertools import permutations
from functools import reduce

class RandomFourTensorNetwork:
    def __init__(self):
        self.bond_dims = {}
        self.edges = {}
        tensor_ids = set([0, 1, 2, 3])
        self.tensors = [[], [], [], []]
        bond_id = 0

	  # Randomly create legs to be contracted
        for _ in range(random.randint(5, 8)):
            t1, t2 = random.sample(tensor_ids, 2)
            self.tensors[t1].append(bond_id)
            self.tensors[t2].append(bond_id)
            self.edges[bond_id] = (t1, t2)
            self.bond_dims[bond_id] = random.randint(5, 15)
            bond_id = bond_id + 1
        
        for t in self.tensors:
            random.shuffle(t)
        
    def contract(self, bond_id: int):
        t1_id = self.edges[bond_id][0]
        t2_id = self.edges[bond_id][1]
        tout = list(set(self.tensors[t1_id]+self.tensors[t2_id]))
        self.tensors[t1_id] = self.tensors[t2_id] = tout
        for tensor in self.tensors:
            if bond_id in tensor:
                tensor.remove(bond_id)
        self.edges.pop(bond_id)
        self.bond_dims.pop(bond_id)

    def fully_contract(self, contraction_order: list):
        for bond_id in contraction_order:
            self.contract(bond_id)

def random_tensor_network():
    return RandomFourTensorNetwork()

def cost(base_tn: RandomFourTensorNetwork, contract_order: list):
    # TODO: Subproblem (b)
    contract_cost = 0
    for i in contract_order:
        (t1,t2) = base_tn.edges[i]
        prod = len(base_tn.tensors[t1])*len(base_tn.tensors[t2])*base_tn.bond_dims[i]
        # example_tn.contract(i)
        contract_cost += prod
    return contract_cost

def max_bond_contraction(tn: RandomFourTensorNetwork):
    # TODO: Subproblem 
    s = tn.bond_dims
    return sorted(range(len(s)), key=lambda k: s[k], reverse = True)
    
def exhaustive_search_contraction(tn: RandomFourTensorNetwork):
    # TODO: Subproblem (c)
    return 0
