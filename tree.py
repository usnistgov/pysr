from deap import gp
import operator, math
import numpy as np

def genPset(n):
    """Return a pset containing all the functions and primitives used in the tree."""
    pset = gp.PrimitiveSet('MATH', arity=n)
    for i in range(0, n):
        eval('pset.renameArguments(ARG'+str(i)+'=\'x_'+str(i)+'\')')
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    def div(x,y):
        return operator.truediv(x,y)
    div.__name__ = 'div'
    pset.addPrimitive(div, 2)
    pset.addPrimitive(operator.pow, 2)
    pset.addPrimitive(math.log, 2)
    pset.addTerminal('C') #constants that get fitted
    #vars are automatically added as terminals
    assert(sum(map(lambda l: l[0:2] == 'x_', pset.mapping.keys())) == n)
    assert(sum(map(lambda l: l == 'C', pset.mapping.keys())) == 1)
    return pset

def evalTree(tree, xvec, Cvec):
    """
    Evaluate the expression tree at given input and constant vectors.

    Input:
    tree -- gp.PrimitiveTree to be evaluated
    xvec -- vector of x_0 through x_n that gives values to each x_i in the tree
    Cvec -- vector of values for the constant C numbered by depth-first order
            (i.e. the zeroth constant in the depth-first representation of the tree
             is Cvec[0], etc.)

    Output:
    Value of the tree evaluated at xvec with constant values Cvec
    """
    assert(np.array(xvec).shape[0]>1)
    assert(np.array(Cvec).shape[0]>1)

    #replace deap objects with strings and ints
    C_index = 0
    for i in range(0, len(tree)):
        if tree[i].name == 'C':
            tree[i] = float(Cvec[C_index])
        if tree[i].name[0:2] == 'x_':
            tree[i] = float(xvec[int(x[2:])])
        else:
            tree[i] = tree[i].name

    #convert the tree from depth-first (deap) to breadth-first (for recursive eval)
    class node:
        def __init__(self, tree):
            self.val = tree[0]
            self.left = node(tree[tree.searchSubtree(1)])
            self.right = node(tree[tree.searchSubtree(tree.searchSubtree(1).end)])
    root = node(tree)

    #recursively evaluate the tree
    def eval(node, xvecs):
        if isinstance(node.val, float):
            return node.val
        elif isinstance(node.val, str):
            return {
                'add' : eval(node.left) + eval(node.right),
                'sub' : eval(node.left) - eval(node.right),
                'mul' : eval(node.left) * eval(node.right),
                'div' : eval(node.left) / eval(node.right),
                'pow' : eval(node.left) ** eval(node.right),
                'log' : math.log(eval(node.left), eval(node.right))
                }[node.val]
        else:
            raise TypeError

    return eval(root)
