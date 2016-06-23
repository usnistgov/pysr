from deap import gp
import unittest, random
random.seed(0)

import tree
pset = tree.genPset(1)

class TestTreeMethods(unittest.TestCase):

    def setUp(self):
        self.tree = gp.PrimitiveTree([pset.mapping['pow'], #((C_0+x_0)*C_1)^C_2
                                      pset.mapping['mul'], #where xvec[i] = x_i 
                                      pset.mapping['add'], #and Cvec[i] = C_i
                                      pset.mapping['C'],
                                      pset.mapping['x_0'],
                                      pset.mapping['C'],
                                      pset.mapping['C']])

    def test_genPset(self):
        n = 100
        pset = tree.genPset(n)
        self.assertTrue(sum(map(lambda l: l[0:2] == 'x_', pset.mapping.keys())) == n)
        self.assertTrue(sum(map(lambda l: l == 'C', pset.mapping.keys())) == 1)

    def test_evalTree(self):
        self.assertEqual(tree.evalTree(self.tree, [2], [3,4,5]), 3200000)

if __name__ == '__main__':
    unittest.main()
