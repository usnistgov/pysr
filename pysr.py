# Mostly stolen from deap's symbreg GP example
import operator
import math
import random
import string
import inspect
import ctypes

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import numpy, numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sympy

# Define new functions
def safeDiv(left, right):
    try:
        return left / right
    except (ZeroDivisionError, OverflowError):
        return 0

def safeExp(left):
    try:
        return numpy.exp(left)
    except (ZeroDivisionError, OverflowError):
        return 0
        
def safeIntMod(left, right):
    try:
        return int(left % right)
    except (ZeroDivisionError, OverflowError):
        return 0

def safePow(left, right):
    try:
        return numpy.power(left,right)
    except (OverflowError):
        return 0

# Operation cost table
OperationCost = {
    operator.add.__name__: 1,
    operator.sub.__name__: 1,
    operator.xor.__name__: 1,
    safePow.__name__: 3,
    safeIntMod.__name__: 4,
    operator.mul.__name__: 2,
    safeDiv.__name__: 4,
    operator.neg.__name__: 1,
    math.sin.__name__: 1,
    safeExp.__name__: 5
}

Ninputs = 2
Nconstants = 4

pset = gp.PrimitiveSet("MAIN", Ninputs + Nconstants)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeExp, 1)
pset.addPrimitive(safeDiv, 2)
pset.addTerminal(-1)
pset.addTerminal(-2)
pset.addTerminal(-3)
pset.addTerminal(1)
pset.addTerminal(2)
pset.addTerminal(3)
pset.addTerminal(-10)
pset.addPrimitive(safePow, 2)
#pset.addPrimitive(math.sin, 1)

sympy_namespace = {}
for i in range(Ninputs):
    pset.renameArguments(**{'ARG'+str(i): 'x_'+str(i)})
    sympy_namespace['x_'+str(i)] = sympy.Symbol('x_'+str(i))
for i in range(Nconstants):
    pset.renameArguments(**{'ARG'+str(Ninputs + i): 'c_'+str(i)})
    sympy_namespace['c_'+str(i)] = sympy.Symbol('c_'+str(i))
    
def Sub(a,b):
    return sympy.Add(a,-b)

def Div(a,b):
    return a*sympy.Pow(b,-1)
    
sympy_namespace['mul'] = sympy.Mul
sympy_namespace['add'] = sympy.Add
sympy_namespace['sub'] = Sub
sympy_namespace['safePow'] = sympy.Pow
sympy_namespace['safeExp'] = sympy.exp
sympy_namespace['safeDiv'] = Div

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalParams(params, func, xvecs, yvec):

    sqerrors = ((func(*(list(IO[1::]) + list(params))) - IO[0])**2 for IO in zip(yvec, *xvecs))

    try:
        errsum = math.fsum(sqerrors)
    except OverflowError:
        errsum = 1e10
    
    #print params, errsum
    if math.isnan(errsum):
        errsum = len(yvec) * 1e+40
    return errsum

def evalInvSqrt(individual, xvecs, yvec):
    # Compute the difficulty of individual
    difficulty = 1
    for _ in individual:
        if isinstance(_, gp.Primitive):
            difficulty += OperationCost[_.name]

    #print str(individual)
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    nb_args = len(inspect.getargspec(func)[0])
    
    # Optimize the "constants"
    res = optimize.fmin(evalParams,
                        [1 for _ in range(nb_args - Ninputs)],
                        args=(func, xvecs, yvec),
                        full_output=True,
                        disp=False)
    errsum = res[1]
    individual.optim_params = res[0]
    individual.difficulty = difficulty
    individual.errsum = errsum
    print errsum / len(yvec), difficulty
    return errsum / len(yvec), difficulty

def main(pool, xvecs, yvec):
    #random.seed(31415926535)
    
    toolbox.register("evaluate", evalInvSqrt, xvecs=xvecs, yvec=yvec)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=10)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(20)#ParetoFront()
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)
    
    #toolbox.register("map", pool.map)

    try:
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 3, stats=mstats,
                                       halloffame=hof, verbose=True)
    except KeyboardInterrupt:
        pass

    for idx, ind in enumerate(hof):
        s = sympy.simplify(sympy.sympify(str(ind), locals = sympy_namespace))
        func = toolbox.compile(expr=ind)
        yfit = list(func(*(list(IO[1::]) + list(ind.optim_params))) for IO in zip(yvec, *xvecs))
        plt.plot([0.9,2.0],[0.9,2.0])
        plt.plot(np.array(yvec), np.array(yfit),'o')
        plt.title('$' + sympy.latex(s) + '$')
        plt.xlim(0.9, 2.0)
        plt.ylim(0.9, 2.0)
        plt.show()
        print idx, s, ind.errsum/len(yvec), ind.difficulty
    
    nodes, edges, labels = gp.graph(hof[0])
    
    ### Graphviz Section ###
    import pygraphviz as pgv

    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

    return pop, log, hof

if __name__ == "__main__":
    
    #import multiprocessing 
    #pool = multiprocessing.Pool()
    
    #xvec = numpy.linspace(-1, 1)
    #yvec = numpy.sin(xvec)
    #yvec = 1/(1+numpy.exp(-10*xvec))
    
    # read data
    import pandas
    pairs = pandas.read_csv('populated_descriptors.csv')
    xvecs = []

    for descriptor in ['Tc', 'V_McGowan']:#:,'NumValenceElectrons',  'Chi0n', 'Chi1n', 'Chi1v', 'Chi1','Kappa1']:
        diff = np.array(pairs[descriptor + '-A'] / pairs[descriptor + '-B'])
        xvecs.append(diff/np.max(diff))
    yvec = pairs['gammaT']
    
    main(None, xvecs, yvec)