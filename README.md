# pysr
a pure-python symbolic regression library built on deap

### Usage
`python -m scoop pysr.py csvfile numgens popsize`
The first n+1 columns of the CSV are x_0 through x_n. The last column is y.

### Recommended parameters
I use popsize=500 and numgens very large so I can just kill it with C-c.

### Pickling
After every generation, the current population is pickled to a file called `pickle`. To restart from a picklefile, just run `pysr.py` in the same directory as `pickle`. To restart from the beginning, delete or move `pickle`.

### Plotting
Just run with numgens and popsize equal to 0 to plot out the current best fit.
