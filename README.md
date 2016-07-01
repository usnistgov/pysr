# pysr
a pure-python symbolic regression library built on deap

### Installation
We use conda, and so should you.
1. Create new environment with: `conda env create -n ENVIRONMENT_NAME -f environment.yml` where `ENVIRONMENT_NAME` is the name you want for your new environment
2. Activate with `source activate ENVIRONMENT_NAME` on linux/OSX or `activate ENVIRONENT_NAME` on windows
3. Deactivate with `source deactivate`

### Usage
`python -m scoop pysr.py csvfile numgens popsize`

The first n+1 columns of the CSV are x_0 through x_n. The last column is y.

### Recommended parameters
I use popsize=500 and numgens very large so I can just kill it with C-c.

### Pickling
After every generation, the current population is pickled to a file called `pickle`. To restart from a picklefile, just run `pysr.py` in the same directory as `pickle`. To restart from the beginning, delete or move `pickle`.

### Plotting
Just run with numgens and popsize equal to 0 to plot out the current best fit.
