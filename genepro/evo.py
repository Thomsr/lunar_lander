from typing import Callable

import numpy as np
from numpy.random import random as randu
from numpy.random import randint as randi
from numpy.random import choice as randc
from numpy.random import shuffle
import time, inspect
from copy import deepcopy
from joblib.parallel import Parallel, delayed

from genepro.node import Node
from genepro.variation import *
from genepro.selection import tournament_selection

class Evolution:
  """
  Class concerning the overall evolution process.

  Parameters
  ----------
  fitness_function : function
    the function used to evaluate the quality of evolving trees, should take a Node and return a float; higher fitness is better

  internal_nodes : list
    list of Node objects to be used as internal nodes for the trees (e.g., [Plus(), Minus(), ...])

  leaf_nodes : list
    list of Node objects to be used as leaf nodes for the trees (e.g., [Feature(0), Feature(1), Constant(), ...])

  pop_size : int, optional
    the population size (default is 256)

  init_max_depth : int, optional
    the maximal depth trees can have at initialization (default is 4)

  max_tree_size : int, optional
    the maximal number of nodes trees can have during the entire evolution (default is 64)

  crossovers : list, optional
    list of dictionaries that contain: "fun": crossover functions to be called, "rate": rate of applying crossover, "kwargs" (optional): kwargs for the chosen crossover function (default is [{"fun":subtree_crossover, "rate": 0.75}])

  mutations : list, optional
    similar to `crossovers`, but for mutation (default is [{"fun":subtree_mutation, "rate": 0.75}])

  coeff_opts : list, optional
    similar to `crossovers`, but for coefficient optimization (default is [{"fun":coeff_mutation, "rate": 1.0}])
  
  selection : dict, optional
    dictionary that contains: "fun": function to be used to select promising parents, "kwargs": kwargs for the chosen selection function (default is {"fun":tournament_selection,"kwargs":{"tournament_size":4}})

  max_evals : int, optional
    termination criterion based on a maximum number of fitness function evaluations being reached (default is None)

  max_gens : int, optional
    termination criterion based on a maximum number of generations being reached (default is 100)

  max_time: int, optional
    termination criterion based on a maximum runtime being reached (default is None)

  n_jobs : int, optional
    number of jobs to use for parallelism (default is 4)

  verbose : bool, optional
    whether to log information during the evolution (default is False)

  elite_archive_size : int, optional
    size of the elite archive to maintain (default is 0, no archive); set > 0 to enable elitist selection

  Attributes
  ----------
  All of the parameters, plus the following:

  population : list
    list of Node objects that are the root of the trees being evolved

  num_gens : int
    number of generations

  num_evals : int
    number of evaluations

  start_time : time
    start time

  elapsed_time : time
    elapsed time

  best_of_gens : list
    list containing the best-found tree in each generation; note that the entry at index 0 is the best at initialization

  elite_archive : list
    list of elite individuals maintained across generations; sorted by fitness (descending)
  """
  def __init__(self,
    # required settings
    fitness_function : Callable[[Node], float],
    internal_nodes : list,
    leaf_nodes : list,
    # optional evolution settings
    n_trees : int,
    pop_size : int=256,
    init_max_depth : int=4,
    max_tree_size : int=64,
    crossovers : list=[{"fun":subtree_crossover, "rate": 0.5}],
    mutations : list= [{"fun":subtree_mutation, "rate": 0.5}],
    coeff_opts : list = [{"fun":coeff_mutation, "rate": 0.5}],
    selection : dict={"fun":tournament_selection,"kwargs":{"tournament_size":8}},
    # termination criteria
    max_evals : int=None,
    max_gens : int=100,
    max_time : int=None,
    # other
    n_jobs : int=4,
    verbose : bool=False,
    elite_archive_size : int=8,
    ):

    # set parameters as attributes
    _, _, _, values = inspect.getargvalues(inspect.currentframe())
    values.pop('self')
    for arg, val in values.items():
      setattr(self, arg, val)

    # fill-in empty kwargs if absent in crossovers, mutations, coeff_opts
    for variation_list in [crossovers, mutations, coeff_opts]:
      for i in range(len(variation_list)):
        if "kwargs" not in variation_list[i]:
          variation_list[i]["kwargs"] = dict()
    # same for selection
    if "kwargs" not in selection:
      selection["kwargs"] = dict()

    # initialize some state variables
    self.population = list()
    self.num_gens = 0
    self.num_evals = 0
    self.start_time, self.elapsed_time = 0, 0
    self.best_of_gens = list()
    self.elite_archive = list()  # archive for elite individuals

    self.memory = None


  def _update_elite_archive(self, candidates):
    """
    Updates the elite archive with promising candidates.
    
    Parameters
    ----------
    candidates : list
      list of individuals with fitness attribute to consider for the archive
    """
    if self.elite_archive_size <= 0:
      return
    
    # Add new candidates to archive
    for individual in candidates:
      self.elite_archive.append(deepcopy(individual))
    
    # Sort by fitness (descending - higher is better) and keep top elite_archive_size
    self.elite_archive.sort(key=lambda x: x.fitness, reverse=True)
    self.elite_archive = self.elite_archive[:self.elite_archive_size]

  def _get_selection_pool(self):
    """
    Creates a selection pool that includes both population and elite archive.
    
    Returns
    -------
    list
      combined population and elite archive
    """
    if self.elite_archive_size <= 0:
      return self.population
    
    # Combine population and elite archive for selection
    combined_pool = self.population + self.elite_archive
    return combined_pool


  def _must_terminate(self) -> bool:
    """
    Determines whether a termination criterion has been reached

    Returns
    -------
    bool
      True if a termination criterion is met, else False
    """
    self.elapsed_time = time.time() - self.start_time
    if self.max_time and self.elapsed_time >= self.max_time:
      return True
    elif self.max_evals and self.num_evals >= self.max_evals:
      return True
    elif self.max_gens and self.num_gens >= self.max_gens:
      return True
    return False

  def _initialize_population(self):
    """
    Generates a random initial population and evaluates it
    """
    # initialize the population
    self.population = Parallel(n_jobs=self.n_jobs)(
        delayed(generate_random_multitree)(self.n_trees, 
          self.internal_nodes, self.leaf_nodes, max_depth=self.init_max_depth )
        for _ in range(self.pop_size))

    for count, individual in enumerate(self.population):
      individual.get_readable_repr()

    # evaluate the trees and store their fitness
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in self.population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      self.population[i].fitness = fitnesses[i]
    # store eval cost
    self.num_evals += self.pop_size
    # store best at initialization
    best = self.population[np.argmax([t.fitness for t in self.population])]
    self.best_of_gens.append(deepcopy(best))
    # update elite archive with initial population
    self._update_elite_archive(self.population)

  def _perform_generation(self):
    """
    Performs one generation, which consists of parent selection, offspring generation, and fitness evaluation
    """
    # select promising parents (from population only)
    selection_pool = self.population
    sel_fun = self.selection["fun"]
    parents = sel_fun(selection_pool, self.pop_size, **self.selection["kwargs"])
    # generate offspring
    offspring_population = Parallel(n_jobs=self.n_jobs)(delayed(generate_offspring)
      (t, self.crossovers, self.mutations, self.coeff_opts, 
      parents, self.internal_nodes, self.leaf_nodes,
      constraints={"max_tree_size": self.max_tree_size}) 
      for t in parents)

    # evaluate each offspring and store its fitness 
    fitnesses = Parallel(n_jobs=self.n_jobs)(delayed(self.fitness_function)(t) for t in offspring_population)
    fitnesses = list(map(list, zip(*fitnesses)))
    memories = fitnesses[1]
    memory = memories[0]
    for m in range(1,len(memories)):
      memory += memories[m]

    self.memory = memory + self.memory

    fitnesses = fitnesses[0]

    for i in range(self.pop_size):
      offspring_population[i].fitness = fitnesses[i]
    # store cost
    self.num_evals += self.pop_size
    
    # Apply elitism: merge offspring with elite archive and keep best pop_size
    if self.elite_archive_size > 0 and len(self.elite_archive) > 0:
      combined = offspring_population + self.elite_archive
      combined.sort(key=lambda x: x.fitness, reverse=True)
      self.population = combined[:self.pop_size]
    else:
      self.population = offspring_population
    
    # update elite archive with the new population
    self._update_elite_archive(self.population)
    
    # update info
    self.num_gens += 1
    best = self.population[np.argmax([t.fitness for t in self.population])]
    self.best_of_gens.append(deepcopy(best))

  def evolve(self):
    """
    Runs the evolution until a termination criterion is met;
    first, a random population is initialized, second the generational loop is started:
    every generation, promising parents are selected, offspring are generated from those parents, 
    and the offspring population is used to form the population for the next generation
    """
    # set the start time
    self.start_time = time.time()

    self._initialize_population()

    # generational loop
    while not self._must_terminate():
      # perform one generation
      self._perform_generation()
      # log info
      if self.verbose:
        print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {}".format(
            self.num_gens, self.best_of_gens[-1].fitness, len(self.best_of_gens[-1])
            ))
