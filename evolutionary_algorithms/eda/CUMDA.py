import numpy

from operator import attrgetter
#from deap import base
#from deap import creator

#creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
#creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMin)

#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", numpy.ndarray, fitness=creator.FitnessMax)

class CUMDA(object):
    """
    Implementation of Continuous Univariate Marginal Distribution Algorithm (cUMDA)
    """
    def __init__(self, N, sigma, mu, lambda_):
        self.dim = N
        self.sigma = numpy.array(sigma)
        self.lambda_ = lambda_
        self.avg=numpy.array(sigma)
        self.mu = mu

    def generate(self, ind_init):
        # Generate lambda_ individuals and put them into the provided class
        arz = self.sigma * numpy.random.randn(self.lambda_, self.dim)+self.avg
        return list(map(ind_init, arz))

    def update(self, population):
        # Sort individuals so the best is first
        sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)

        # Compute the average of the mu best individuals
        z = sorted_pop[:self.mu]
        self.avg = numpy.mean(z, axis=0)

        # Adjust variance of the distribution
        self.sigma = numpy.sqrt(numpy.sum(numpy.sum((z - self.avg)**2, axis=1)) / (self.mu*self.dim))
