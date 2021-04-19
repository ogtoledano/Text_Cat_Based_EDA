import numpy
from operator import attrgetter


class EMNA(object):
    """
    This implementation was taking of a paper as below:
    
    Fabien Teytaud and Olivier Teytaud. 2009.
    Why one must use reweighting in estimation of distribution algorithms. 
    In Proceedings of the 11th Annual conference on Genetic and
    evolutionary computation (GECCO '09). ACM, New York, NY, USA, 453-460.
    
    In this you can see Estimation of Multivariate Normal Algorithm (EMNA) 
    as described by Algorithm 1. 
    """
    def __init__(self, centroid, sigma, mu, lambda_):
        self.dim = len(centroid)
        self.centroid = numpy.array(centroid)
        self.sigma = numpy.array(sigma)
        self.lambda_ = lambda_
        self.mu = mu

    def generate(self, ind_init):
        # Generate lambda_ individuals and put them into the provided class
        arz = self.centroid + self.sigma * numpy.random.randn(self.lambda_, self.dim)
        return list(map(ind_init, arz))

    def update(self, population):
        # Sort individuals so the best is first
        sorted_pop = sorted(population, key=attrgetter("fitness"), reverse=True)

        # Compute the average of the mu best individuals
        z = sorted_pop[:self.mu] - self.centroid
        avg = numpy.mean(z, axis=0)

        # Adjust variance of the distribution
        self.sigma = numpy.sqrt(numpy.sum(numpy.sum((z - avg)**2, axis=1)) / (self.mu*self.dim))
        self.centroid = self.centroid + avg

        
        