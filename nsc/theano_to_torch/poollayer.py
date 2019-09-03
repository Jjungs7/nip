#-*- coding: UTF-8 -*-
import torch
import numpy

class LastPoolLayer(object):
	def __init__(self, input):
        self.input = input
        self.output = input[-1]
        self.params = []

    def save(self, prefix):
        pass

class MeanPoolLayer(object):
    def __init__(self, input, ll):
        self.input = input
		self.output = torch.sum(input, dtype=torch.float32) / torch.unsqueeze(ll, 1)
        #self.output = T.sum(input, axis=0, acc_dtype='float32') / ll.dimshuffle(0, 'x')
        self.params = []

    def save(self, prefix):
        pass


class MaxPoolLayer(object):
    def __init__(self, input):
        self.input = input
		self.output = torch.max(input, axis = 0)
		#self.output = T.max(input, axis = 0)
        self.params = []

    def save(self, prefix):
        pass

class Dropout(object):
    def __init__(self, input, rate, istrain):
        self.input = input
        rate = numpy.float32(rate)
        srng = numpy.random.RandomState()
        #srng = T.shared_randomstreams.RandomStreams()
        mask = srng.binomial(n=1, p=numpy.float32(1-rate), size=input.shape)
        #self.output = T.switch(istrain, mask*self.input, self.input*numpy.float32(1-rate))
        self.output = torch.where(istrain, mask*self.input, self.input*numpy.float32(1-rate))
        self.params = []

    def save(self, prefix):
        pass
