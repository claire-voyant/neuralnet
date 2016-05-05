########################
# SciPy and NumPy#######
# MatPlotLib############
########################

from mlxtend.evaluate import plot_decision_regions
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
matplotlib.use('Agg')

########################
# Perceptron Class######
########################

class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for x in range(self.epochs):
            errors = 0
            for i,target in zip(X,y):
                update = self.eta * (target - self.guess(i))
                self.w_[1:] += update * i
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def nn_in(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def guess(self,X):
        return np.where(self.nn_in(X) >= 0.0,1,-1)

class GradientDescent(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.nn_in(X)
            errors = (y - output)

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def nn_in(self,X):
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def on(self,X):
        return self.nn_in(X)

    def guess(self,X):
        return np.where(self.on(X) >= 0.0,1,-1)

########################
# set up for perceptron#
########################

########################
# artificially create###
# numerics for input####
########################

_x = 1
_o = -1
_b = 0

########################
# Split amount for #####
# data training#########
########################

split = 66.0

########################
# Open data file and####
# format into set#######

_f = 'input2'
_file = open(_f,'r')

########################
# Create our sets to####
# train and test########
# with a total set######
########################

total = list()
test = list()
train = list()

########################
# Iterator tracking#####
# sort of a hack########
########################

k = 0

########################
# Split the data up#####
# into their own lists##
########################

for line in _file:
    total.append(line.strip().split(','))
    k += 1
_file.close()

########################
# Calculate our split###
# amount of the dataset#
########################

num_test = int(k*(split/100.0))

########################
# Create our training###
# and testing lists#####
# using the split#######
########################

k = 0
for each in total:
    if k <= num_test:
        train.append(each)
    else:
        test.append(each)
    k += 1

########################
# Adjust the class to###
# be numeric 1 is#######
# positive and -1 is####
# negative##############

for x in range(0,len(train)):
    _set = -1
    if train[x][-1] == 'positive':
        _set = 1
    train[x][-1] = _set

for x in range(0,len(test)):
    _set = -1
    if test[x][-1] == 'positive':
        _set = 1
    test[x][-1] = _set

########################
# Create training input#
# and testing input#####
# by separating the#####
# class from data#######
########################

_X = list()
_y = list()

for x in train:
    temp = x.pop()
    _y.append(temp)
    _X.append(x)

__X = list()
__y = list()

for x in test:
    temp = x.pop()
    __y.append(temp)
    __X.append(x)

#######################
# Convert characters###
# to numerics in data##
#######################

numberTrainData = list()
numberTestData  = list()

for each in _X:
    temp = list()
    for i in each:
        if i == 'x':
            temp.append(_x)
        elif i == 'o':
            temp.append(_o)
        else:
            temp.append(_b)
    numberTrainData.append(temp)

for each in __X:
    temp = list()
    for i in each:
        if i == 'x':
            temp.append(_x)
        elif i == 'o':
            temp.append(_o)
        else:
            temp.append(_b)
    numberTestData.append(temp)

#print len(numberTrainData)
#print len(numberTestData)

#######################
# Write out formatted##
# data to an output####
# to debug#############
#######################

output = open('output.txt','w')

output.write("Train Data\n")
for each in numberTrainData:
    output.write(str(each))
    output.write("\n")

output.write("Train Class\n")
for each in _y:
    output.write(str(each))
    output.write("\n")

output.write("Test Data\n")
for each in numberTestData:
    output.write(str(each))
    output.write("\n")

output.write("Test Class\n")
for each in __y:
    output.write(str(each))
    output.write("\n")


trainData = np.array(numberTrainData)
trainClass = np.array(_y)

testData = np.array(numberTestData)
testClass = np.array(__y)

#print "shape: ", trainData.shape
#print "shape.length: " , len(trainData.shape)
#print "ele_1: " ,trainData.shape[1]

#print trainData
#print trainClass

_t = list()
_val = 0.0
_cur = 0

neural_net = Perceptron(epochs=100,eta=0.01).train(trainData,trainClass)

for each in testData:
    ret = neural_net.guess(each)
    _t.append(ret)
    if ret == testClass[_cur]:
        _val += 1
    _cur += 1

print "Perceptron: " ,
print "correct classification: ", _val/float(len(testClass))
output.write("Weights")
output.write(str(neural_net.w_))
output.close()
plt.title('Neural Network: Perceptron')
plot(range(1,len(neural_net.errors_)+1),neural_net.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.savefig('./figures/basic_perceptron.png')

neural_net = GradientDescent(epochs=10000,eta=0.00001).train(trainData,trainClass)

_t = list()
_val = 0.0
_cur = 0

for each in testData:
    ret = neural_net.guess(each)
    _t.append(ret)
    if ret == testClass[_cur]:
        _val += 1
    _cur += 1

print "Gradient Decent: ",
print "correct classification: " , _val/float(len(testClass))
plt.plot(range(1,len(neural_net.cost_)+1),(neural_net.cost_),marker='o')
plt.xlabel('Iterations')
plt.ylabel('SSE')
plt.title('Neural Network: Gradient Descent')
plt.savefig('./figures/gradient_descent.png')


