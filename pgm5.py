import numpy as np
#X = np.array(([2, 9],[1, 5], [3, 6]), dtype=float)
X=np.array(([1.23,1.71,2.43,15.6,127],[1.2,1.78,2.14,14.2,100],[12,24.51,27.42,22,88],[12.69,21.53,22.26,20.7,101]),dtype=float)
t = np.array(([1,0], [1,0], [0,1],[0,1]), dtype=float) #one hot representation of the target classes
X = X/np.amax(X,axis=0) # normalization process

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=100 #number of training iterations

eta=0.1 #Set learning rate

inputlayer_neurons = 5 #number of features in data set

hiddenlayer_neurons = 4 #number of hidden layers neurons

output_neurons = 2 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
#Forward Propogation
    net=np.dot(X,wh)
    net=net + bh
    hidden_act = sigmoid(net)
    outnet=np.dot(hidden_act,wout)
    outnet= outnet+ bout
    output = sigmoid(outnet)
#Backpropagation
    EO = t-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hidden_act)
    d_hiddenlayer = EH * hiddengrad

    wout += hidden_act.T.dot(d_output) *eta


#Let us test the learned network with a new sample
test=[12,24.51,27.42,22,88]; #[1.2,1.78,2.14,14.2,100]
test = test/np.amax(test,axis=0)
test_net=np.dot(test,wh)
test_net=test_net + bh
hlayer_act = sigmoid(test_net)
out_net=np.dot(hlayer_act,wout)
out_net= out_net+ bout
test_output = sigmoid(out_net)
test_class=np.round(test_output)
if(test_class[0][0]==1):
    print("Test sample is a Class-1 wine")
else:
    print("Test sample is a Class-2 wine")