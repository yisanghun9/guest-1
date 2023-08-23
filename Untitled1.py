#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import time


# In[2]:


class neuralNetwork:
    

    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate, decay_factor, decay_steps):

        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.wih = numpy.array(self.wih, ndmin=2, dtype='complex128')
        self.wih += 1j * numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))

        self.whh = numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.whh = numpy.array(self.whh, ndmin=2, dtype='complex128')
        self.whh += 1j * numpy.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))

        self.who = numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))
        self.who = numpy.array(self.who, ndmin=2, dtype='complex128')
        self.who += 1j * numpy.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        self.lr = learningrate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        
        self.activation_function = lambda x: 1 / (1 + numpy.exp(-x))                      
        
    def custom_learning_rate(self, initial_lr, epoch):
        decayed_lr = initial_lr * (self.decay_factor**(epoch//self.decay_steps))
        return decayed_lr
        
    def train(self, inputs_list, targets_list, epoch):
        inputs = numpy.array(inputs_list, dtype=complex, ndmin=2).T
        targets = numpy.array(targets_list, dtype=complex, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih, inputs)
        hidden1_act = self.activation_function(numpy.abs(hidden1_inputs))
        hidden1_outputs_real = hidden1_act * numpy.cos(numpy.angle(hidden1_inputs))
        hidden1_outputs_imag = hidden1_act * numpy.sin(numpy.angle(hidden1_inputs))
        hidden1_outputs = hidden1_outputs_real + 1j * hidden1_outputs_imag

        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_act = self.activation_function(numpy.abs(hidden2_inputs))
        hidden2_outputs_real = hidden2_act * numpy.cos(numpy.angle(hidden2_inputs))
        hidden2_outputs_imag = hidden2_act * numpy.sin(numpy.angle(hidden2_inputs))
        hidden2_outputs = hidden2_outputs_real + 1j * hidden2_outputs_imag

        final_inputs = numpy.dot(self.who, hidden2_outputs)
        final_outputs_real = self.activation_function(final_inputs.real)
        final_outputs_imag = self.activation_function((-1j * final_inputs.imag).real)
        final_outputs = final_outputs_real #+ 1j * final_outputs_imag

        output_errors = targets - final_outputs
        hidden2_errors = numpy.dot(self.who.T, output_errors)
        hidden1_errors = numpy.dot(self.whh.T, hidden2_errors)
        
        self.lr = self.custom_learning_rate(self.lr, epoch)
              
        self.who += self.lr * numpy.dot(output_errors * (1.0 - numpy.square(final_outputs)), numpy.conj(hidden2_outputs).T)
        self.whh += self.lr * numpy.dot(hidden2_errors * (1.0 - numpy.square(hidden2_outputs)), numpy.conj(hidden1_outputs).T)
        self.wih += self.lr * numpy.dot(hidden1_errors * (1.0 - numpy.square(hidden1_outputs)), numpy.conj(inputs).T)

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, dtype=complex, ndmin=2).T

        hidden1_inputs = numpy.dot(self.wih, inputs)
        hidden1_act = self.activation_function(numpy.abs(hidden1_inputs))
        hidden1_outputs_real = hidden1_act * numpy.cos(numpy.angle(hidden1_inputs))
        hidden1_outputs_imag = hidden1_act * numpy.sin(numpy.angle(hidden1_inputs))
        hidden1_outputs = hidden1_outputs_real + 1j * hidden1_outputs_imag

        hidden2_inputs = numpy.dot(self.whh, hidden1_outputs)
        hidden2_act = self.activation_function(numpy.abs(hidden2_inputs))
        hidden2_outputs_real = hidden2_act * numpy.cos(numpy.angle(hidden2_inputs))
        hidden2_outputs_imag = hidden2_act * numpy.sin(numpy.angle(hidden2_inputs))
        hidden2_outputs = hidden2_outputs_real + 1j * hidden2_outputs_imag

        final_inputs = numpy.dot(self.who, hidden2_outputs)
        final_outputs_real = self.activation_function(final_inputs.real)
        final_outputs_imag = self.activation_function((-1j * final_inputs.imag).real)
        final_outputs = final_outputs_real #+ 1j * final_outputs_imag
        
        return final_outputs_real


# In[6]:


input_nodes = 784
hidden_nodes1 = 200
hidden_nodes2 = 100
output_nodes = 10

learning_rate = 0.001

decay_factor = 0.1
decay_steps = 100

n = neuralNetwork(input_nodes, hidden_nodes1, hidden_nodes2, output_nodes, learning_rate,decay_factor, decay_steps)


def find_learning_rate(neural_net, training_data_list):
    initial_lr = 1e-6
    lr_multiplier = 1.1
        
    best_loss = float('inf')
    best_lr = initial_lr
        
    for lr in numpy.geomspace(initial_lr, 1.0, num=10):
        neural_net.lr = lr
        total_loss = 0
            
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])/255*0.99)+0.01
            targets = numpy.zeros(output_nodes)+0.01
            targets[int(all_values[0])] = 0.99
            outputs = n.query(inputs)
            loss = numpy. mean(numpy.square(targets-outputs))
            total_loss += loss
            
        average_loss = total_loss / len(training_data_list)
            
        if average_loss < best_loss:
            best_loss = average_loss
            best_lr = lr
                
        return best_lr



training_data_file = open("mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

print("Start Training..")

epochs = 50

t_s = time.time()
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets,e)
        
    print("epochs:", e+1, "/", epochs)
t_f = time.time()

print("timecost = {0:0.2f} sec".format(t_f-t_s))

test_data_file = open("mnist_train_100.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

optimal_lr = find_learning_rate(n,training_data_list)
print("Optimal Learning Rate:", optimal_lr)



for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
    
scorecard_array = numpy.asarray(scorecard)
print("Performance =", scorecard_array.sum() / scorecard_array.size)


# In[ ]:




