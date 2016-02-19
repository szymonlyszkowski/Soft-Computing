# Soft-Computing

This project contains solutions for soft-computing subject labs.

##Task 0 Linear Neuron

Main purpose of this task was artificial neuron model implementation. Simple neuron is activated by identity function and is it trained using **Delta 
principle**. Task should prove that neural network is trained correctly for equation set with exactly one solution.

##Task 1 MADALINE Network

Objective of this exercise was to implement Multiple Adaptive Linear (MADALINE) network which is able to recognize letter patterns. Letters are implemented 
as arrays of 0 and 1 which corresponds respectively to white and black pixel. Each neuron has fixed weights which are **not** trained during network 
execution. Result of MADALINE network execution are neuron outputs. Neuron with highest output value which can be <0,1> is chosen as recognized letter. 
Package training_set_letters/letters.py contains patterns for letters wchich can be recognized.

##Task 2 Kohonen Network

Objective of this exercise was to implement Kohonen network which will be able to compress images. Implementation handles gray-scale images. Images should 
be squares and frame used to scan image have dimensions: 4x4, 8x8, 16x16. Such image array is encoded to list of tuples where elements are: (frame id, 
neuron_decoding). Using such object to as input to neural network it is possible to obtain compressed image. 

##Task 3 Multilayer Perceptron

Objective of this exercise was to implement Perceptron consisting of 3 layers: input(copying), hidden, output. Method used to train network was *backward 
propagation of error*. Perceptron is supposed to recognize 4 vectors: [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1].


