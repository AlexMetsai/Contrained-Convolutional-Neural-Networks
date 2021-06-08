# Contrained-Convolutional-Neural-Networks
A Keras implementation of a Constrained Convolutional Neural Network [[1]](#1), used for image manipulation detection. 

This code was written by reading the paper, the authors do mention that they released some source code, but since I wanted to implement the paper my own, I didn't look at their code. Therefore, it is not guaranteed that this code is one hundred percent on par with what they did. But I do think it's pretty close.  

It should be noted that a fully connected layer is used instead of the extremely randomized tree mentioned in the paper. 

Tested on keras 2.1.2 with Tensorflow 1.4.0 as backend.

## References
<a id="1">[1]</a> 
Belhassen Bayar and Matthew C. Stamm  (2018). 
Constrained Convolutional Neural Networks: A New Approach Towards General Purpose Image Manipulation Detection
IEEE Transactions on Information Forensics and Security, volume 13.

<a id="2">[2]</a> 
Belhassen Bayar and Matthew C. Stamm  (2016). 
A deep learning approach to universal image manipulation detection using a new convolutional layer
2016 ACM Workshop on Information Hiding and Multimedia Security
