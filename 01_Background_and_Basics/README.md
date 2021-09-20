## Background and Basics
Assignment 1 of EVA7 Phase 1

#### Q. What are Channels and Kernels (according to EVA)?
> Channel is a container for similar kind of information or same features. It is also a collection of all neurons that contain information about a specific feature.

> Kernel is a feature extractor, filter and usually a 3x3 matrix that is convolved on an image to output a channel.

#### Q. Why should we (nearly) always use 3x3 kernels?
> Suppose we have an image with 7x7 pixels and if we convolve on this image with 3x3 kernel then we will get 5x5 channel and if we convolve on this with 3x3 kernel again, we will get 3x3 channel and and if we convolve on this with 3x3 kernel again, we will get 1x1 channel. But if we are convolving the 7x7 image with 7x7 kernel, we will get 1x1 channel. So, essentially three 3x3 kernels are equivalent to one 7x7 kernel. And since convolution is nothing but muliplications, using three 3x3 kernels will be lighter. And stacking smaller layers is always lighter than having bigger ones. It also tends to improve the result because there will be more layers and deeper networks. But using small layers will have it's disadvantages.

#### Q. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
> We would have to perfom the convolution 99 times to reach close to 1x1 from 199x199. You can find all the layer outputs [here](q3_output.txt)

#### Q. How are kernels initialized?
> Kernels are usually initialized randomly and then we will use some optimization techniques to optimize the values of the kernel so that it can perform better with the given task. Few initialization strategies would be:
> - Setting all values to either 1 or 0 or some constant.
> - Sample from a distribution.

#### Q. What happens during the training of a DNN?
> The neural network tries to extract features that will be useful for solving the given problem. Let's take the task of recognizing images, the neural network would first extract the edges and gradients, then textures and patters, then parts of objects and then objects. All these features will then help neural network to predict which image is which.
