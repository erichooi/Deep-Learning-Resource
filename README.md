# Deep Learning Study Material

When I start to learn about deep learning, it was hard for me to find the infomation about different algorithms and tutorials. So, I decided to gather the information that I found during the learning of the deep learning and share it here. Besides, I also want to add code examples that I found online into the deep learning model because it is better that we can learn by understanding the coding itself

## Fundamental Tutorials

These are the tutorials that I think are fundamental for those who want to start learning deep learning. So, if you do not know what to start, feel free to start by finishing the tutorials and courses here.

* [Deep Learning by Andrew Ng](https://www.coursera.org/specializations/deep-learning) - this course by Andrew Ng really taught me the fundamental mathematic concept behind different deep learning model.
* [Deep Learning Book](http://www.deeplearningbook.org/) - this is probably one of the best book in explaining the concept of deep learning algorithm that available for free online.
* [Neural Network and Deep Learning](http://neuralnetworksanddeeplearning.com/) - a really great book that teaches about deep learning by [@michael_nielsen](https://twitter.com/michael_nielsen). It also contains **code** that demonstrate some of the deep learning concept.

## Deep Learning Model

This part will contain the resource for different deep learning model and useful concepts that are used in training a deep learning model

### Word2Vec

#### Skip-Gram

Concept:

* [Chris McCormick's post](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) - This post explained the neural network architecture of the skip-gram model in easy understand way with intuition in how the model works.

#### Continuous Bag of Word (CBOW)

#### Glove

### Recurrent Neural Network

Concept:

* [Andrew Karpathey's blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - a crazy long blog post explained about the use case of RNN.
* [iamtrask's post](http://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/) - this blog by [@iamtrack](https://twitter.com/iamtrask) shown the explanation of RNN concept using **numpy**
* [Only Numpy: Vanilla Recurrent Neural Network Deriving Back propagation Through Time Practice](https://towardsdatascience.com/only-numpy-vanilla-recurrent-neural-network-back-propagation-practice-math-956fbea32704) - easy understanding explaination of RNN using numpy
* [Peter's note](http://peterroelants.github.io/posts/rnn_implementation_part01/) - this note by [Peter Roelants](https://github.com/peterroelants) about the implementation of RNN

#### Gated Recurrent Unit (GRU)

#### Long-Short Term Memory (LSTM)

Concept:

* [Colah's post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - this is a must read post to understand about the shortage of RNN and how LSTM can help to overcome the problem (It also have some pretty good diagram explaining about LSTM)

Code:

* [Chris's implementation of LSTM using numpy](http://chris-chris.ai/2017/10/10/LSTM-LayerNorm-breakdown-eng/) - the author had demo the code for implementation of the LSTM using numpy only. Good for understanding of the underlaying concept of the LSTM
* [Introduction to LSTM from analyticsvidhya.com](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/) - this is a blog from the famous www.analyticsvidhya.com which explained about the concept of LSTM as well as its coding example using **Keras**
* [Vanilla LSTM with numpy](http://blog.varunajayasiri.com/numpy_lstm.html) - an implementation of vanilla LSTM using numpy by Varuna Jasiri. It used AdaGrad algorithm as stochastic gradient descent
* [Deriving LSTM Gradient for Backpropagation](https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/) - Agustinus Kristiadi's Blog implementation LSTM using numpy

### Convolutional Neural Network

## Github Repository for Deep Learning

After reading for some time, I try to implement the concept. But it was hard for newcomers like me to actually start to code all the algorithms from scratch. Thus, I found some github repo that can help me in better understanding the coding part.

* [nfmcclure/tensorflow_cookbook](https://github.com/nfmcclure/tensorflow_cookbook) - this repo contains the source code from [Tensorflow Machine Learning Cookbook](https://www.packtpub.com/big-data-and-business-intelligence/tensorflow-machine-learning-cookbook) which teaches a lot of machine learning and deep learning using Tensorflow.
* [Kulbear/deep-learning-coursera](https://github.com/Kulbear/deep-learning-coursera) - sometime when I faced difficulty in understanding some concepts and mathematic formula from *deep learning by Andrew Ng*, I will refer to this repo which the author have published the solution for the exercise. (It is important to understand how can we tackle the problems in other people's perspective also)
* [mbadry1/DeepLearning.ai-Summary](https://github.com/mbadry1/DeepLearning.ai-Summary) - the author's note about *deep learning by Andrew Ng* is really amazing.