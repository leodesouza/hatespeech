# hatespeech

https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech

Project folder structure and best practices inspired by The AI Summer 
1: https://github.com/The-AI-Summer/Deep-Learning-In-Production/tree/master/1.%20From%20a%20notebook%20to%20serving%20millions%20of%20users
2: https://theaisummer.com/best-practices-deep-learning-code/

Best Practices
3 -https://github.com/The-AI-Summer/Deep-Learning-In-Production/tree/master/2.%20Writing%20Deep%20Learning%20code%3A%20Best%20Practises

**dataloader** is quite self-explanatory. All the data loading and data preprocessing classes and functions live here.

**evaluation** is a collection of code that aims to evaluate the performance and accuracy of our model.

**executor**: in this folder, we usually have all the functions and scripts that train the model or use it to predict something in different environments. And by different environments I mean: executors for GPUs, executors for distributed systems. This package is our connection with the outer world and it’s what our “main.py” will use.

**model** contains the actual deep learning code (we talk about tensorflow, pytorch etc)

**notebooks** include all of our jupyter/colab notebooks in one place.

**ops**: this one is not always needed, as it includes operations not related with machine learning such as algebraic transformations, image manipulation techniques or maybe graph operations.

**utils**: utilities functions that are used in more than one places and everything that don’t fall in on the above come here.