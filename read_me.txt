Shrink or sink submissions 
Since the data for STL is very less  I used heavy data augimentation with flipping images and rotating images , and I also did normalization
Now the architecture of my model is using 5 layers of CNN 
and one MLP along with batch normalization along 
To make decrase the parameters for MLP I took the mean of every channels passesd down by conv layers and made them parameters

Intially I trained with batch size of 32 and learning rate of 10 power -3 along with decay and a shedular but I saw 
that loss was decreasing near 45-50 epochs but very very slowwly , so then I increase the learning rate and ran another 30 epochs , which gave me my final correct state

Also I used data loaders which augumented the data from train data set every new batch they took

After all this to make the model even smaller I did quantisation to 8 bit int, it decreased  the accuracy around 1.5 percent but size decreased immensly 

I got an accuracy of  0.696875
and model size is 64 kb

This is my final quantised model

