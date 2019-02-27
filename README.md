# some summary of the algorithms.
1. KNN: the important part is the distance matrix. using the matrix operation without loops is fastest!
2. ID3: entropy and InfoGain methods, recursive tree creation. In fact the code structure is not good and not easy to prune.
3. C4.5: change the InfoGain methods to InfoGainRatio.
4. CART: As the regression and the classification metrics is almost the same, I just create the classification model. And the prune metrics is easy than the book. Maybe I just want to know better about the algorithm, I have to go still.
