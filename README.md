# some summary of the algorithms.
# fix some file type error in 2019/7/16
1. KNN: the important part is the distance matrix. using the matrix operation without loops is fastest!
2. ID3: entropy and InfoGain methods, recursive tree creation. In fact the code structure is not good and not easy to prune.
3. C4.5: change the InfoGain methods to InfoGainRatio.
4. CART: As the regression and the classification metrics is almost the same, I just create the classification model. And the prune metrics is easy than the book. Maybe I just want to know better about the algorithm, I have to go still.
5. LR: the important part is the loss_function and the derivative of the J(w) and use matrix to show them. Add the l2 penalty, you can just change a little codes to use the l1 or Lp..
6. Perception:create the dual form and the original form Perceptron! Try hard!
7. AdaBoost: A silly work for AdaBoost. Just for special 2 class classification. But you will know the process in Adaptive Boosting methods..
8. RandomForest: just set the n_estimators. we can set the max_features, but I just do it very easy. if you want to change to the ExtraTreesClassifier, just set the splitter='random'. I don't show the n_sub concept in the code. just easy...
9. DNN: using 8 lines to produce the one layer DNN. the complex DNN is just for Regression problems. if want to change to Classification, just change the Loss function and add the softmax..But it just works well on few layers. Maybe the weights initialization is not good??
