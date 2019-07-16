#Stanford Courses are available, no more changes are made.
import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    r1 = np.sum(np.square(X), axis=1).reshape(num_test, 1)*np.ones((1, num_train))
    r2 = np.ones((num_test, 1))*np.sum(np.square(self.X_train),axis=1)
    dists = np.sqrt(r1 + r2-2*np.dot(X, self.X_train.T))
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      closest_y = []
      matri = np.argsort(dists[i])[:k]
      closest_y = self.y_train[matri]
      y_pred[i] = np.argmax(np.bincount(closest_y))
    return y_pred

#compare with sklearn's KNeighborsClassifier using mnist dataset
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mnist = fetch_mldata("MNIST original")
X, y = mnist["data"].astype(np.int64), mnist['target'].astype(np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1,random_state=42)
X_train = X_train[:900]
y_train = y_train[:900]
X_test = X_test[:100]
y_test = y_test[:100]
knn_sklearn = KNeighborsClassifier(n_neighbors = 7)
knn_sklearn.fit(X_train, y_train)
knn_own = KNearestNeighbor()
knn_own.train(X_train, y_train)
dists = knn_own.compute_distances_two_loops(X_test)
print(accuracy_score(knn_sklearn.predict(X_test), y_test))
print(accuracy_score(knn_own.predict_labels(dists, k=7), y_test))
#output:0.91
#output:0.91
