# 6. SVM Applications

# **Linear SVM Classification**

The figure shows part of the iris dataset. The two classes can clearly be separated easily with a straight line. 

![Untitled](6%20SVM%20Applications/Untitled.png)

The left plot:

- The dashed line is so bad that it does not even separate the classes properly.
- The other two models work perfectly on this training set, but their decision boundaries come so close to the instances that these models will probably not perform as well on new instances.

The right plot: an SVM classifier

- The line not only separates the two classes but also stays as far away from the closest training instances as possible.

The decision boundary is fully determined (or “supported”) by the instances located on the
edge of the street (*support vectors*). So adding more training instances “off the street” will not affect the decision boundary.

<aside>
💡 SVMs are sensitive to the feature scales.

![Untitled](6%20SVM%20Applications/Untitled%201.png)

</aside>

# **Soft Margin Classification**

If we strictly impose that all instances be off the street and on the right side, this is called *hard margin classification*. 

- Only works if the data is linearly separable
- Quite sensitive to outliers.

![Untitled](6%20SVM%20Applications/Untitled%202.png)

When the iris dataset with just one additional outlier: 

- on the left, it is impossible to find a hard margin,
- on the right, the decision boundary will probably not generalize as well.

To avoid these issues it is preferable to use a more flexible model. The objective is to find a good balance between keeping the street as large as possible and limiting the *margin violations*. This is called *soft margin classification*.

In Scikit-Learn’s SVM classes, you can control this balance using the `C` hyperparameter: a smaller `C` value leads to a wider street but more margin violations.

![Untitled](6%20SVM%20Applications/Untitled%203.png)

<aside>
💡 If your SVM model is overfitting, you can try regularizing it by reducing C.

</aside>

The following Scikit-Learn code loads the iris dataset, scales the features, and then trains a linear SVM model (using the `LinearSVC` class with *C* = 1 and the *hinge loss* function, described shortly) to detect Iris-Virginica flowers.

```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica
svm_clf = Pipeline([
 ("scaler", StandardScaler()),
 ("linear_svc", LinearSVC(C=1, loss="hinge")),
 ])
svm_clf.fit(X, y)
```

```python
>>> svm_clf.predict([[5.5, 1.7]])
array([1.])
```

Unlike Logistic Regression classifiers, SVM classifiers do not output probabilities for each class.

The `LinearSVC` class regularizes the bias term, so you should center the training set first by subtracting its mean. This is automatic if you scale the data using the `StandardScaler`. Moreover, make sure you set the loss hyperparameter to "`hinge`", as it is not the default value. Finally, for better performance you should set the dual hyperparameter to `False`, unless there are more features than training instances (we will discuss duality later in the chapter).

# **Nonlinear SVM Classification**

Consider the left plot: it represents a simple dataset with just one feature $x_1$. This dataset is not linearly separable, as you can see. But if you add a second feature $x_2$ $= (x_1)^2$, the resulting 2D dataset is perfectly linearly separable.

![Untitled](6%20SVM%20Applications/Untitled%204.png)

To implement this idea using Scikit-Learn, you can create a `Pipeline` containing a `PolynomialFeatures` transformer (discussed in “`Polynomial Regression`”), followed by a `StandardScaler` and a `LinearSVC`. 

Let’s test this on the moons dataset: this is a toy dataset for binary classification in which the data points are shaped as two interleaving half circles. You can generate this dataset
using the `make_moons()`function:

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
 ("poly_features", PolynomialFeatures(degree=3)),
 ("scaler", StandardScaler()),
 ("svm_clf", LinearSVC(C=10, loss="hinge"))
 ])
polynomial_svm_clf.fit(X, y)
```

![Untitled](6%20SVM%20Applications/Untitled%205.png)

# **Polynomial Kernel**

When using SVMs you can apply an almost miraculous mathematical technique called the *kernel trick* . It makes it possible to get the same result as if you added many polynomial features, even with very high degree polynomials, without actually having to add them.

This trick is implemented by the `SVC` class. Let’s test it on the moons dataset:

```python
from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
 ("scaler", StandardScaler()),
 ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
 ])
poly_kernel_svm_clf.fit(X, y)
```

This code trains an SVM classifier using a 3rd-degree polynomial kernel. It is represented on the left of figure. On the right is another SVM classifier using a 10th degree polynomial kernel. Conversely, if it is underfitting, you can try increasing it. The hyperparameter `coef0` controls how much the model is influenced by highdegree polynomials versus low-degree polynomials.

![Untitled](6%20SVM%20Applications/Untitled%206.png)

# **Adding Similarity Features**

Another technique to tackle nonlinear problems is to add features computed using a *similarity function* that measures how much each instance resembles a particular *landmark*.

Next, let’s define the similarity function to be the Gaussian *Radial Basis Function* (*RBF*) with *γ* = 0.3

$$
\phi_\gamma(\mathbf{x}, \ell)=\exp \left(-\gamma\|\mathbf{x}-\ell\|^2\right)
$$

Therefore its new features are $x_2 = \exp (–0.3 × 12) ≈ 0.74$ and $x_3 = exp (–0.3 × 22) ≈ 0.30.$

![Untitled](6%20SVM%20Applications/Untitled%207.png)

# **Gaussian RBF Kernel**

Let’s try the Gaussian RBF kernel using the SVC class:

```python
rbf_kernel_svm_clf = Pipeline([
 ("scaler", StandardScaler()),
 ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
 ])
rbf_kernel_svm_clf.fit(X, y)
```

![Untitled](6%20SVM%20Applications/Untitled%208.png)

Conversely, a small gamma value makes the bell-shaped curve wider, so instances have a larger range of influence, and the decision boundary ends up smoother. So *γ* acts like a regularization hyperparameter: if your model is overfitting, you should reduce it, and if it is underfitting, you should increase it (similar to the C hyperparameter).