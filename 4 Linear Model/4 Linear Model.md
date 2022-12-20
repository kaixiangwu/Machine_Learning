# 4. Linear Model (1)

# Linear Regression

**An Example: Electricity Prediction**

- Problem:Predict the peak power consumption in summer.
  
  
    | Date | High Temperature(F) | Peak Demand(GW) |
    | --- | --- | --- |
    | 2011-06-01 | 84.0 | 2.651 |
    | 2011-06-02 | 73.0 | 2.081 |
    | 2011-06-03 | 75.2 | 1.844 |
    | … | … | … |
- Feature Engineering:
    - Primary key cannot be used as the features.
    - Instance Space: High Temperature(F)
    - Label Space: Peak Demand(GW)
- How to map from X to y?
    - Exploratory Data Analysis(EDA)to ease model selection: EDA
      
        ![https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213192518896.png](https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213192518896.png)
        
        - Why different peak demands for the same high temperature?
          
            It is inaccurate when using only one feature for prediction.
            

## Linear Regression: Hypothesis Space

- Training data: $(\boldsymbol{x_1},y_1),(\boldsymbol{x_2},y_2),...,(\boldsymbol{x_n},y_n)$
    - Feature vector: $\boldsymbol{x} \in \mathbb{R}^d$, response: $y \in \mathbb{R}$
- Add a constant dimension to feature vector x:
  
    $$
    \boldsymbol{x}=\left(\begin{array}{c} 1 \\ \boldsymbol{x_1} \\ \cdots \\ \boldsymbol{x_d} \end{array}\right) \in \mathbb{R}^{d+1} 
    $$
    
- Prediction of hypothesis $h$ parametrized by $\boldsymbol{\theta}$
  
    $$
    h_{\boldsymbol{\theta}}(\boldsymbol{x})=\boldsymbol{\theta_0}+\sum_{j=1}^a \boldsymbol{\theta_j} \boldsymbol{x_j}=\sum_{j=0}^a \boldsymbol{\theta_j}\boldsymbol{x_j}=\boldsymbol{\theta} \cdot \boldsymbol{x}
    $$
    
    - $\boldsymbol{\theta}$ is the model’s *parameter vector*, containing the bias term and the feature weights $\boldsymbol{\theta}_0$ to $\boldsymbol{\theta}_n$.
    - $\boldsymbol{x}$ is the instance’s *feature vector*, containing $\boldsymbol{x}_0$ to $\boldsymbol{x}_0$, with $\boldsymbol{x}_0$ always equal to 1.
    - $**h_{\boldsymbol{\theta}}(\boldsymbol{x})$** is the hypothesis function, using the model parameters $\boldsymbol{\theta}$
    
    > In Machine Learning, vectors are often represented as *column vectors*, which are 2D arrays with a single column. 
    If $**\boldsymbol{\theta}**$ and $**\boldsymbol{x}**$ are column vectors, then the prediction is: $*\hat y = \boldsymbol{\theta}^T\boldsymbol{x}*$
    > 

## Linear Regression: Loss Function

- Use the **squared loss (L2 loss)** to compute the error on training set
  
    $$
     \hat{\epsilon}(h)=\sum_{i=1}^n\left(h\left(\boldsymbol{x}_i\right)-y_i\right)^2
    $$
    
    Or the Mean Square Error (MSE)
    
    $$
    \operatorname{MSE}\left(\mathbf{X}, h_{\boldsymbol{\theta}}\right)=\frac{1}{m} \sum_{i=1}^m\left(\boldsymbol{\theta}^T \mathbf{x}_{i}-y_{i}\right)^2
    $$
    
    To simplify notations, we will just write $\text{MSE}(\boldsymbol{\theta})$ instead of $\text{MSE}(\boldsymbol{X},\boldsymbol{\theta})$
    
    ![https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213194217778.png](https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213194217778.png)
    

## Linear Regression: Analytical Solution

![https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213200243725.png](https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/image-20221213200243725.png)

**Optimization** 

$$
\min _{\boldsymbol{\theta}} \sum_{i=1}^n\left(h_{\boldsymbol{\theta}}\left(\boldsymbol{x}_i\right)-y_i\right)^2
$$

- **Training error of linear regression:**
  
    $$
    \begin{aligned} \hat{\epsilon}(\boldsymbol{\theta}) & =\sum_{i=1}^n\left(\boldsymbol{\theta}^T \boldsymbol{x}_i-y_i\right)^2 \\ & =\|X \boldsymbol{\theta}-y\|^2 \end{aligned} 
    $$
    
- Put everything in matrix form:
  
    data matrix(design matrix) n×(d+1)
    
    label matrix n
    
    $$
     \boldsymbol{X}=\left[\begin{array}{c}-\boldsymbol{x}_1^T- \\ -\boldsymbol{x}_2^T- \\ \vdots \\ -\boldsymbol{x}_n^T-\end{array}\right], \quad \boldsymbol{y}=\left[\begin{array}{c}y_1 \\ y_2 \\ \vdots \\ y_n\end{array}\right]
    $$
    
    Computing the gradient of $\boldsymbol{\theta}$ and setting it to zero yields the optimal parameter $\boldsymbol{\theta}^\ast$: 
    
    $$
    \begin{aligned}
      \hat{\epsilon}(\boldsymbol{\theta}) & =\|\boldsymbol{X} \boldsymbol{\theta}-\boldsymbol{y}\|^2 \\
      \nabla_{\boldsymbol{\theta}} \hat{\epsilon}(\boldsymbol{\theta}) & =2 \boldsymbol{X}^T(\boldsymbol{X} \boldsymbol{\theta}-\boldsymbol{y})=\mathbf{0} \\
      & \Rightarrow \boldsymbol{X}^T \boldsymbol{X} \boldsymbol{\theta}=\boldsymbol{X}^T \boldsymbol{y} \\
      & \Rightarrow \boldsymbol{\theta}=\left(\boldsymbol{X}^T \boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}
      \end{aligned}
    $$
    
    - Computational complexity is big! $O[d^2(d+n)]$

Let’s generate some linear-looking data to test this equation $\boldsymbol{\theta}=\left(\boldsymbol{X}^T \boldsymbol{X}\right)^{-1} \boldsymbol{X}^T \boldsymbol{y}$

```python
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

```python
>>> theta_best
array([[4.21509616],
			 [2.77011339]])
```

We would have hoped for $*\theta_0*$ = 4 and $*\theta_1*$ = 3 instead of $*\theta_0*$ = 4.215 and $*\theta_1*$ = 2.770. Close

enough, but the noise made it impossible to recover the exact parameters of the original function.

Now you can make predictions using $**\hat {\boldsymbol{\theta}}**$:

```python
>>> X_new = np.array([[0], [2]])
>>> X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
>>> y_predict = X_new_b.dot(theta_best)
>>> y_predict
array([[4.21509616],
 [9.75532293]])
```

Plot this model’s predictions

```python
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
```

![Untitled](4%20Linear%20Model/Untitled.png)

Performing linear regression using Scikit-Learn is quite simple:

```python
>>> from sklearn.linear_model import LinearRegression

>>> lin_reg = LinearRegression()
>>> lin_reg.fit(X, y)
>>> lin_reg.intercept_, lin_reg.coef_
(array([4.21509616]), array([[2.77011339]]))
>>> lin_reg.predict(X_new)
array([[4.21509616],
 [9.75532293]])
```

Now we will look at very different ways to train a Linear Regression model, better
suited for cases where there are a large number of features, or too many training
instances to fit in memory.

## Optimization: **Gradient Descent**

For general differentiable loss function,use **Gradient Descent**(GD)

Concretely, you start by filling **θ** with random values (*random initialization*), and then you improve it gradually, taking one baby step at a time, each step attempting to decrease the cost function, until the algorithm *converges* to a minimum

- Gradient​
    - Gradient shows direction that function varies fastest
      
        $$
        \begin{gathered} \boldsymbol{g}=\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) \\ g_j=\nabla_{w_j} J(\boldsymbol{w}) \end{gathered} 
        $$
        

![Untitled](4%20Linear%20Model/Untitled%201.png)

![https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/tiduxiajiang-1.png](https://typora-1309501826.cos.ap-nanjing.myqcloud.com//%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/tiduxiajiang-1.png)

- Progress:
    - First-order Taylor approximation of the objective function:
      
        $$
        J(\boldsymbol{w})=J\left(\boldsymbol{w}_0\right)+\left(\boldsymbol{w}-\boldsymbol{w}_0\right)^T \boldsymbol{g}+\cdots
        $$
        
    - Go along gradient for a step with a small rate $\eta$:
      
        An important parameter in Gradient Descent is the size of the steps, determined by the *learning rate* hyperparameter. If the learning rate is too small, then the algorithm will have to go through many iterations to converge, which will take a long time
        
    
    $$
    J(\boldsymbol{w}-\eta \boldsymbol{g}) \approx J(\boldsymbol{w})-\eta \boldsymbol{g}^T \boldsymbol{g}
    $$
    
    ![Untitled](4%20Linear%20Model/Untitled%202.png)
    
    ![Untitled](4%20Linear%20Model/Untitled%203.png)
    
    - Repeat this step, and we get the Gradient Descent(GD) algorithm.
- Finally, not all cost functions are convex .There may be holes, ridges, plateaus, and all sorts of irregular terrains, making convergence to the minimum very difficult.
  
    The figure below shows the two main challenges with Gradient Descent: if the random initialization starts the algorithm on the left, then it will converge to a *local minimum*, which is not as good as the *global minimum*. If it starts on the right, then it will take a very long time to cross the plateau, and if you stop too early you will never reach the global minimum.
    
    ![Untitled](4%20Linear%20Model/Untitled%204.png)
    

Fortunately, the MSE cost function for a Linear Regression model happens to be a *convex function.*

In fact, the cost function has the shape of a bowl, but it can be an elongated bowl if
the features have very different scales.

![Untitled](4%20Linear%20Model/Untitled%205.png)

- On the left the Gradient Descent algorithm goes straight toward the minimum, thereby reaching it quickly,
- On the right it first goes in a direction almost orthogonal to the direction of the global minimum, and it ends with a long march down an almost flat valley. It will eventually reach the minimum, but it will take a long time.

> When using Gradient Descent, you should ensure that all features have a similar scale (e.g., using Scikit-Learn’s StandardScaler class), or else it will take much longer to converge.
> 

### **Batch Gradient Descent**

- *Partial derivatives of the cost function*
  
    $$
    \frac{\partial}{\partial \theta_j} \operatorname{MSE}(\boldsymbol{\theta})=\frac{2}{m_i} \sum_{i=1}^m\left(\boldsymbol{\theta}^T \mathbf{x}^{(i)}-y^{(i)}\right) x_j^{(i)}
    $$
    
- *Gradient vector of the cost function*
  
    $$
    \nabla_{\boldsymbol{\theta}} \operatorname{MSE}(\boldsymbol{\theta})=\left(\begin{array}{c}\dfrac{\partial}{\partial \theta_0} \operatorname{MSE}(\boldsymbol{\theta}) \\\dfrac{\partial}{\partial \theta_1} \operatorname{MSE}(\boldsymbol{\theta}) \\\vdots \\\dfrac{\partial}{\partial \theta_n} \operatorname{MSE}(\boldsymbol{\theta})\end{array}\right)=\frac{2}{m} \mathbf{X}^T(\mathbf{X} \boldsymbol{\theta}-\mathbf{y})
    $$
    

Once you have the gradient vector, which points uphill, just go in the opposite direction to go downhill.

$$
\boldsymbol{\theta}^{\text {(next step })}=\boldsymbol{\theta}-\eta \nabla_{\boldsymbol{\theta}} \operatorname{MSE}(\boldsymbol{\theta})
$$

A quick implementation of this algorithm:

```python
eta = 0.1 # learning rate
n_iterations = 1000
theta = np.random.randn(2,1) # random initialization
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients
```

```python
>>> theta
array([[4.21509616],
 [2.77011339]])
```

The first 10 steps of Gradient Descent using three different learning rates.

![Untitled](4%20Linear%20Model/Untitled%206.png)

- On the left, the learning rate is too low.
- In the middle, the learning rate looks pretty good.
- On the right, the learning rate is too high: the algorithm diverges, jumping all over the place and actually getting further and further away from the solution at every step.

How to set the number of iterations?

- A simple solution is to set a very large number of iterations but to interrupt the algorithm when the gradient vector becomes tiny—that is, when its norm becomes smaller than a tiny number *ϵ* (called the *tolerance*).

### **Stochastic Gradient Descent**

*Stochastic Gradient Descent* just picks a random instance in the training set at every step and computes the gradients based only on that single instance.

Due to its stochastic (i.e., random) nature, this algorithm is much less regular than Batch Gradient Descent: instead of gently decreasing until it reaches the minimum, the cost function will bounce up and down, decreasing only on average.

![Untitled](4%20Linear%20Model/Untitled%207.png)

The final parameter values are good, but not optimal. When the cost function is very irregular , this can actually help the algorithm jump out of local minima, so Stochastic Gradient Descent has a better chance of finding the global minimum than Batch Gradient Descent does.

One solution to this dilemma is to gradually reduce the learning rate → *simulated annealing*

![Untitled](4%20Linear%20Model/Untitled%208.png)

> When using Stochastic Gradient Descent, the training instances must be independent and identically distributed (IID), to ensure that the parameters get pulled towards the global optimum, on average. 
A simple way to ensure this is to shuffle the instances during training.
> 

To perform Linear Regression using SGD the `SGDRegressor` class of Scikit-Learn, which defaults to optimizing the **squared error cost function**.

- Run for maximum 1000 epochs (max_iter=1000) or until the loss drops
by less than 1e-3 during one epoch (tol=1e-3)
- Start with a learning rate of 0.1 (eta0=0.1)

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

```python
>>> sgd_reg.intercept_, sgd_reg.coef_
(array([4.24365286]), array([2.8250878]))
```

### **Mini-batch Gradient Descent**

Stochastic Gradient Descent: at each step, instead of computing the gradients based on the full training set (as in Batch GD) or based on just one instance (as in Stochastic GD), Mini batch GD computes the gradients on small random sets of instances called *minibatches*

![Untitled](4%20Linear%20Model/Untitled%209.png)

Comparison of algorithms for Linear Regression:

![Untitled](4%20Linear%20Model/Untitled%2010.png)