# 3. K-Nearest Neighbor

# ****Nearest Neighbor(1NN)****

![Untitled](3%20K-Nearest%20Neighbor/Untitled.png)

- To classify a new example $x$:
    - Label $x$ with the label of the closest example to $x$in the training set.
- Euclidean Distance
  
    $$
    D\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\sqrt{\sum\limits_{k=1}^d\left(x_{i k}-x_{j k}\right)^2}
    $$
    

# **K-Nearest-Neighbors (KNN)**

![Untitled](3%20K-Nearest%20Neighbor/Untitled%201.png)

- Low latitude large-scale data
- $x$ is close to a blue point.
  
    But most of the next closest points are red.
    
- Find $k$ nearest neighbors of $x$.
- Label $x$ with the majority label within the $k$ nearest neighbors

**Consider:How can we handle ties for even values of k?**

> **KNN rule is certainly simple and intuitive,but does it work?**
> 
> - KNN is a Goo Approximator on any smooth distribution:
>     - Converges to perfect solution if clear separation
>         
>         $$
>         \lim _{n \rightarrow \infty} \epsilon_{\mathrm{KNN}}(n) \leq 2 \epsilon^*\left(1-\epsilon^*\right)
>         $$
>         
>         (Cover Hart,1969)
>         
> - KNN is Good Model on complex distributions.

# ****The Effect of K****

![Untitled](3%20K-Nearest%20Neighbor/Untitled.jpeg)

![Untitled](3%20K-Nearest%20Neighbor/Untitled%201.jpeg)

- Increasing k simplifies the decision boundary
  
    Majority voting means less emphasis on individual points
    
- Choose best k on the validation set.(Often set k as an odd number).

# KNN: Inductive Bias

- Similar points have similar labels.
- All dimensions are created equal!

# Feature Normalization

- Z-score normalization:
    - For each feature dimension j,compute based on its samples:
    - Mean $\mu_j=\dfrac{1}{N} \sum\limits_{i=1}^N x_{i j}$
    - Variance $\sigma_j=\sqrt{\dfrac{1}{N} \sum\limits_{i=1}^N\left(x_{i j}-\mu_j\right)^2}$
    - Normalize the feature into a new one: $\hat{x}_{i j} \leftarrow \dfrac{x_{i j}-\mu_j}{\sigma_j}$

# Distance Selection

- Cosine Distance: $\cos \left(\boldsymbol{x_i}, \boldsymbol{x_j}\right)=\dfrac{\left\langle \boldsymbol{x_i}, \boldsymbol{x_j}\right\rangle}{\left\|\boldsymbol{x_i}\right\|\left\|\boldsymbol{x_j}\right\|}$
- Minkowski Distance: $D_p\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\sqrt[p]{\sum\limits_{k=1}^d\left|x_{i k}-x_{j k}\right|^p}$
- Mahalanobis Distance: $D_M\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right)=\sqrt{\left(\boldsymbol{x}_i-\boldsymbol{x}_j\right)^{\mathrm{T}} \boldsymbol{M}\left(\boldsymbol{x}_i-\boldsymbol{x}_j\right)}$

# Weighted KNN

- Vanilla KNN uses majority vote to predict discrete outputs.
- Weighted KNN assigns greater weights to closer neighbors.
    - Weighting is a common approach used in machine learning.
- Distance-weighted Classification:
  
    $$
    \hat{y}=\operatorname{argmax}_{c \in y} \sum_{\boldsymbol{x}^{\prime} \in \operatorname{KNN}_c(\boldsymbol{x})} \frac{1}{D\left(\boldsymbol{x}, \boldsymbol{x}^{\prime}\right)^2}
    $$
    
- Distance-weighted Regression:
  
    $$
    \hat{y}=\sum_{x^{\prime} \in \operatorname{KNN}(x)} \frac{1}{D\left(x, x^{\prime}\right)^2} y^{\prime}
    $$
    
- Other weighting function (e.g.exponential family)can also be used.

# ****KNN Summary****

- When to use:
    - Few attributes per instance (expensive computation)
    - Lots of training data (curse of dimensionality
- Advantages:
    - Agnostically learn complex target functions
    - Do not lose information (store original data)
    - Data number can be very large (big pro!)
    - Class number can be very large (biggest pro!)
        - All other ML algorithms may fail here!
- Disadvantages:
    - Slow at inference time (acceleration a must)
    - Ineffective in high dimensions(curse of dimensionality)
    - Fooled easily by irrelevant attributes (feature engineering crucial)