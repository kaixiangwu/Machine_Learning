# 5. SVM Theory

# **Which is the Best Classifier?**

- Consider the simplest linearly separable setting. Which classifier is the best classifier, ie. will yield the lowest test error?
  
    ![Untitled](5%20SVM%20Theory/Untitled.png)
    
- With noise growing, only one linear classifier left.
    - This classifier has the **largest margin** to training data.
    - This classifier is **the most robust classifier** to the noisy data.
    
    ![Untitled](5%20SVM%20Theory/Untitled%201.png)
    

# Support Vector Machine: Margin

- Margin: Twice of the distance to the closest points of either class.
- Problem: How to find the linear classifier with the largest margin?
- Requirements:
    - The margin is the largest.
    - Classify all data points correctly.
- Constrained optimization problem:
  
    $$
    \begin{gathered}\max _{w, b} \operatorname{margin}(\boldsymbol{w}, b) \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1,1 \leq i \leq n\end{gathered}
    $$
    

![Untitled](5%20SVM%20Theory/Untitled%202.png)

- How to quantify the margin?
  
    $$
    r=\frac{|\boldsymbol{w} \cdot \boldsymbol{x}+b|}{\|\boldsymbol{w}\|_2}
    $$
    
    $$
    \gamma=\frac{1}{\|\boldsymbol{w}\|_2}+\frac{|-1|}{\|\boldsymbol{w}\|_2}=\frac{2}{\|\boldsymbol{w}\|_2}
    $$
    
- **Hard-margin** Support Vector Machine (SVM):
  
    $$
    \begin{gathered}\max _{\boldsymbol{w}, b} \frac{2}{\|\boldsymbol{w}\|_2} \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1,1 \leq i \leq n\end{gathered}
    $$
    
    - Non-convex problem. But it is equivalent to:
      
        $$
        \begin{gathered}\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|_2^2 \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1,1 \leq i \leq n\end{gathered}
        $$
    
- This is only for the linearly separable case. Hardly used in practice !

# Linearly Non-Separabe C|assification

![Untitled](5%20SVM%20Theory/Untitled%203.png)

- In the linearly non-separable case, we **cannot find a solution** to the hard-margin support vector machine (left).
- Even we can finda solution,small margin may cause overfitting.
- Instead of constraining all data points to be correctly classified:
    - Allow some points on the wrong side of the margin.
    - Their number should be small.
    
    ![Untitled](5%20SVM%20Theory/Untitled%204.png)
    
    $$
    \begin{gathered}\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|_2^2 \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1-\xi_i \\\xi_i \geq 0, \sum_{i=1}^n \xi_i \leq n^{\prime} \\1 \leq i \leq n\end{gathered}
    $$
    
    - Slack variables: $\xi_i,i\in[n]$
    - Computationally, we re-express in the (Lagrangian) equivalent form:
      
        **Soft-margin** Support Vector Machine (SVM)
        
        $$
        \begin{gathered}\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|_2^2+C \sum_{i=1}^n \xi_i \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1-\xi_i \\\xi_i \geq 0,1 \leq i \leq n\end{gathered}
        $$
        

# Optimization

- Consider a general optimization problem:
  
    $$
    \begin{aligned}& \min _x f(x) \\& \text { s.t. } x \in \mathcal{X}\end{aligned}
    $$
    
- This problem is called a convex optimization problem if
    - $\mathcal{X}$ is a convex domain and $f(x)$ is a convex function
    - $\mathcal{S}$ is convex set if $x,y\in \mathcal{S}\Rightarrow \forall \lambda\in[0,1],\lambda x+(1-\lambda )y\in S$
      
        ![Untitled](5%20SVM%20Theory/Untitled%205.png)
        
    - $f$ is convex function if $\forall \lambda\in[0,1],x,x'\in\text{dom}(f):$
      
        $$
        f\left(\lambda x+(1-\lambda) x^{\prime}\right) \leq \lambda f(x)+(1-\lambda) f\left(x^{\prime}\right)
        $$
        
        ![Untitled](5%20SVM%20Theory/Untitled%206.png)
        

## Equality Constraints

- Consider equality constraints:
  
    $$
    \begin{gathered}\min _{\boldsymbol{x} \in \mathbb{R}^d} f(\boldsymbol{x}) \\\text { s. t. } g(\boldsymbol{x})=0\end{gathered}
    $$
    
    ![Untitled](5%20SVM%20Theory/Untitled%207.png)
    
- For any $\boldsymbol{x}$ satisfying $g(\boldsymbol{x})= 0$:
    - $\nabla_{\boldsymbol{x} }g(\boldsymbol{x})$ is orthogonal to the tangent of the surface.
    - Since: the value $g(\boldsymbol{x})$ will not change after moving a short distance along the constraint surface.
- For the optimal $\boldsymbol{x}^\ast$ on the constraint surface that minimizes $f(\boldsymbol{x}^\ast)$:
    - $\nabla_{\boldsymbol{x} }f(\boldsymbol{x})$ is orthogonal to the tangent of the surface.
    - Otherwise: we could increase the value of $f(\boldsymbol{x})$ by moving a short distance along the constraint surface.
- So $\nabla_{\boldsymbol{x} }f(\boldsymbol{x})$ and $\nabla_{\boldsymbol{x} }g(\boldsymbol{x})$ are parallel vectors:
  
    $$
    \nabla f+\mu\nabla g=0 \,\text{for}\,\mu\neq0
    $$
    
- Lagrangian function of this problem is defined by:
  
    $$
    L(\boldsymbol{x}, \mu)=f(\boldsymbol{x})+\mu g(\boldsymbol{x})
    $$
    
    $$
    \begin{aligned}& -\nabla_x L=0 \Leftrightarrow \nabla f+\mu \nabla g=0 \\& -\frac{\partial}{\partial \mu} L=0 \Leftrightarrow g(x)=0\end{aligned}
    $$
    
    - This gives $d + 1$ equations that determine $\boldsymbol{x} \in \mathbb{R}^d$ and $\mu$.
    - Stationary point (gradient is zero on the tangent) of $L(\boldsymbol{x}, \mu)$ gives the $\boldsymbol{x}$ that minimizes $f(\boldsymbol{x})$ subject to constraint $g(\boldsymbol{x} )= 0$.

## Inequality Constraints

- Consider inequality constraints:
  
    $$
    \begin{gathered}\min _{\boldsymbol{x} \in \mathbb{R}^d} f(\boldsymbol{x}) \\\text { s. t. } g(\boldsymbol{x}) \leq 0\end{gathered}
    $$
    
    ![Untitled](5%20SVM%20Theory/Untitled%208.png)
    
- If the constrained stationary point $\boldsymbol{x}^\ast$ lies in the reigon $g(\boldsymbol{x}) \leq 0$ (e.g. $\boldsymbol{x}_A$)
    - Solve  $\nabla_{\boldsymbol{x}} f$ directly.
    - $(\boldsymbol{x}^\ast,\lambda)$ is a stationary point of $L(\boldsymbol{x}, \lambda)=f(\boldsymbol{x})+\lambda g(\boldsymbol{x})$
- Else the stationary point point $\boldsymbol{x}^\ast$ lies on the surface $g(\boldsymbol{x}) = 0$(e.g. $\boldsymbol{x}_B$)
    - $f(\boldsymbol{x})$ will only be at a minimum if its gradient oriented into the region $g(\boldsymbol{x}) < 0 \Leftrightarrow \nabla f+\lambda\nabla g=0$ for some $\lambda > 0$
    - Otherwise: exist a point in region $g(\boldsymbol{x}) < 0$  with a smaller $f(\boldsymbol{x})$

## Karush-Kuhn-Tucker Conditions (KKT)

The solution to the constrained optimization problem with inequality constraints is yielded by optimizing the **Lagrangian function**

$$
L(\boldsymbol{x}, \lambda)=f(\boldsymbol{x})+\lambda g(\boldsymbol{x})
$$

with respect to $\boldsymbol{x}$ and $\lambda$ subject to:

$$
\begin{aligned}&\text { Karush-Kuhn-Tucker (KKT) }\\&\begin{array}{ll}g(\boldsymbol{x}) \leq 0 & \text { (Primal feasibility) } \\\lambda \geq 0 & \text { (Dual feasibility) } \\\lambda g(\boldsymbol{x})=0 & \text { (Complementary slackness) }\end{array}\end{aligned}
$$

## General Lagrangian Function

- Consider constrained optimization problem (Primal Problem):
  
    $$
    \begin{gathered}\min _{x \in \mathbb{R}^d} f_{\boldsymbol{k}}(\boldsymbol{x}) \\\text { s.t. } g_j(\boldsymbol{x}) \leq 0 \text { for } j=1, \ldots, J \\h_k(\boldsymbol{x})=0 \text { for } k=1, \ldots, K\end{gathered}
    $$
    
    - [IGNORE] The optimization problem satisfies some regularity conditions:
        - Linearity constraint qualification
        - Linear independence (equal vs. inequal) constraint qualification
- With Lagrange multipliers $\lambda,\mu$, the Lagrangian function is defined as
  
    $$
    L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})=f(\boldsymbol{x})+\sum_{j=1}^J \lambda_j g_j(\boldsymbol{x})+\sum_{k=1}^K \mu_k h_k(\boldsymbol{x})
    $$
    
- KKT conditions: for $1\leq j \leq J$
    - Primal feasibility: $g_j(\boldsymbol{x}) \leq 0 ,h_k(\boldsymbol{x})=0$
    - Dual feasibility: $\lambda_j \geq 0$
    - Complementary slackness: $\lambda_j g_j(\boldsymbol{x})=0$
    - The gradient of the Lagrangian function w.r.t. $\boldsymbol{x}$ vanishes to $\boldsymbol{0}$:
      
        $$
        \nabla_x f(\boldsymbol{x})+\sum_{j=1}^J \lambda_j \nabla_x g_j(\boldsymbol{x})+\sum_{k=1}^K \mu_k \nabla_x h_k(\boldsymbol{x})=\mathbf{0}
        $$
    
- if $\boldsymbol{x}^\ast,\boldsymbol{\lambda}^\ast,\boldsymbol{\mu}^\ast$ satisfy KKT for a convex problem, then $\boldsymbol{x}^\ast$ is an optimum:
  
    $$
    f\left(\boldsymbol{x}^\ast\right)=L\left(\boldsymbol{x}^\ast, \boldsymbol{\lambda}^\ast, \boldsymbol{\mu}^\ast\right)
    $$
    

## Primal Problem and Dual Problem

- Primal Problem:
  
    $$
    \begin{gathered}\min _{x \in \mathbb{R}^d} f_{\boldsymbol{k}}(\boldsymbol{x}) \\\text { s.t. } g_j(\boldsymbol{x}) \leq 0 \text { for } j=1, \ldots, J \\h_k(\boldsymbol{x})=0 \text { for } k=1, \ldots, K\end{gathered}
    $$
    
- The Lagrangian Dual Problem:
  
    $$
    \begin{gathered}\max _{\boldsymbol{\lambda} \in \mathbb{R}^J, \boldsymbol{\mu} \in \mathbb{R}^K} \Gamma(\boldsymbol{\lambda}, \boldsymbol{\mu}) \triangleq \inf _{\boldsymbol{x}} L_{\boldsymbol{d}}(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) \\\text { s.t. } \lambda_j \geq 0 \text { for } j=1, \ldots, J\end{gathered}
    $$
    
    - It can be proved that $\Gamma(\boldsymbol{\lambda}, \boldsymbol{\mu})$ is always concave for any primal problems.
    - Theorem: Pointwise infimum of affine functions is concave.
- How to compute the Lagrangian dual function $\Gamma(\boldsymbol{\lambda}, \boldsymbol{\mu})$ ?
    - Compute $\boldsymbol{x}=\psi(\boldsymbol{\lambda},\boldsymbol{\mu})$ from $\nabla_{\boldsymbol{x}}L=0$ (If closed-form, then easy)
- The relationship between primal objective and dual function:
    - If $\bar{\boldsymbol{x}}$ is any point that satisfies the constraints:
      
        $$
        \inf _x L(\boldsymbol{x}, \boldsymbol{\lambda}, \boldsymbol{l}) \leq f(\overline{\boldsymbol{x}})+\sum_{j=1}^J \lambda_j g_j(\bar{\boldsymbol{x}})+\sum_{k=1}^K \mu_k h_k(\overline{\boldsymbol{x}}) \leq f(\overline{\boldsymbol{x}})
        $$
        
    - So for any $\boldsymbol{\lambda}, \boldsymbol{\mu} \text { s.t. } \boldsymbol{\lambda} \geq \mathbf{0}, \Gamma(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq f\left(\boldsymbol{x}^*\right) \text {. }$

# Soft-SVM: Dual Problem

Soft-margin SVM

$$
\begin{gathered}\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|_2^2+C \sum_{i=1}^n \xi_i \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1-\xi_i \\\xi_i \geq 0,1 \leq i \leq n\end{gathered}
$$

Lagrangian function (with 2n inequality constraints):

$$
\begin{gathered}L(\boldsymbol{w}, b, \boldsymbol{\alpha}, \boldsymbol{\xi}, \boldsymbol{\mu}) \\=\frac{1}{2}\|\boldsymbol{w}\|_2^2+C \sum_{i=1}^n \xi_i+\sum_{\substack{i=1 \\}}^n \alpha_i\left(1-\xi_i-y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right)\right)-\sum_{i=1}^n \mu_i \xi_i \\\alpha_i \geq 0, \mu_i \geq 0, i=1, \ldots, n\end{gathered}
$$

- Take the partial derivatives of Lagrangian w.rt $\boldsymbol{w},b,\xi_i$ and set to zero:
  
    $$
    \begin{aligned}& -\frac{\partial L}{\partial w}=\mathbf{0} \Rightarrow \boldsymbol{w}=\sum_{i=1}^n \alpha_i y_i \boldsymbol{x}_i \\& -\frac{\partial L}{\partial b}=0 \Rightarrow \sum_{i=1}^n \alpha_i y_i=0 \\& -\frac{\partial L}{\partial \xi_i}=0 \Rightarrow C=\alpha_i+\mu_i, i=1, \ldots, n\end{aligned}
    $$
    
- Dual Problem of Soft-SVM:
  
    $$
    \begin{gathered}\max _\alpha \sum_{i=1}^n \alpha_i-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j\left(x_i \cdot x_j\right) \\\text { s.t. } \sum_{i=1}^n \alpha_i y_i=0 \\0 \leq \alpha_i \leq C, 1 \leq i \leq n\end{gathered}
    $$
    
    - Solved by Quadratic Program with linear constraints: slow!
    - Solved by Sequential Minimal Optimization (SMO): fast!
        - After solving for $\boldsymbol{\alpha}$, we can solve for $\boldsymbol{w}^\ast=\sum_{i=1}^n \alpha_i^\ast y_i \boldsymbol{x}_i ,b^\ast=y_j-\sum_{i=1}^n \alpha_i^\ast y_i (\boldsymbol{x}_i\cdot \boldsymbol{x}_j )$
        - Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines
- Define Hinge loss $\ell(f(\boldsymbol{x}), y)=\max \{0,1-y f(\boldsymbol{x})\}$
- Theorem: Soft-SVM is equivalent to Regularized Risk Minimization:

$$
\begin{gathered}\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|\boldsymbol{w}\|_2^2+C \sum_{i=1}^n \xi_i \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right) \geq 1-\xi_i \\\xi_i \geq 0,1 \leq i \leq n\end{gathered}
$$

$$
\min _{\boldsymbol{w}, b} \lambda\|\boldsymbol{w}\|_2^2+\sum_{i=1}^n [1-y_i\left(\boldsymbol{w} \cdot \boldsymbol{x}_i+b\right)]_+
$$

# Support Vector Regression (SVR)

- $\epsilon$ -insensitive loss:
  
    $$
    \max \left(\left|h_w(\boldsymbol{x})-y\right|-\epsilon, 0\right)
    $$
    

$$
\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|_2^2+\frac{c}{n} \sum_{i=1}^n \max \left(0,\left|\boldsymbol{w} \cdot \boldsymbol{x}_i+b-y_i\right|-\boldsymbol{\epsilon}\right)
$$

# Transductive SVM (TSVM)

- Semi-supervised learning from labeled data $\mathcal{L}$ and unlabeled data $\mathcal{U}$.
- Transductive SVM:
  
    $$
    \begin{gathered}\min _{\boldsymbol{w}, b, \xi} \frac{1}{2}\|w\|_2^2+C_L \sum_{i=1}^L \xi_i+C_U \sum_{i=L+1}^{L+U} \xi_i \\\text { s.t. } y_i\left(\boldsymbol{w} \cdot \phi\left(\boldsymbol{x}_i\right)+b\right) \geq 1-\xi_i, \xi_i \geq 0,1 \leq i \leq L \\|\boldsymbol{w} \cdot \phi\left(\boldsymbol{x}_i\right)+b | \geq 1-\xi_i, \xi_i \geq 0, L+1 \leq i \leq L+U\end{gathered}
    $$
    
- This minimization problem is equivalent to minimizing
  
    $$
    \min _{w, b} \frac{1}{2}\|w\|_2^2+C_L \sum_{i=1}^L \max \left\{0,1-y_i\left(\boldsymbol{w} \cdot \phi\left(\boldsymbol{x}_i\right)+b\right)\right\}+C_U \sum_{i=L+1}^{L+U} \max \left\{0,1-\left|\boldsymbol{w} \cdot \phi\left(\boldsymbol{x}_i\right)+b\right|\right\}
    $$
    
- Self-training: iteratively labeling unlabeled data $\boldsymbol{x}\in\mathcal{U}$ by the pseudo
  
    labels of big confidence:$\text { if } \max f(x)>t, \text { then } \hat{y}(x)=\arg \max _y f_y(\boldsymbol{x})$ 
    

# Kernel Method

## Polynomial Kernel

- For $p = 2,d = 2$:
  
    $$
    \boldsymbol{\Phi}: \mathbb{R}^2 \rightarrow \mathbb{R}^3, \quad \boldsymbol{x}=\left(x_1, x_2\right) \rightarrow \boldsymbol{\Phi}(\boldsymbol{x})=\left(x_1^2, x_2^2, \sqrt{2} x_1 x_2\right)
    $$
    
- Kernel function induced by above polynomial basis functions:
  
    $$
    \begin{gathered}\boldsymbol{\Phi}\left(\boldsymbol{x}_1\right) \cdot \boldsymbol{\Phi}\left(\boldsymbol{x}_2\right)=\left(x_{11}^2, x_{12}^2, \sqrt{2} x_{11} x_{12}\right) \cdot\left(x_{21}^2, x_{22}^2, \sqrt{2} x_{21} x_{22}\right) \\=\left(x_{11} x_{21}+x_{12} x_{22}\right)^2=\left(\boldsymbol{x}_1 \cdot \boldsymbol{x}_2\right)^2=k\left(\boldsymbol{x}_1, \boldsymbol{x}_2\right)\end{gathered}
    $$
    
    ![Untitled](5%20SVM%20Theory/Untitled%209.png)
    
- Quadratic feature map for $\boldsymbol{x}=(x_1, x_2,..., x_d)\in\mathbb{R}^d$
  
    $$
    \boldsymbol{\Phi}(\boldsymbol{x})=\left(x_1, \ldots, x_d, x_1^2, \ldots, x_d^2, \sqrt{2} x_1 x_2, \ldots, \sqrt{2} x_i x_j, \ldots, \sqrt{2} x_{d-1} x_d\right)
    $$
    
- But for any $\boldsymbol{x}_1,\boldsymbol{x}_2\in\mathbb{R}^d$ and dot product:
  
    $$
    k\left(\boldsymbol{x}_1, \boldsymbol{x}_2\right)=\boldsymbol{\Phi}\left(\boldsymbol{x}_1\right) \cdot \boldsymbol{\Phi}\left(\boldsymbol{x}_2\right)=\left(\boldsymbol{x}_1 \boldsymbol{x}_2\right)+\left(\boldsymbol{x}_1 \cdot \boldsymbol{x}_2\right)^2
    $$
    

## RBF Kernel: Feature Vector

- Consider RBF kernel (1-dim):、
  
    $$
    k\left(x_1, x_2\right)=\exp \left(-\frac{\left(x_1-x_2\right)^2}{2}\right)
    $$
    
- We claim that $\boldsymbol{\phi}: \mathbb{R}\rightarrow l_2$,defined by
  
    $$
    [\boldsymbol{\Phi}(x)]_j=\frac{1}{\sqrt{j !}} e^{-\frac{x^2}{2}} x^j \text { for } j \in\{0,1, \ldots, \infty\}
    $$
    
    is the infinite-dimensional feature vector induced by the RBF kernel.
    
- $\boldsymbol{\Phi}$ is an element of $l_2$ (Space of square-summable sequences).
  
    $$
    \sum_{j=0}^{\infty}\left\{[\boldsymbol{\Phi}(x)]_j\right\}^2=\sum_{j=0}^{\infty} \frac{1}{j !} e^{-x^2} x^{2 j}=e^{-x^2} \sum_{j=0}^{\infty} \frac{1}{j !} x^{2 j}=1 \leq \infty
    $$
    
    $$
    k\left(x_1, x_2\right)=\boldsymbol{\Phi}\left(x_1\right) \cdot \boldsymbol{\Phi}\left(x_2\right)=\exp \left(-\frac{\left(x_1-x_2\right)^2}{2}\right)
    $$
    

## Kernel Matrix

- How to verify that a function can be used as a kernel function?
Find its basis function Ф? It is too hard for most kernel functions.
- Theorem(Mercer): If $k(\cdot,\cdot)$  is a symmetric function on space $\mathcal{X}\times\mathcal{X}$
  then $k$ is a kernel function
  
    For any input set $(\boldsymbol{x}_1, \boldsymbol{x}_2,..., \boldsymbol{x}_m)$,$\boldsymbol{K}=\left(\begin{array}{ccc}k\left(\boldsymbol{x}_1, \boldsymbol{x}_1\right) & \cdots & k\left(\boldsymbol{x}_1, \boldsymbol{x}_m\right) \\\vdots & \ddots & \vdots \\k\left(\boldsymbol{x}_1, \boldsymbol{x}_m\right) & \cdots & k\left(\boldsymbol{x}_m, \boldsymbol{x}_m\right)\end{array}\right)$
  
    The kernel matrix is **semi-definite**.
  
- For kernel functions $k_1,k_2,...k_s$ and $\gamma_1,\gamma_2,...,\gamma_s>0$
  
    $\sum_{i=1}^S \gamma_i k_i$ is also (multi-)kernel function, because $\sum_{i=1}^S \gamma_i \boldsymbol{K}_i \geq 0$
    

## Soft-SVM: Dual Problem with Basis Function

- Dual Problem of Soft-SVM:
  
    $$
    \begin{gathered}\max _\alpha \sum_{i=1}^n \alpha_i-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j\left(\Phi\left(x_i\right) \cdot \Phi\left(x_j\right)\right) \\\text { s.t. } \sum_{i=1}^n \alpha_i y_i=0 \\0 \leq \alpha_i \leq C, 1 \leq i \leq n\end{gathered}
    $$
    
    - After solving for $\boldsymbol{\alpha}$, we can solve for $\boldsymbol{w}=\sum_{i=1}^n \alpha_i y_i \boldsymbol{\Phi}\left(\boldsymbol{x}_i\right)$
    - Testing: $f(\boldsymbol{x})=\boldsymbol{w} \cdot \boldsymbol{\Phi}(\boldsymbol{x})+b=\sum_{i=1}^n \alpha_i y_i \boldsymbol{\Phi}\left(x_i\right) \cdot \boldsymbol{\Phi}(\boldsymbol{x})+b$
    - During training and testing, only need to compute $\boldsymbol{\Phi}\left(x_i\right) \cdot \boldsymbol{\Phi}(\boldsymbol{x})$.
- Dual Problem of Soft-SVM:
  
    $$
    \begin{gathered}
    \max _\alpha \sum_{i=1}^n \alpha_i-\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j k\left(\boldsymbol{x}_i, \boldsymbol{x}_j\right) \\
    \text { s.t. } \sum_{i=1}^n \alpha_i y_i=0 \\
    0 \leq \alpha_i \leq C, 1 \leq i \leq n
    \end{gathered}
    $$
    
    - After solving for $\boldsymbol{\alpha}$, we can solve for $\boldsymbol{w}=\sum_{i=1}^n \alpha_i y_i \boldsymbol{\Phi}\left(\boldsymbol{x}_i\right)$
    - Testing: $f(\boldsymbol{x})=\boldsymbol{w} \cdot \boldsymbol{\Phi}(\boldsymbol{x})+b=\sum_{i=1}^n \alpha_i y_i k\left(\boldsymbol{x}_i, \boldsymbol{x}\right)+b$
    - During training and testing, only need to compute $\boldsymbol{\Phi}\left(x_i\right) \cdot \boldsymbol{\Phi}(\boldsymbol{x})$