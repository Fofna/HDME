# This is our projet Repository for High Dimensional Matrix Estimation Course
\begin{document}

\maketitle

\begin{abstract}
Recommender systems are algorithms designed to make personalized recommendations to users based on their past behavior, preferences, or demographics. The Netflix Prize was a recommender system competition and a reference to compare different algorithm. In this report, we present \textit{Large-scale Parallel Collaborative Filtering for the Netflix Prize} \cite{als}
\end{abstract}

\tableofcontents 

\section{Recommender systems and the Neflix prize}

Recommender systems have become ubiquitous in e-commerce, online entertainment platforms, and many other domains. For example, Netflix uses a recommender system to suggest TV shows and movies to its users based on their viewing history. Amazon also employs a recommender system to suggest products to its customers based on their purchase history, searches, and browsing behavior. There are several types of recommender systems, including:
\begin{itemize}
    \item Collaborative Filtering: based on the user-item interaction data to find similar users and items, and make recommendations based on that.
    \item Content-Based Filtering: uses features of items and users to make recommendations.
    \item Hybrid Recommender Systems: combines the strengths of both collaborative and content-based filtering methods.
\end{itemize}

The Netflix Prize was a global competition launched by Netflix in 2006, with the goal of improving the accuracy of its recommender system. The challenge was to predict movie ratings for its customers based on their past interactions with the platform. The final winner was determined when a team achieved a 10\% improvement in RMSE compared to the Netflix baseline. The Netflix training dataset consists of 100,480,507 ratings given by 480,189 users to 17,770 movies. Each rating is represented by a quadruplet of the form $(user, movie, date\ of\ grade, grade)$. The qualifying dataset contains more than 2,817,131 triplets of the form $(user, movie, date\ of\ grade)$. This makes the Netflix competition naturally geared toward collaborative filtering.\\

The following is a mathematical formulation of the problem:
Given a matrix $R$ of dimensions $m \times n$ representing the user-item interaction data, where $R_{i,j}$ is the rating given by user $i$ to item $j$ and a missing value means that the user has not rated the item.\\

The goal was to predict the missing values in the matrix $R$ using a recommendation algorithm and minimize the Root Mean Squared Error (RMSE) between the predicted ratings and the actual ratings:
$$RMSE = \sqrt{\frac{\sum_{(i,j)\in T} (R_{i,j} - \hat{R}_{i,j})^2}{|T|}}$$
where $T$ is the set of user-item pairs for which we have both actual and predicted ratings, and $\hat{R}_{i,j}$ is the predicted rating for user $i$ and item $j$.\\

The Netflix Prize was won in 2009 by a team called BellKor's Pragmatic Chaos, who improved the RMSE of Netflix's recommender system by 10.06\%. The winning solution was a combination of several algorithms, including matrix factorization and collaborative filtering techniques.
 
\section{Collaborative filtering and Matrix factorization}
\subsection{\textbf{Collaborative filtering and challenges}}
Collaborative filtering is a method for making recommendations by analyzing the behavior and preferences of a large group of users. The algorithm utilizes a mathematical model to identify relationships between users and items, and then predicts what items a user may be interested in based on their past behavior and the behavior of similar users.\\

This is achieved by creating a user-item matrix, where the rows represent users, the columns represent items, and the entries are the ratings or preferences that users have given to those items. The algorithm then applies a similarity measure such as cosine similarity or Pearson correlation or KNN to find similar users, and finally aggregates their preferences to generate recommendations for a target user. The underlying assumption of the collaborative filtering approach is that if two user share the same opinion on an item, they are more likely to share the same opinion on another. Collaborative filetering faces a number of issues:
\begin{itemize}
    \item Cold start refers to the difficulty in making accurate recommendations for new users or items with limited interaction data. Collaborative filtering algorithms rely on the user-item interaction patterns and with limited interaction data, the system may not have enough information to generate accurate recommendations.
    \item Scalability is a concern when the number of users and items grow larger, leading to increased computation time and memory requirements. Collaborative filtering algorithms often require computing user-user or item-item similarity matrices, which can become infeasible with larger datasets.
    \item Sparsity is another issue in collaborative filtering, as it is common for users to interact with only a small fraction of all available items. This can lead to difficulty in finding similar users or items for making recommendations, and result in less accurate recommendations.
\end{itemize}
To address these challenges, various techniques have been proposed, such as matrix factorization, hybrid methods that combine collaborative filtering with content-based recommendations, and more recent developments like deep learning-based models. Matrix factorization, which is at the core of the technique we present in this report, decomposes the user-item interaction matrix into lower-dimensional representations of users and items, capturing their latent factors.

\subsection{\textbf{Matrix Factorization}}

Given the user-item interaction matrix $R$ of dimensions $m \times n$, we can represent it as the product of two low-rank matrices $U$ and $V$ of dimensions $m \times f$ and $f \times n$ respectively, where $f$ is the number of latent factors:
$$R = U \cdot V$$
Here, each row of $U$ represents a user and each column of $V$ represents an item, and the dot product of a row of $U$ and a column of $V$ gives the predicted rating for that user-item pair.\\

The loss to minimize is the mean squared error between the predicted ratings and the actual ratings:
$$Loss = \frac{1}{t} \sum_{(i,j)\in T} (R_{i,j} - \hat{R}_{i,j})^2$$
where $T$ is the set of user-item pairs for which we have both actual and predicted ratings, $t$ is the cardinality of $T$, and $\hat{R}_{i,j} = U_i \cdot V_j$ is the predicted rating for user $i$ and item $j$, $U_i$ being the $ith$ row of $U$ and $V_j$ the $jth$ column of $V$.\\

However, this simple matrix factorization method can lead to overfitting to the training data and result in poor generalization performance on unseen data. This is because the number of parameters to estimate ($k*m+k*n$) exceeds by far the number of ratings we have $t$.\\

Regularization is used to overcome this issue and impose constraints on the solution to prevent overfitting. Tikhonov regularization, more commonly known as L2 or Ridge regularization, adds a penalty term to the loss function proportional to the sum of squares of the elements of the matrices $U$ and $V$:
$$Loss_{reg} = \frac{1}{t} ||R - \hat{R}||_F^2 + \lambda (||\Gamma_U U||_F^2 + ||\Gamma_V V||_F^2)$$
Where $\Gamma_U$ and $\Gamma_V$ are the regularization matrices for the matrices $U$ and $V$, respectively. $||\cdot||_F$ denotes the Frobenius norm of a matrix and $\lambda$ is a regularization parameter that controls the strength of the penalty.\\

The optimization problem becomes finding the matrices $U$ and $V$ that minimize the regularized loss function $Loss_{reg}$. The goal of the article is 2 fold. First, it presents Alternative-Least-Squares with Weighted-$\lambda$-Regularization, an iterative algorithm to solve the above optimization problem and Tikhonov regularization matrices to eliminate the overfitting problem. Then the article presents a parallel implementation of the algorithm.

\section{Alternative-Least-Squares with Weighted-$\lambda$-Regularization}
\subsection{\textbf{Algorithm and iteration step calculation}}
\begin{algorithm}
\caption{Matrix Factorization for Movie Recommendation}
\begin{algorithmic}[1]
\State $V \gets k \times m$ matrix with small random values
\State $V_1 \gets$ vector of average ratings for each movie
\While {$RMSE$ on probe dataset $>$ 1 bps}
\State $U \gets \arg\min_{U} L_(R, U, V)$
\State $V \gets \arg\min_{V} L_(R, U, V)$
\EndWhile
\end{algorithmic}
\end{algorithm}

The authors opt for the following regularization matrices $\Gamma_U = \text{diag}(n_{u_i})$ and $\Gamma_V = \text{diag}(n_{v_j})$ which they call a weighted-$\lambda$-regularization. $n_{u_i}$ and $n_{v_j}$ are the number of ratings by user $i$ and the number of ratings on item $j$ respectively.\\

This helps to avoid overfitting by placing more weight (shrinking more) on the columns of $U$ and $V$ that correspond to users or movies with more ratings. This means that for users or movies with more ratings, the regularization term is larger, effectively shrinking their columns of $U$ and $V$ more. This in turn reduces the influence of these users or movies in the final prediction, making the model more robust to noise in the data and reducing overfitting.\\

This regularization is also convenient because it always to take advantage of the sparcity. We need only to minimize a sum over the existing ratings:
$$
f(U, V)=\sum_{(i, j) \in I}\left(r_{i j}-\mathbf{u}_i^T \mathbf{v}_j\right)^2+\lambda\left(\sum_i n_{u_i}\left\|\mathbf{u}_i\right\|^2+\sum_j n_{v_j}\left\|\mathbf{v}_j\right\|^2\right)
$$

Alternating between $U$ and $V$ by fixing one and solving for the other allows for a closed form solution. Actually, we can even solve for each row of $U$ or column of $V$ independently which allows parallelization:

$$
 \mathbf{u}_i=A_i^{-1} V_{T_i} R_{u_i}, \quad \forall i
 $$
 $$
 \mathbf{v}_j=B_j^{-1} U_{T_j}^T R_{v_j}, \quad \forall j
$$

With $A_i=V_{T_i} V_{T_i}^T+\lambda n_{u_i} I$, $V_{T_i}$ the sub-matrix of $V$ where we select only items (columns) rated by user $i$ and $R_{u_i}$ is a vector that has the ratings by user $i$\\

And with $B_j=U_{T_j}^T U_{T_j}+\lambda n_{v_j} I$, $U_{T_j}$ the sub-matrix of $U$ where we select only users (rows) who rated item $j$ and $R_{v_j}$ is a vector that has the ratings of item $j$

\subsection{\textbf{Parallel ALS}}

The ability to solve for the minimization of columns/rows independently at each matrix minimization allows the parallelization which is cricial given that in the case of the Netflix Prize competition, $U$ has $480189$ rows and $V$ has $17770$ columns.\\

The authors achieve this using Matlab by instanciating multiple "lab". Each lab stores and updates part of the columns of $U$ and part of the rows of $V$. A "gather" function groups and share the updated matrices for the next matrix minimization.

\section{Results}

The study found that the ALS-WR algorithm does not overfit data as the number of iterations or hidden features increases. The results show that as the number of hidden features increases, the RMSE score improves, reaching its best score of 0.8985 when $k$ is $400~500$. This represents a 5.56\% improvement over Netflix's benchmark, making it one of the top single-method performances.

\section{Analysis}
\subsection{\textbf{Strengths of the ALS-WR Method}}
\begin{itemize}
   \item ALS-WR allows solving the minimization independently for each row/column, which enables parallelization and speeds up calculations (for 480K rows and 17k movies)
   \item The ALS-WR method combines collaborative filtering and matrix factorization techniques to achieve high accuracy in recommendations. The article provides a detailed explanation of the algorithm and its scalability, making it a useful guide for researchers and practitioners.
   \item Its potential applications extend to other recommendation systems such as music or book recommendations, and to other types of data such as images or text.
\end{itemize}
\subsection{\textbf{Limitations of the ALS-WR Method}}
Despite being a popular method for matrix factorization, ALS-WR has several limitations that make it less ideal for some applications.
Indeed :
\begin{itemize}
   \item ALS-WR does not guarantee non-negative factorization of the input matrix, and can easily get stuck in local minima due to its sensitivity to initialization.
   \item Computationally, the method can be expensive and time-consuming to converge, making it unsuitable for large datasets with high computational complexity.
   \item ALS-WR leverages sparsity as it minimizes a sum only indexed by existing ratings, not all possible ratings. The algorithm does not take into account the sparsity of the input matrix, leading to suboptimal results for sparse matrices.
   \item Additionally, ALS-WR provides no interpretability of the latent factors, making it difficult to understand the underlying relationships and patterns in the data.
\end{itemize}
A more effective approach is to combine Non-Negative Matrix Factorization (NMF) with ALS-WR. NMF guarantees non-negative factorization and aids ALS-WR's sensitivity to initialization by serving as a starting point. The combination addresses limitations of both methods, capitalizing on their strengths for improved matrix factorization. NMF ensures non-negative factorization and considers the input matrix's sparsity, while ALS-WR refines the solution and enhances computational efficiency. 
\section{\textbf{Implications}}
\subsection{\textbf{Enhancing ALS with NMF}}
\subsubsection{Definition of NMF}
Given a non-negative User-Item matrix $\mathbf{R} \in \mathbb{R}{+}^{m \times n}$ and integer $k$ of latent factors, we want to find a non-negative matrix $\boldsymbol{U} \in \mathbb{R}{+}^{m \times k}$ (User matrix) and $\boldsymbol{V} \in \mathbb{R}_{+}^{k \times n}$ (Rating matrix) such that :
$$
\frac{1}{2}\|\mathbf{R}-\boldsymbol{U} \boldsymbol{V}\|_F^2
$$
is minimized. \\
The non-negative rank of the Ratings matrix R, 
$rank_{+}(R)$, is the size of the smallest exact 
non-negative factorization $R = UV$, thus we have :
\begin{equation}
\text{rank}(R) \leq \text{rank}_{+}(R) \leq \min{(m, n)}.
\end{equation}
\subsubsection{Some comments}
\begin{itemize}
    \item NMF is not unique :
    If X is nonnegative and with nonnegative inverse, then $UXX^{-1}V$ is equivalent valid decomposition.
    \item It has no order :
    NMF components lack any defined order of importance, with no component being inherently more significant than another.
    \item The method is not hierarchical : 
    The components in a rank-(k+1) decomposition can be distinct from those in a rank-k decomposition.
    \item The algorithm operates in an anti-negative semiring, without subtraction, where each rank-1 component explains a portion of the whole and can result in sparse factors.
\end{itemize} 
NMF has a wide range of applications, including in neuroscience, image understanding, air pollution research, text mining, bioinformatics, microarray analysis, mineral exploration, and weather forecasting.
\subsubsection{Computation}
In the alternating non-negative least squares iteration, elements of $V$ are updated together at first, then all elements of $U$ are updated simultaneously.
$$
\begin{aligned}
V:=\operatorname{argmin}_{V \geq 0}\|R-U V\|_F^2 \\
U:=\operatorname{argmin}_{U \geq 0}\|R-U V\|_F^2
\end{aligned}
$$
The solution for each row of $V$ (or column of $U$) requires solving a non-negative least squares problem, which unfortunately lacks a straightforward closed-form solution.
The non-negative least squares problem has the general form :
minimize $\|R x-b\|^2$ such that $x \geq 0$;
The problem of non-negative least squares is a convex optimization problem that can be tackled by using any constrained optimization solver. One classic method for solving this problem is through active set methods. To develop these methods, the variables are divided into a free set $\mathcal{I}$ and a constrained set $\mathcal{J}$, and after differentiating the norm square, we write the KKT equations in the form :
$$
\begin{aligned}
x_{\mathcal{I}} & =R_{\mathcal{I}}^{\dagger} b & & x_{\mathcal{I}} \geq 0 \\
R_{\mathcal{J}}^T(R x-b) & \geq 0 & & x_{\mathcal{J}}=0
\end{aligned}
$$
The problem of figuring out the partitioning of variables into $\mathcal{I}$ and $\mathcal{J}$ is crucial for computing the solution $x$ via an ordinary least squares solve. A simple approach is to make an initial guess for $\mathcal{I}$ and $\mathcal{J}$, and then improve the guess iteratively by shifting variables between the sets until the optimal solution $x$ is achieved. \\ We start from a non-negative random $x$ and random sets $\mathcal{I}, \mathcal{J}$, then we :

\begin{algorithm}[H]
\caption{Algorithm for Non-Negative Least Squares}
\begin{algorithmic}[1]
\State Compute $p=R_{\mathcal{I}}^{\dagger} b-x$.
\State Compute a new point $x:=x+\alpha p$ by choosing $\alpha \leq 1$ such that it is maximized, but still satisfies the non-negativity of the new point.
\If {$\alpha<1$}
\State Shift the index of the component that turned zero from $\mathcal{I}$ to $\mathcal{J}$.
\State Compute another step.
\ElsIf {$\alpha=1$ and $g_{\mathcal{J}}=R_{\mathcal{J}}^T(R x-b)$ has any negative elements}
\State Relocate the index connected to the component with the greatest negative value of $g_{\mathcal{J}}$ from $\mathcal{J}$ to $\mathcal{I}$.
\State Compute another step.
\Else
\State $g_{\mathcal{J}} \geq 0$ and KKT conditions are satisfied. Stop.
\EndIf
\end{algorithmic}
\end{algorithm}
This approach has a slow convergence due to limited changes to the guess of free variables per iteration, whereas alternate methods rapidly change the free variable set for improved convergence.
\subsection{\textbf{Non Negative Multiplicative Update}}
\subsubsection{Theorical Motivations}
The goal is to decompose the User- Item matrices into two non-negative matrices and where we replace the uniform step size $\alpha_{k}$ of Projected Gradient Descent Iteration with a different one (non-negative) for each entry of M and U using an alternating multiplicative update rule : 
$$
\begin{aligned}
U^{\text {new }} & =\left[U+P \odot\left(R V^T-U V V^T\right)\right]_{+} \\
V^{\text {new }} & =\left[V+P^{\prime} \odot\left(U^T R-U^T U V\right)\right]_{+}, 
\end{aligned}
$$
With $P=U \oslash\left(U V V^T\right)$, and $\quad P^{\prime}=V \oslash\left(U^T U V\right)$ defining the nonnegative scaling matrices.
The $\odot$ and $\oslash$ denotes respectively elementwise multiplication and division. \\
By rewriting $U^{\text {new }}$ and $V^{\text {new }}$, we obtain :
$$
\begin{aligned}
U^{\text {new }} & =P \odot\left(R V^T\right)=U \oslash\left(U V V^T\right) \odot\left(R V^T\right) \\
V^{\text {new }} & =P^{\prime} \odot\left(U^T R\right)=V \oslash\left(U^T U V\right) \odot\left(U^T R\right)
\end{aligned}
$$
Since, the elements of $U$ and $V$ are scaled by non-negative factors at each step, resulting in non-negative outcomes, the step sizes are gradually increased as the elements approach zero, eliminating the need for non-negative projections.
This algorithm is the most popular and one of the best which guarantees fast converge to a local minima.
\begin{algorithm}[H]
\caption{Non Negative Multiplicative Update Algorithm}
\begin{algorithmic}[1]
\State $\mathbf{U} \leftarrow \operatorname{random}(m, k)$
\State $\boldsymbol{V} \leftarrow \operatorname{random}(k, n)$
\Repeat
\State $v_{i j} \leftarrow v_{i j} \frac{\left(\boldsymbol{U}^{\top} \boldsymbol{R}\right){i j}}
{\left(\boldsymbol{U}^{\top} \boldsymbol{V U}\right){i j}+\varepsilon}$
\State $u_{i j} \leftarrow u_{i j} \frac{\left(\mathbf{R} \boldsymbol{V}^{\top}\right){i j}}
{\left(\boldsymbol{U V V}^T\right){i j}+\varepsilon}$
\Until{convergence}
\end{algorithmic}
\end{algorithm}
The factors $\varepsilon$ guard against division by zero.\\
A python version of Algorithms 1, 2 and 3 above  are available here.
\subsubsection{Results}
The RMSE on the test data is xxxx after a learning period about 2min  using Apache Spark. This well justifies the efficiency of the algorithms. See the details of the project here.
\subsection{\textbf{Recommendations for future research}}
An efficient Recommender System has been developed by recent research (2022) to improve recommendation results by considering the multidimensional nature of real-world scenarios. Non Negative Tensor Factorization (NTF) is used to extend Non Negative Matrix Factorization into n-dimensions, integrating context-based information and discovering hidden structures in the data that cannot be captured by a Non-Negative Matrix Factorization (NMF) and its equivalent which only model the relationships in a 2-dimensional matrix structure. Therefore, Non Negative Tensor Factorisation algorithm should be use to build a complete recommender system with rich information.
