---
layout: post
title: PageRank and graph based recommender system
header-style: text
category: Algorithm
catalog: true
mathjax: true
tags: 
    - 2017
    - Algorithm
    - RecommenderSystem
---

*So why PageRank and graph based recommender system? Because the graph based PersonalRank [[1]](#1) is derived from the Topic-Sensitive PageRank [[2]](#2). So if one want to completely understand PersonalRank, it's better to start with the famous PageRank[[3]](#3).*     

# 1. PageRank
According to wikipedia：
>PageRank (PR) is an algorithm used by Google Search to rank websites in their search engine results. PageRank was named after Larry Page,[1] one of the founders of Google. PageRank is a way of measuring the importance of website pages.   

The principle of PageRank is actually very simple: use the link relationship between web pages to determine the importance of a web page, and use the concept of random surfer to avoid the problem of Rank Sink in the Dead End, and at the same time, personalize the web page. [[3]](#3).      

Suppose u is a page, Fu is the page pointed to by u, Bu is the page that points to u, and Nu is the number of pages pointed to by u, equal to |Fu|. c is the normalization factor, then the rank of each page is:   
$$R(u) = c \displaystyle\sum_{v\in B_u}\frac{R(v)}{N_v}$$    

However, consider a simple case: when two or more web pages are linked to each other and there are no other outbound links, the rank values of these web pages will gradually increase toward 1 during the iteration process, while the others approache 0, causing a problem named Rank Sink. To solve this, we assume that users do not just click on the hyperlinks in the webpage, but have a certain probability of randomly browsing some webpages (random surfers) in all webpages. Therefore, the formula for the vector E that considers the random probability of browsing the web page is calculated as follows:     
      
$$R'(u) = c \displaystyle\sum_{v\in B_u}\frac{R'(v)}{N_v} + cE(u)$$     
Written by the matrix is:      
$$R'(u) = c(AR' + E)$$   
*norm 1 of R’ is 1*   
So we have：    
$$R'(u) = c(A + E * 1) * R'$$   
That is, R' is an eigenvector of (A + E*1). If we use another notation [[2]](#2) to define the random surfer probability as alpha, the formula can also be written as:   
$$Rank = (1-\alpha)M * Rank + \alpha p$$   
The introduction of E (or p) not only solves the problem of rank sink, but also provides personalized recommendations for users.   


# 2. Topic-Sensitive PageRank

However, the above algorithm ignores a problem: the user's judgment on the importance of the web page is not the same. Although it is possible to personalize search results using different E, it is difficult to maintain such a large matrix when the number of users increases significantly. Therefore, we've got Topic-Sensitive PageRank [[2]](#2), which uses the topic to maintain several vectors of topics and then associates the relevance between the user and the subject, thereby recommending search results for the user. Topic-Sensitive PageRank is calculated as follows:   
$$Rank = (1-\alpha)M * Rank + \alpha v$$    
Where v is maintained for each topic and records the relationship between all pages and topics. If there are four topics A, B, C, and D, and each topic maintains a vector of v, there are five websites a, b, c, d, e, where b, d is related to topic B, then the $$v_b$$ vector is {0, 0.5, 0, 0.5, 0 }.    
After getting the rank value for each topic, you can get user preferences in various ways and recommend the link to the topic. 

# 3. PersonalRank

Because the PersonalRank algorithm [[1]](#1) in "Recommender System Practice" does not have too much actual implementation content, the iterative algorithm described in the book is basically incapable of solving large data sets such as movielens. But it also introduced a Stanford doctoral dissertation, which has about 168 pages. After checking it out, Python has already sparse matrix-related methods, so I'll just put aside the reinvention of the wheel and refer to the tutorial...

## 3.1 PersonalRank：
The description of PersonalRank in the book "Recommender System Practice" is as follows:    

>Assuming that a personalized recommendation is to be given to the user u, a random surf may be performed on the user item bipartite graph starting from the node vu corresponding to the user u. When walking to any node, we first decide, according to the probability α, whether to continue walking, or stop the walk and start again from the vu node. If it is decided to continue walking, then a node is randomly selected from the nodes pointed to by the current node as a node that walks through the next time. In this way, after many random surfs, the probability that each item node is visited will converge to a number. The weight of the item in the final recommendation list is the access probability of the item node.    

The principle is similar to PageRank. The difference is that the links in PageRank are directional, and the connections between PersonalRank people and objects are undirected or bidirectional. The iteration formula is as follows:   

$$ f(n) = 
\begin{cases}
    \alpha \displaystyle\sum_{v'\in (v)}\frac{PR(v')}{|out(v')|}       & \quad \text{if }(v \neq v_u)\\
    (1- \alpha) + \alpha \displaystyle\sum_{v'\in (v)}\frac{PR(v')}{|out(v')|}   & \quad \text{if }(v = v_u)  
\end{cases}
$$   

or as matrix form：       
  
$$Rank = (1-\alpha)M^T Rank + \alpha r_0$$   
Among them, $$r_0$$ can be regarded as the v vector in Topic-Sensitive PageRank, which means that each random surf has the probability of α returning to the starting node. In the end we got the rank of each item and person as follows:  
$$Rank = \alpha (I- (1-\alpha)M^T)^{-1}R_0 $$     
Or written as a system of linear equations:
$$(I- (1-\alpha)M^T)Rank =\alpha R_0 $$    
Therefore, in order to get the Rank, we only need to solve the above equation, which is much faster than the iterative algorithm.

## 3.2 Solve

In fact, the inversion of large sparse matrix is still a complicated problem, especially when the coefficient matrix is generally not a sparse matrix after calculation. So the above equation can also be considered to be the solve of linear equations. Both of these methods are integrated in the scipy. However, because the system of equations is quite slow to solve, it can be time consuming to solve a large number of users, so I used the data set of [movielens 100k](https://grouplens.org/datasets/movielens/) which is not too large for my pc.


### 3.2.1 Sparse matrix
One way to get a sparse matrix in scipy is to first create a sparse matrix and then fill in the elements. Because the sum of the values of each row is 1 and the probability of each node to the surrounding nodes is equal, the probability of each value is 1/Out(u). The code is as below:   

```python
def makeMatrixA(train, item_users, itemNum,  alpha):
    userNum = len(train)
    M = lil_matrix((userNum + itemNum,userNum +itemNum))
    for item, users in item_users.items():
        val = 1. / len(users)
        for user in users:
            M[userNum + item - 1, user - 1] = val
    for user, items in train.items():
        val = 1. / len(items.keys())
        for item in items.keys():
            M[user - 1, userNum + item - 1] = val
    #Solving a matrix of coefficients
    A = lil_matrix(np.eye(userNum + itemNum)- (1-alpha) * M.T)
    #Inverse matrix
    A = inv(M)
    return A

```


### 3.2.1 Solving linear equations
There is a generalized minimal residual (gmres) method [[5]](#5) for solving sparse matrix linear equations in scipy. It should be noted that b can only be a vector and cannot be a sparse matrix. The code is as follows:
```python
def solveRankLinalg(trainUser, A, user, userNum):
    b = np.zeros((A.shape[0],1))
    b[user - 1] = 1
    r = gmres(A, b, tol=1e-08, maxiter=1)[0][userNum:]
    rank = {}
    for i in range(len(r)):
        if i+1 not in trainUser:
            rank[i + 1] = r[i]
    return sorted(rank.items(), key = lambda x: x[1], reverse = True)[0:10]
```

### 3.2.2 The inverse matrix

To get the inverse matrix of a sparse matrix, one can use the scipy.sparse.linalg.inv() method, but after comparing with matlab, it is found that the accuracy is not very high, and there is no parameter that can adjust the accuracy, so the final result is not as good as the mothod of solving linear equations, yet the speed is several times faster. Since the inverse matrix is obtained, the rank value of each user is the corresponding column of the matrix multiplied by a coefficient. The code is as below:

```python
def invRank(invM, train, userNum, itemNum, items_pool):
    rank = dict()
    for user in range(len(train)):
        rankU = invM[:,user].A
        rank_tmp = dict()
        rankUIndex = sorted(range(len(rankU)-userNum), key=lambda k: rankU[k+userNum], reverse= True)
        for item in rankUIndex:
            if item+1 not in train[user+1]:
                rank_tmp[item+1] = rankU[item][0]
        rank[user + 1] = dict(sorted(rank_tmp.items(), key = lambda x: x[1], reverse = True)[0:10])
    return rank
```



## 3.3 Result
Because the accuracy of the method of inversion matrix is not very high (perhaps because the inverse matrix error?), so I use the method of solving linear equations to get the results of different α：

alpha   | recall    | precision | coverage  | popularity    | hit number
---     | ---       | ---       | ---       | ---           |  ---
0.1     | 9.32%     | 29.63%    | 2.75%     | 5.7243        | 2794
0.3     | 10.45%    | 33.24%    | 4.40%     | 5.6751        | 3135
0.5     | 11.06%    | 35.19%    | 4.95%     | 5.6412        | 3318
0.7     | 11.35%    | 36.09%    | 5.50%     | 5.6249        | 3403
0.9     | 11.51%    | 36.61%    | 5.69%     | 5.6186        | 3452
0.95    | 11.52%    | 36.63%    | 5.69%     | 5.6179        | 3454
1       | 2.70%     | 8.58%     | 2.02%     | 4.6462        | 809


It can be seen that each index achieves the best result when α is taken as 0.9. However, if the results are compared with the collaborative filtering algorithm, although the accuracy and recall rate reach a similar level, we still have a low coverage rate. According to the data obtained in the book "Recommender System Practices", it seems that the overall coverage of personalRank is indeed not very high.


*参考资料*
1. <span id="1"></span>《推荐系统实践》
2. <span id="2"></span>Taher H. Haveliwala,Topic-Sensitive PageRank, 2002, 
3. <span id="3"></span>Page, Lawrence and Brin, Sergey and Motwani, Rajeev and Winograd, Terry (1999) The PageRank Citation Ranking: Bringing Order to the Web. Technical Report. Stanford InfoLab.
4. <span id="4"></span>[PageRank-Wikipedia](https://www.wikiwand.com/en/PageRank)
5. <span id="5"></span>[Generalized minimal residual method-Wikipedia](https://www.wikiwand.com/en/Generalized_minimal_residual_method)
6. [浅析PageRank算法--博客](http://blog.codinglabs.org/articles/intro-to-pagerank.html)
7. [PersonalRank：一种基于图的推荐算法---博客园](http://www.cnblogs.com/zhangchaoyang/articles/5470763.html)