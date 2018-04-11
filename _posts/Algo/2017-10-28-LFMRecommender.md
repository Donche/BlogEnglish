---
layout: post
title: Latent Factor Recommender System
category: Algorithm
catalog: true
mathjax: true
tags: 
    - 2017
    - Algorithm
    - RecommenderSystem
---

*Of course, there are many other algorithms besides the neighborhood-based recommender system mentioned earlier, such as Latent Factor Models. Because it was introduced in "Recommender System Practices" [[1]](#1), I found some errors in formulas and codes, and some codes lack explanations. ~~At the same time, because I spend a lot of time debugging parameters~~, so I decided to write a blog to record it*   

# 1. Latent Factor Models
The Latent Factor Models is the most popular research topic in the recommender system field in recent years. The main idea is to connect user interests and items through latent factors.      
In fact, The Latent Factor Models is to classify users and items. According to a category of user interest, the item under the category is recommended to the user. Then the question is, how to classify the user's interests, and How to classify items? How to determine the weight of the item in the category?     
If you ask experts to determine the weight of the classification... No, we cannot artificially classify such a large data set, not even free labor.     

Then what to do?      
David Wheeler once said, All problems in computer science can be solved by another level of indirection[[2]](#2).     
So just add a layer of coefficients between the user and the item. Starting from the data, an algorithm is used to automatically obtain the classification weights of items and users. Not only is it more accurate, it can get reliable weights, and it also reduces the manpower required to mark items.      

# 2. Implementation of LFM   
user u's interest in item i can be calculated through the following equation:   

$$Preference(u,i) = r_{ui} = p_u^Tq_i = \sum_{k=1}^{F}p_{u,k}q_{i,k}$$ 

Among them, k is a latent class, p_uk is the relationship between user u and latent class k, q_ik is the relation between item i and latent class k. Multiplying the two, and we can get the weight between the user and the item.        

## 2.1 Selection of positive and negative samples

In order to obtain the values of p and q, we need to take a number of positive and negative samples for each user and analyze their preferences. For movie scoring problems, it is more convenient to obtain the samples. However, for invisible feedback data sets, that is, we have only the positive sample data set, the choice of negative samples is very important. Through the result of 2011 KDD Cup, Yahoo! Music recommendation system competition, we found that the selection of negative samples needs to follow the following principles [[1]](#1):     
1. Ensure the balance of positive and negative samples for each user.     
2. When sampling negative samples, select items that are popular and users do not have behavior.     

I found the second point logically reasonable. If there is a big hit and I still don't want to see this movie, I guess maybe I really don’t like it. Such popular items that have not been chosen represent one's interest preference more generally than unpopular items. The negative sampling code is as follows:   

```python
def RandomSelectNegativeSample(self, items):
    ret = dict()
    for i in items.keys():
        ret[i] = 1
    n = 0
    for i in range(0, len(items) * 3):
        item = items_pool[random.randint(0, len(items_pool) - 1)]
        if item in ret:
            continue
        ret[item] = 0
        n + = 1
        if n > len(items):
            break
    return ret
```

It should be noted that the items_pool in the code is a list of all items. Because of the repetition, the probability of each item being selected is proportional to the frequency of its occurrence.   

While there is also another way to actually use it, that is, to select the most popular items [[4]](#4) each time, the code is as follows:   

```python
def RandSelectNegativeSamples(items, items_pool):
    ret = dict()
    for i in items.keys():
        ret[i] = 1
    posiSize = len(ret)
    n = 0
    negativeSamples = sorted(items_pool.items(), key = itemgetter(1), reverse = True)
    for item, v in negativeSamples:
        if n > posiSize:
            break
        if item not in ret:
            ret[item] = 0
        n += 1
    return ret
```

The items_pool in the code is a dict that stores all items and thier occurrences. We can get the most popular items by sorting them, then select the number of negative samples that are consistent with the number of positive samples and set the value to 1. However, the result is not as good as I expected, so it has not been adopted.     
In my practice, I also found that it is better to have a negative sample for a uniform area of each item. This seems to contradict the conclusion in the book, yet to be examined...    


## 2.2 Optimization

We can get the exact p and q by optimizing the objective function:    
$$ C = \displaystyle\sum_{(u,i)\in K}(r_{ui} - \hat{r_{ui}})^2  = \displaystyle\sum_{(u,i)\in K}(r_{u,i} - \displaystyle\sum_{k=1}^{K}p_{u,k}q_{i,k})^2 + \lambda ||p_u||^2 + \lambda ||q_i||^2 $$    
The latter two are regular terms to prevent overfitting. If there is some machine learning foundation, you must be very familiar to this form. Gradient descent is the most common method to solve the equation. Each partial derivative of the objective function requires the following function:       

$$\frac{\partial C}{\partial p_{uk}} = 2(r_{ui} - \displaystyle\sum_{k=1}^Kp_{uk}q_{ik})(-q_{ik}) + 2\lambda p_{uk} $$   
$$\frac{\partial C}{\partial q_{ik}} = 2(r_{ui} - \displaystyle\sum_{k=1}^Kp_{uk}q_{ik})(-p_{uk}) + 2\lambda q_{ik} $$        
Using the gradient descent method, the following recurrence formula is obtained:   
$$p_{uk} = p_{uk} - \alpha \frac{\partial C}{\partial p_{uk}}$$   
$$q_{ik} = q_{ik} - \alpha \frac{\partial C}{\partial q_{ik}}$$   
Among them, α is the learning rate, which needs experimentation to determine the exact value. In summary, the code is as follows (you can also use vectorization to speed up, the code is [here](#NegaSample)):    

```python
def LatentFactorModel(user_items, F, N, alpha, flambda):
    users_pool = []
    items_pool = []
    for user, items in user_items.items():
        users_pool.append(user)
        for item in items.keys():
            items_pool.append(item)
    [P, Q] = InitModel(users_pool, items_pool, F)
    for step in range(0,N):
        for user, items in user_items.items():
            samples = RandSelectNegativeSamples(items, items_pool)
            for item, rui in samples.items():
                eui = rui - Predict(P[user], Q[item])
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - flambda * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - flambda * Q[item][f])
        alpha *= 0.9
    return [P, Q]
``` 
In InitModel function, you can initialize each value of p, q to 1------ No! This is of course not acceptable. If they are initialized to the same value, then each user and item will have the same hidden class weight. Therefore, when it is implemented specifically, it can be initialized to a random number between 0 and 1:
```random.random()```      
Another function that need to mentioned is Predict(P[user], Q[item]). Because we need to output the rank value  between 0-1, after summing all the coefficients together, we need to use the sigmoid function to normalize:     
```python
def Predict(Puser, Qitem):
    rank = 0
    for f,puf in Puser.items():
        rank += puf * Qitem[f]
    rank = 1.0/(1+math.exp(-rank)) 
    return rank
```
## 2.3 Recommendation
When recommending items to the user, multiply each of the implicit coefficients of p and q and add them up to obtain the final weight. The code is as follows:   
```python
def Recommend(user, P, Q):
    rank = dict()
    for f, puf in P[user].items():
        for item in Q.keys():
            if item not in rank:
                rank[item] = 0
            rank[item] += puf * Q[item][f]
    return sorted(rank.items(), key = lambda x: x[1], reverse = True)[0:10]
```

# 3. Possible problems
## 3.1 Nan in P and Q
Firstly, I set the parameters as in the book,  F = 100, N = 18, alpha = 0.02, lambda = 0.01, ratio = 1, the result is a catastrophe, the recall rate is about 1 percent, many of the P, Q matrix are Nan. After a long time of debugging, I found out that on a small data set, p and q were diverging... This story tells us that the tuneup still has to be done for each specified case.

At the same time, the result of Nan may also be due to the small amount of data + cold start problems [[3]](#3). After checking it out, I found that there was no such condition in my case.

## 3.2 Negative sample selection

Because each iteration cycle takes a different negative sample, the error function is always maintained at a very high level, but the result does improve with the number of iterations.
I also refer to another method of negative sample selection [[4]](#4). I placed the selection of negative samples out of the loop, that means each iteration has a definite negative sample. The result was found to be much worse, so this method is not recommended. The code is as follows:

```python
def LatentFactorModel(user_items, F, N, alpha, flambda):
    users_pool = []
    items_pool = dict()
    samples = dict()
    for user, items in user_items.items():
        users_pool.append(user)
        for item in items.keys():
            if item not in items_pool:
                items_pool[item] = 0
            items_pool[item] += 1
    for user in users_pool:
        samples[user] = RandSelectNegativeSamples(items, items_pool)
    [P, Q] = InitModel(users_pool, items_pool, F)
    for step in range(0,N):
        for user, items in user_items.items():
            for item, rui in samples[user].items():
                eui = rui - Predict(P[user], Q[item])
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[item][f] - flambda * P[user][f])
                    Q[item][f] += alpha * (eui * P[user][f] - flambda * Q[item][f])
        alpha *= 0.9
    return [P, Q]
```

## 3.3 <span id="NegaSample"></span>Vectorization

Because the code contains a large number of arrays of multiplication and summation, the use of dict + cycle summation is obviously very inefficient, so you can consider using numpy to speed up. Writing P, Q as a numpy ndarray can greatly speed up the program. Experiments have found that merely changing P[user] and Q[item] to one-dimensional arrays can already reduce the computation time by about 7 times. In practice, each item of P and Q needs to be initialized to ```np.random.rand(1,F)```, where F is the number of hidden classes. P, Q iteration code is as follows:

```python
def LatentFactorModel(user_items, F, N, alpha, flambda, falpha):
    users_pool = []
    items_pool = dict()
    #samples = dict()
    for user, items in user_items.items():
        users_pool.append(user)
        for item in items.keys():
            if item not in items_pool:
                items_pool[item] = 0
            items_pool[item] += 1
    [P, Q] = InitModel(users_pool, items_pool, F)
    for step in range(0,N):
        err_sum = 0
        for user, items in user_items.items():
            samples = RandSelectNegativeSamplesNouse(items, list(items_pool.keys()))
            for item, rui in samples.items():
                eui = rui - Predict(P[user], Q[item])
                err_sum += eui
                P[user] += alpha * (eui * Q[item] - flambda * P[user])
                Q[item] += alpha * (eui * P[user] - flambda * Q[item])
        alpha *= falpha
        print(err_sum)
    return [P, Q]
```

It should be noted that because of the use of stochastic gradient descent, the loop of the previous layer should not be vectorized, otherwise it is easy to diverge the overflow.


# 4. Result analysis
Alpha is found important during the tuning process. The number of iterations and the proportion of positive and negative samples are also important. The results are as follows:

F = 100, lambda = 0.01， ratio = 1    

alpha   | N    | recall | precision | coverage  | popularity    | hit number
---     | ---  | ---    | ---       | ---       | ---           |  ---
0.02    | 10   | 6.16%  | 30.63%    | 28.28%    | 7.1996        | 18498
0.02    | 20   | 6.89%  | 34.22%    | 31.07%    | 7.1581        | 20671
0.015   | 10   | 5.62%  | 27.92%    | 25.11%    | 7.2905        | 16865
0.015   | 20   | 6.38%  | 31.67%    | 28.49%    | 7.2492        | 19131

F = 100, lambda = 0.01， ratio = 5   

alpha   | N    | recall | precision | coverage  | popularity    | hit number
---     | ---  | ---    | ---       | ---       | ---           |  ---
0.02    | 10   | 7.18%  | 35.66%    | 12.70%    | 7.1858        | 21539
0.02    | 20   | 1.29%  | 6.42%     | 56.57%    | 1.4448        | 3879
0.015   | 10   | 6.94%  | 34.50%    | 11.40%    | 7.2615        | 20835
0.015   | 20   | 7.86 % | 39.03%    | 16.14%    | 7.2084        | 23575

It can be seen that each performance obtained under the condition of ratio=1 has reached an ideal state, which is similar to that obtained by the previous neighborhood-based algorithm. It is estimated that if you continue to optimize each parameter, the indicator will be further improved.


*Reference Materials*
1. <span id="1"></span>推荐系统实践
2. <span id="2"></span>https://en.wikipedia.org/wiki/David_Wheeler_(British_computer_scientist)?oldformat=true
3. <span id="3"></span>http://zhangyi.space/ji-yu-yin-yu-yi-mo-xing-latent-factor-modelde-dian-ying-tui-jian-xi-tong-ji-qi-zai-sparkshang-de-shi-xian/
3. <span id="4"></span>http://blog.csdn.net/sinat_33741547/article/details/52976391

