---
layout: post
title: Neighborhood-based Recommender System
category: Algorithm
catalog: true
mathjax: true
tags: 
    - 2017
    - Algorithm
    - RecommenderSystem
---

# 1. Neighborhood-based algorithm

The neighborhood-based algorithm can be divided into two categories, one is user-based collaborative filtering algorithm, and the other is item-based collaborative filtering algorithm. Both of these algorithms are already mature, so this blog is just written as notes. The data set used is the 1M data set provided by MovieLens and can be downloaded from [this site](https://grouplens.org/datasets/movielens/). Load the data set and divide each user's watch movie by 7:3 into a training set and a test set.    
*This blog is based on the book "Recommended System Practice" [[1]](#1)*

# 2. User-based collaborative filtering algorithm
This algorithm is mainly divided into two parts:    
1. Calculate user similarity    
2. Generate a recommendation list for users based on their similarity    
 
## 2.1 Calculate user similarity
Assuming there are users u and v, the similarity between them is:  
$$\omega_{uv}=\frac{|N(u)\bigcap N(v)|}{|N(u)\bigcup N(v)|}$$   
Or use cosine similarity:   
$$\omega_{uv}=\frac{|N(u)\bigcap N(v)|}{\sqrt{|N(u)||(v)|}}$$   

According to these two formulae, we need to calculate the numerator and denominator separately. If we calculate them respectively, when the user volume becomes large, most of the users do not have common favorite items, and the calculation efficiency will be very low. Therefore, we count the inverted list of an item-user, and then traverse the inverted list to count the collection of items of interest between each two users. The code is as follows:  

```python
def UserSimilarity(train):
    #inverted list
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    #user matrix
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N[u] = N.get(u,0) + 1
            for v in users:
                if u == v:
                    continue
                if u not in C:
                    C[u] = dict()
                #C[u][v] = C[u].get(v,0) + 1
                C[u][v] = C[u].get(v,0) + 1 / math.log(1 + len(users))
    #user's similarity
    W = dict()
    for u, related_users in C.items():
        if u not in W:
            W[u] = dict()
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

```
In fact, the above code has already counted N(u) by the way, so the user's similarity is obtained.

So can we go to the next step?   
no, not yet.   

Now suppose that some users like to watch movies (of course). They have all watched dozens of popular movies such as "Star Wars", "The Professional", "The Shawshank Redemption" etc. ... So are their interests so similar? Under more possibilities, some of them are fans of Tarkovsky, some are with Nolan, some Hitchcock and some Marvels. In case Ishii’s fans are recommended to Hitchcock’s film and cause great psychological harm to the user, we can say that the recommendation system failed.    

So how to avoid these popular movies from excessively affecting the calculation of user interest similarity? So easy, just reduce the weight on the famous movies. The formula is as follows:   
$$\omega_{u,v} = \frac{\sum_{i\in N(u)\bigcap N(v)} \frac{1}{log1 + |N(i)|}}{\sqrt{|N(u)||N(v)|}}$$   

## 2.2 Recommendation
After getting the K users that are most similar with the target user u, they can recommend items that are of interest to these users but not heard by the target user u. The weight of each item i for user u can be calculated using the following formula:    
$$p(u,i)=\displaystyle\sum_{v\in S(u,K)\bigcap N(i)} \omega_{u,v}r_{v,i} $$   

All r is equal to 1 if a implicit feedback data is used. If you also want to use the score data, you can use the score as a weight (have not tried). The results can obtained by sorting the score and selecting the first ten items (Top10)     

```python
def Recommend(user, train, W, K):
    rank = dict()
    interacted_items = train[user]
    for v, wuv in sorted(W[user].items(), key=itemgetter(1), reverse=True)[0:K]:
        for i, rvi in train[v].items():
            if i in interacted_items:
                continue
            if i not in rank:
                rank[i] = 0
            rank[i] += wuv * rvi
    return sorted(rank.items(), key = itemgetter(1), reverse = True)[0:10]
```

## 2.3 Result Test
As mentioned in the previous section, the most important measures of a recommended algorithm are the following parameters:
* Precision and recall rate
* popularity
* coverage
Cross-validation can be used during testing, which is more accurate and time-consuming. I test directly with the test data. The code is as follows:
```python
def PrecisionRecall(test, train, W, K):
    hit = 0
    n_recall = 0
    n_precision = 0
    recommend_items = set()
    all_items = set()
    item_popularity = dict()
    #Count all items and their occurrences
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                all_items.add(item)
                item_popularity[item] = 0
            item_popularity[item] += 1
    popularity = 0
    n = 0
    for user, items in test.items():
        rank = Recommend(user, train, W, K)
        hitkeys = set(dict(rank).keys()).intersection(items)
        hit += len(hitkeys)
        n_recall += len(items)
        n_precision += len(rank)
        for item, pui in rank:
            recommend_items.add(item)
            popularity += math.log(1 + item_popularity[item])
            n += 1
    coverage = len(recommend_items) / (len(all_items) * 1.0)
    popularity /= n * 1.0
    return [hit / (1.0 * n_recall), hit / (1.0 * n_precision), hit, n_recall, n_precision, coverage, popularity]
```
With different K, the following results are obtained:         

K   | recall| precision | coverage  | popularity    | hit number
--- | ---   | ---       | ---       | ---           |  ---
5   | 5.72% | 28.40%    | 55.03%    | 6.5460        | 17154
10  | 6.70% | 33.28%    | 44.76%    | 6.7169        | 20102
20  | 7.54% | 37.48%    | 34.42%    | 6.8582        | 22637
40  | 8.05% | 40.00%    | 26.16%    | 6.9771        | 24160
80  | 8.46% | 42.02%    | 19.89%    | 7.0811        | 25378


It can be seen that the recommendation system has the highest precision and recall rate with K = 80, but coverage and popularity are not ideal, especially coverage.        

# 3. Item-based collaborative filtering algorithm      

It is so similar to user-based collaborative filtering algorithm:     
1. Calculate item similarity      
2. Generate a recommendation list for the user based on the similarity of the item and the user's historical behavior     

## 3.1 Calculate item similarity

Assuming items u and v, the similarity between them is:      
$$\omega_{uv}=\frac{|N(u)\bigcap N(v)|}{|N(u)\bigcup N(v)|}$$   
Or use cosine similarity:      
$$\omega_{uv}=\frac{|N(u)\bigcap N(v)|}{\sqrt{|N(u)||(v)|}}$$     
As above, we first count the inverted list of a user-item, and then traverse the inverted list to count the similar set of every two items. Since our dataset itself is a user-item arrangement, we can omit this part of the processing. The code is as follows:        
```python
def ItemSimilarity(train):
    C = dict()
    N = dict()
    for u, items in train.items():
        for i in items.keys():
            N[i] = N.get(i,0) + 1
            for j in items.keys():
                if i == j:
                    continue
                if i not in C:
                    C[i] = dict()
                C[i][j] = C[i].get(j,0) + 1 / math.log(1 + len(items) * 1.0)
    #Calculate similarity matrix
    W = dict()
    for i,related_items in C.items():
        W[i] = dict()
        for j, cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W
```
Similarly, we still need to take some extra actions to the super-active users. Several users who've seen a large number of films easily make many unrelated cool items appear similar. The penalty method is the same as what we did to the previous user-based collaborative filtering algorithm.     

## 3.2 Item recommendation       
User u's interest in item i is calculated by the following formula (which is the same as the previous one):
$$p(u,i)=\displaystyle\sum_{v\in S(i,K)\bigcap N(u)} \omega_{i,v}r_{u,v} $$      

One of the benefits of item-based collaborative recommendations is that you can give a reason for the recommendation: based on your favorite item XXXX, we recommend you such items:...     

And, Karypis found that if the ItemCF similarity matrix is normalized by the maximum value, the recommendation accuracy can be improved. The research shows that if the item similarity matrix w has been obtained, the similarity after normalization can be obtained by the following formula:     
$$\omega_{ij}' = \frac{\omega_ij}{max_j \omega_{ij}}$$     


## 3.3 Parameter test

I ran the algorithm with the same parameters and got the following result:    

K   | recall| precision | coverage  | popularity    | hit number
--- | ---   | ---       | ---       | ---           |  ---
5   | 7.70% | 38.24%    | 21.34%    | 7.0742        | 23097
10  | 7.86% | 39.03%    | 18.33%    | 7.1771        | 23573
20  | 7.85% | 39.02%    | 16.15%    | 7.2462        | 23570
40  | 7.71% | 38.32%    | 14.75%    | 7.2805        | 23148
40  | 7.55% | 37.53%    | 13.39%    | 7.2929        | 22666


As can be seen from the table, the precision and recall rate peaked when K = 10. The coverage and popularity decrease monotonically with the increase of K.      

# 4. Analysis      

## 4.1 The difference in the amount of data
For example, in websites such as news websites where content is updated very quickly, item-based collaborative filtering algorithms cause a large amount of calculations and occupy a large amount of space to store item relationship matrices, and performance is not so good. It is also unrealistic to update such a large matrix every day. At this time we can recommend items based on the similarity between users.       
For another example, in the movie recommendation system, the movie update is very slow and the number may be far less than the number of users. The recommendation based on the item is very reasonable. And shopping websites, for example, are the same.      

## 4.2 Time difference
Take another example of a news website: as a very time-sensitive application, outdated news is rarely seen. At the same time, the website generates a large amount of new information every day. It is difficult to achieve this matrix alone. On the contrary, the user's growth is not so much, and updating the user's similarity is more feasible.      
For another example, in a shopping site, users do not need to select a large number of newly-listed products, and the user's buying interest rarely changes drastically within a short period of time, so that the use of item-based recommendations becomes the first choice.     


# 5. Accelerate with multiprocessing
First, cite the description of the other blog [[2]](#2):     
>Multithreading in Python is not really multithreading. If you want to make full use of the resources of multicore CPUs, you need to use multiple processes in most cases in Python. Python provides a very easy way to use multiprocessing package. All we need is to define a function, and Python will take care of everything else. With this package, you can easily convert from a single process to concurrent execution. Multiprocessing supports child processes, communications and data sharing, different forms of synchronization and it also provides components such as Process, Queue, Pipe, and Lock.        
>When using Python for system management, especially when operating multiple file directories at the same time or controlling multiple hosts remotely, parallel operation can save a lot of time. When the number of objects to be operated is not large, multiple processes in the multiprocessing process can be used directly to generate multiple processes.      
>*Pool* can provide a specified number of processes. When a new request is submitted to the pool, if the pool is not yet full, a new process will be created to execute the request. But if the number of processes in the pool having reached the specified maximum value, the request waits until a process in the pool is finished before creating a new process to it.      

The pool of multiprocessing can automatically generate multi-process python applications. It's easy to use, we only need to change a few lines. The first one is of course``` import multiprocessing ```          
Because Pool must run in __main__, you must add the ```if __name__ == "__main__"``` line. Then you need to call the function and pass the parameters to Pool.map(), the code is as follows:      
```python
pool = multiprocessing.Pool(2)
rank_res = pool.map(Recommend,inp)
pool.close()
pool.join()
```
The above code creates two processes and passes each value in the inp to the Recommend function. Then it closes the process pool and waits for all processes to finish.     
Here we need to pay a little attention to modifying the input and output of the Recommend function. Since this function is called outside of the loop, all results are stored in rank_res, which is a list and needs to be manually converted to dict.    


The effect is of course very significant. Using ItemCF, when N = 5, the total computation time used without multi-processes is 1205s. Simultaneous calculations are performed using 4 processes. The required time is reduced to 552s.


*Reference Materials*
1. <span id="1"></span>《推荐系统实践》 项亮
2. <span id="2"></span>http://www.cnblogs.com/kaituorensheng/p/4445418.html   
3. <span id="3"></span>http://www.cnblogs.com/whatisfantasy/p/6440585.html


# Appendix
n_recall 300086 n_precision 60400      
ItemCF    
7：3    
5：recall 0.05% precision 0.25% hit 153 coverage 19.18% popularity 2.2626 归一化   
5 recall: 5.72%  precision: 28.40  hit: 17154  coverage: 55.03 popularity: 6.5460   
10 recall: 6.70	precision: 33.28 hit: 20102	coverage: 44.76	popularity: 6.7169   
20 recall: 7.54	precision: 37.48 hit: 22637	coverage: 34.42	popularity: 6.8582   
40 recall: 8.05	precision: 40.00 hit: 24160	coverage: 26.16	popularity: 6.9771   
80 recall: 8.46	precision: 42.02 hit: 25378	coverage: 19.89	popularity: 7.0811   
3:1   
5: recall 7.70% precision 38.24% hit 23097 coverage 21.34% popularity 7.0742    
10：recall 7.86% precision 39.03% hit 23573 coverage 18.33% popularity 7.1771   
20: recall 7.85% precision 39.02% hit 23570  coverage 16.15% popularity 7.2462   
40: recall 7.71% precision 38.32% hit 23148 coverage 14.75% popularity 7.2805   
80: recall 7.55% precision 37.53% hit 22666 coverage 13.39% popularity 7.2929   

UserCF   
5:  recall 6.14% precision 25.20% hit 15224 coverage 54.33% popularity 6.6178   
10: recall 7.32% precision 30.04% hit 18145 coverage 43.21% popularity 6.7914   
20: recall 8.26% precision 33.91% hit 20482 coverage 34.31% popularity 6.9309   
40: recall 8.96% precision 36.79% hit 22223 coverage 26.33% popularity 7.0461   
80: recall 9.27% precision 38.06% hit 22993 coverage 20.66% popularity 7.1460   
160:recall 9.32% precision 38.26% hit 23112 coverage 15.44% popularity 7.2343   
