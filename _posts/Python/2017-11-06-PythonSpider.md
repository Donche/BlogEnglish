---
layout: post
title: Get More Than 80,000 Albums Using Handwritten Web Crawler
header-style: text
category: Spider
catalog: true
tags: 
    - 2017
    - Python
    - Spider
---

*This project is inspired by another similar work[[1]](#1). In addition, I also refered to a book about spider. I intended to do it with Scrapy, a powerful web crawling framework. But it turns that my first try was a good one and it got more than 50,000 albums for me. At the same time it was not banned by the web site (probably because it just crawled too slow. Anyway I'll try next time to make it fast enough to get banned...). Then I thought it may be a good one and useful for this small project*     


This aim of this project was to get information of the albums as much as possible, including the name of the album, name of the artist, music genre, type of compilation(album, EP, single...), release time, ratings and the number of people who rated. Some of the information is missing, which needed to be replaced by None. The web site I used was Douban, one of the most influential web 2.0 websites in China. The crawl begins at my recent favorite collection *Guruguru Brain Wash*. New urls can be retrieved in the recommandation area(Those who like *Guruguru Brain Wash* also like ...).      

Untill now, I've got information on more than 80,000 albums. Since data base was used to to store all the information(albums and urls), theoretically, I can continue whenever I want. And because the recommandations are mainly over 7.0, so what we got were all high rated albums.   


# 1. Architecture


About the general architecture of a web crawler, I refered to wikipedia, which can be represented as follows:   
![img](https://raw.githubusercontent.com/Donche/en/master/_posts/Python/SpiderStructure.png)   

Among them, the text and metadata is the content to be obtained, the downloader is responsible for downloading the web page, the parser is responsible for parsing the downloaded content, and the URL manager is responsible for managing the link (the link that has been crawled, the link waiting to be crawled, and the failed links, etc...) The Scheduler is responsible for scheduling the entire crawler.    

The process of specific reptile work is as follows:    
1. The scheduler queries for links that are not crawled    
2. Obtain a link if there is one, otherwise, finish the crawler application     
3. The downloader downloads the web page according to the obtained link, and the scheduler passes the downloaded web page to the parser    
4. The parser parses the downloaded data to get new urls and valuable data   
5. The scheduler passes the obtained urls to the url manager   
6. The scheduler passes the valuable data obtained to the database   
7. Jump to 1   
  

# 2. Scheduler

The scheduler is mainly responsible for coordinating the work of various parts and determining the working order of the entire crawler. The code is as follows:   

```python
class Spider(object):
    def __init__(self):
        #Initialization
        self.mongo = mongoDBThing.MongoDBThing()
        self.downloader = Downloader.Downloader()
        self.parser = Parser.Parser()

    def craw(self, root_url):

        count = 1
        self.mongo.add_new_url(root_url)
        while self.mongo.has_new_url():
            try:
                new_url = self.mongo.get_new_url()
                print('crawl %d : %s' % (count, new_url))
                html_cont = self.downloader.download(new_url)
                if html_cont == 404:
                    self.mongo.add_404_url(new_url)
                else:
                    print('parsing')
                    new_urls, new_data = self.parser.parse(new_url, html_cont)
                    print('adding new urls')
                    self.mongo.add_new_urls(new_urls)
                    print('collecting data')
                    self.mongo.collect_data(new_data)
                if count == 50000:
                    break
                count += 1
            except:
                print('craw failed')
        self.mongo.output()

if __name__ == "__main__":
    obj_spider = Spider()
    url_start = 'https://music.douban.com/subject/26590388/'
    obj_spider.craw(url_start)

```


# 3. Database

Why do we need a database in a crawler without storing the data directly in memory or writing files?    


Of course it is because of convenience. We can crawl the web while looking through the information we've got, or set breakpoints, do not worry about the unexpected crash. In a word, its robustness is much better.     
So I used mongoDB to store all the urls and crawled content. There are many tutorials on mongoDB that you can refer to. Before working on the crawler, remember to open the mongoDB service. If it's the first time, you also need to create the corresponding database and collection in the shell before you can use it in python.   

The database is mainly responsible for storing all the valuable data obtained. What also related to the database is the url manager, which needs to complete the following major tasks:   

* Add new links   
* Check if there is a link to be crawled    
* Get new links   

Therefore, there are three types of links that the database needs to store: new links, old links(crawled), and failed links. Every time the scheduler gets a new link, it needs to move this link from new links to old links. The Python code is as follows:   

```python
class MongoDBThing(object):
    def __init__(self):
        client = pymongo.MongoClient('localhost', 27017)
        db = client.MyMusic
        self.newUrlsCol = db.newUrls
        self.oldUrlsCol = db.oldUrls 
        self.notFoundUrls = db.notFoundUrls
        self.music = db.music
        
    def add_new_url(self, url):
        if url is None:
            return
        if self.newUrlsCol.find({'url':url}).count() == self.oldUrlsCol.find({'url':url}).count() == 0:
            self.newUrlsCol.insert({'url': url})

    def add_new_urls(self, urls):
        if urls is None or len(urls) == 0:
            return
        for url in urls:
            self.add_new_url(url)
            
    def has_new_url(self):
        return self.newUrlsCol.find().count() != 0
    
    def get_new_url(self):
        urlDoc = self.newUrlsCol.find_one()
        self.newUrlsCol.remove(urlDoc)
        self.oldUrlsCol.insert(urlDoc)
        return urlDoc['url']
    
    def add_404_url(self, url):
        self.notFoundUrls.insert({'url':url})
         
    def collect_data(self, data):
        if data is None: 
            return
        self.music.insert(data)

```

# 4. Parser

The parser is the most complex part of the web crawler. Because it needs to be designed based on the specific web pages. And the parser also needs to be redesigned each time the web page changes.      
Here I mainly use the *beautifulsoup* and *re* packages to parse web pages. *Beautifulsoup* is a very powerful python package. The built-in founction *find()* can quickly locate any area of a web page by id or class. It also supports regular expression search.     
Generally, the parser consists of three parts: web parsing, getting new links, and obtaining valuable data. The data parsed by the web page is passed directly to the latter two parts for processing. The first part is as follows:   

```python
    def parse(self, page_url, html_cont):
        if page_url is None or html_cont is None:
            print('page_url is none')
            return
        soup = BeautifulSoup(html_cont, 'html.parser')
        print('getting new urls')
        new_urls = self._get_new_urls(soup)
        print('getting new data')
        new_data = self._get_new_data(page_url, soup)
        return new_urls, new_data
```

## 4.1 Getting new links
Press F12 on the web page to open the chrome development tool, and then I found the recommended part of the watercress. The entire recommended area is in a class = "content clearfix" part, as shown below:    
![img](https://raw.githubusercontent.com/Donche/Donche.github.io/master/_posts/Python/doubanF12.jpg)   

The links for each of the recommended albums are in the hyperlinks tab and are in the format  "https://music.douban.com/subject/" so we can use all hyperlinks within this area that satisfy this condition as new Link to use. code show as below:    

```python
def _get_new_urls(self, soup):
    new_urls = set()
    recommend = soup.find('div', class_='content clearfix')
    links = recommend.find_all('a', href=re.compile(r"https://music\.douban\.com/subject/\d+/$")) 
    for link in links:
        new_url = link['href']
        new_urls.add(new_url)
    return new_urls
```

## 4.2 Obtaining data
The method for obtaining these information is the same as above. For example, the album name is found as follows:
![img](https://raw.githubusercontent.com/Donche/Donche.github.io/master/_posts/Python/doubanAlbumName.jpg)
The first h1 within the wrapper is the album name. And it's the same for scores and the number of people. Others are quite simple, after confirming that all the information is in "info", we can simply use regular expression. The code is as below:   

```python
def _get_new_data(self, page_url, soup):
    res_data = {}
    res_data['url'] = page_url
    print('parse url success')
    try:
        res_data['AlbumName'] = soup.find('div', id='wrapper').h1.text.strip()
        res_data['score'] = soup.find('strong', class_='ll rating_num').string
        res_data['sco_num'] = soup.find('div',class_='rating_sum').a.span.text
        info = soup.find('div', id='info')
        if info.find(text=re.compile('表演者')):
            res_data['performer'] = info.find(text=re.compile('表演者')).next_element.text.strip()
        else:
            res_data['performer'] = 'None'
        if info.find(text=re.compile('流派')):
            res_data['genre'] = info.find(text=re.compile('流派')).next_element.strip()
        else:
            res_data['genre'] = 'None'
        if info.find(text=re.compile('专辑类型')):
            res_data['type'] = info.find(text=re.compile('专辑类型')).next_element.strip()
        else:
            res_data['type'] = 'None'
        if info.find(text=re.compile('发行时间')):
            res_data['time'] = info.find(text=re.compile('发行时间')).next_element.strip()
        else:
            res_data['time'] = 'None'
    except:
        print('parse data failed')
    finally:
        return res_data
```
The reason for so many *if else* is that some of the information may not be available. If not, beautifulsoup naturally cannot find the corresponding content. If we call next_element on a None, it will throw an exception. In doing so, even if one of the information is missing, we can continue to parse the next message, ensuring that no information will be missed.   

# 5. Downloader

The downloader is mainly responsible for downloading the web content, so it is relatively simple. However, many websites have anti-crawler mechanisms. When a large number of visits are detected within a short period of time, they will refuse to access and throw a 403 error. There are many tutorials to deal with these kind of problems, [such as this](http://www.shenjian.io /blog/?p=275). Because my spider is rather slow (without multithreading, averaging about two pages per second), and no cookies have been set, so I have got tens of thousands of pages and it still works. (I'll add multithreading and try again later) . But even if the Douban did not banned my web crawler, I still added 403 processing steps in a simple way: every time I get a refuse, the crawler wait for another half a minute. The code is shown as below:   

```python
def _download(self, url):
    if url is None:
        print('url None')
        return None
    try:  
        print('Requesting')
        headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.79 Safari/537.36',
        'Accept-Language':' zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Connection':'keep-alive',
        'referer':'baidu.com'}
        opener = urllib.request.build_opener()
        headall = []
        for key,value in headers.items():
            item = (key,value)
            headall.append(item)
        opener.addheaders = headall
        urllib.request.install_opener(opener)
        print('Opening url')
        response = urllib.request.urlopen(url, timeout = 10)
        print('checking attributes')
    except urllib.error.HTTPError as e:
        print('error: ' + str(e))
        if e.code == 403:
            return 403
        elif e.code == 404:
            return 404
        else:
            return
    if response.getcode() != 200:
        print('get_new_url failed')
        return None
    return response.read()

def download(self, url):
    count = 1
    sleeptime = 30
    while True:
        res = self._download(url)
        if res == 404:
            return 404
        elif res == 403:
            if count > 30:
                return
            print('waiting 403, waiting time:',sleeptime * count)
            sleep(sleeptime * count)
            count += 1
        else:
            return res
```


*Reference Materials*
1. <span id="1"></span> http://blog.csdn.net/xuelabizp/article/details/51068441
2. <span id="2"></span> 《精通Python 网络爬虫》 韦玮
3. <span id="3"></span> http://www.imooc.com/article/15028