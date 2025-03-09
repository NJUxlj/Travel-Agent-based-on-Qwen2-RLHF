import json
import math
import os
import pickle
import sys
from typing import Dict, List


class BM25:
    EPSILON = 0.25
    PARAM_K1 = 1.5  # BM25算法中超参数
    PARAM_B = 0.6  # BM25算法中超参数
    
    
    def __init__(self, corpus:Dict):
        self.corpus_size = 0  # 文档数量
        self.wordNumsOfAllDoc = 0  # 用于计算文档集合中平均每篇文档的词数 -> wordNumsOfAllDoc / corpus_size
        self.doc_freqs = {}  # 记录每篇文档中查询词的词频
        self.idf = {}  # 记录查询词的 IDF
        self.doc_len = {}  # 记录每篇文档的单词数
        
        self.docContainedWord = {}  # 记录每个单词在哪些文档中出现
        
        self._initialize(corpus)
        
        
    def _initialize(self, corpus:Dict):
        """
            根据语料库构建倒排索引
        """
        # nd = {} # word -> number of documents containing the word
        for index, document in corpus.items():
            self.corpus_size+=1
            self.doc_len[index] = len(document) # 文档的单词数
            self.wordNumsOfAllDoc += len(document) # 所有文档的单词数

            frequencies = {} # 一篇文档中的单词出现的频率
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs[index] = frequencies # 一篇文档中单词出现的频率
            
            
            # 构建词到文档的倒排索引，将每个单词映射到一个文档集合
            for word in frequencies.keys():
                if word not in self.docContainedWord:
                    self.docContainedWord[word]  = set()
                self.docContainedWord[word].add(index)
                    
                
        # 计算 idf
        idf_sum = 0  # 用于计算 IDF 的分母
        negative_idfs = []
        # 计算原始IDF  
        # idf = log((总文档数 - 包含该词文档数 + 0.5) / (包含该词文档数 + 0.5))  

        # # 负值修正策略  
        # average_idf = 总IDF / 词数  
        # eps = EPSILON * average_idf  
        # self.idf[负IDF词] = eps  # 用平均IDF的25%替代负值  
        
        for word in self.docContainedWord.keys():
            doc_nums_contained_word = len(self.docContainedWord[word])
            idf = math.log(self.corpus_size-doc_nums_contained_word + 0.5) - \
                math.log(doc_nums_contained_word + 0.5)
            
            self.idf[word] = idf        
            idf_sum+=idf
            
            if idf < 0:
                negative_idfs.append(word)
        
        
        average_idf = idf_sum / len(self.idf)
        eps = BM25.EPSILON * average_idf
        
        for word in negative_idfs:
            self.idf[word] = eps
            
        
        print("==============================")
        print("self.doc_freqs",[f"doc_id:{k}, " for k, v in self.doc_freqs.items()])
        print("self.idf", self.idf)
        print("self.doc_len", self.doc_len)
        print("self.docContainedWord", self.docContainedWord)
        print("self.corpus_size", self.corpus_size)
        print("========================================")

    
    @property
    def avgdl(self):
        return self.wordNumsOfAllDoc / self.corpus_size
    
    
    
    
    def get_score(self, query:List, doc_index):
        """
        计算查询 q 和文档 d 的相关性分数
        :param query: 查询词列表
        :param doc_index: 为语料库中某篇文档对应的索引
        
        score(D, Q) = Σ IDF(q_i) * [ (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1*(1 - b + b*|D|/avgdl)) ]
        """
        score = 0
        b = BM25.PARAM_B
        k1 = BM25.PARAM_K1
        avgdl = self.avgdl
        
        doc_freqs = self.doc_freqs[doc_index]
        
        for word in query:
            if word not in doc_freqs:
                continue
            score += self.idf[word] * (doc_freqs[word] * (k1+1) / (doc_freqs[word] + k1*(1-b+b*self.doc_len[doc_index]/avgdl)))
        
        return [doc_index, score]
        
    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in self.doc_len.keys()]
        return scores
    
    
    





BM25_ALGORITHM_DESC = """

### BM25算法原理详解

BM25（Best Matching 25）是信息检索领域广泛使用的相关性评分算法，核心思想是通过概率模型评估查询与文档的相关性。其核心公式为：

```
score(D, Q) = Σ IDF(q_i) * [ (f(q_i,D) * (k1 + 1)) / (f(q_i,D) + k1*(1 - b + b*|D|/avgdl)) ]
```

**核心要素**：
1. **IDF（逆文档频率）**：衡量词语区分能力
   - 公式：`IDF(q_i) = log[(N - n(q_i) + 0.5)/(n(q_i) + 0.5) + 1]`
   - 其中N是总文档数，n(q_i)是包含q_i的文档数

2. **TF（词频）调整**：通过参数k1控制词频饱和度
   - 当k1=0时完全忽略词频
   - 典型值范围：1.2~2.0

3. **长度归一化**：通过参数b平衡文档长度影响
   - b=1时完全归一化，b=0时忽略长度
   - 使用`|D|/avgdl`计算相对长度
   
   
   BM25的本质：
        它就是query中每个词的idf分数的累加
   

### 代码类解析

```python
class BM25:
    # 类常量定义
    EPSILON = 0.25  # 负IDF修正系数
    PARAM_K1 = 1.5  # 词频饱和度参数
    PARAM_B = 0.6   # 长度归一化参数

    def __init__(self, corpus: Dict):
        # 初始化数据结构
        self.corpus_size = 0          # 文档总数
        self.wordNumsOfAllDoc = 0     # 语料库总词数
        self.doc_freqs = {}           # {文档ID: {词: 词频}}
        self.idf = {}                 # {词: IDF值}
        self.doc_len = {}             # {文档ID: 词数}
        self.docContainedWord = {}    # {词: 包含该词的文档集合}
```

#### 关键方法解析

1. **倒排索引构建** (`_initialize`)
```python
for index, document in corpus.items():
    # 统计文档长度
    self.doc_len[index] = len(document)
    
    # 构建词频字典
    frequencies = {}
    for word in document:
        frequencies[word] = frequencies.get(word, 0) + 1
    
    # 构建倒排索引
    for word in frequencies:
        self.docContainedWord.setdefault(word, set()).add(index)
```

2. **IDF计算优化**
```python
# 计算原始IDF
idf = log((总文档数 - 包含词数 + 0.5) / (包含词数 + 0.5))

# 负值修正策略
average_idf = 总IDF / 词数
eps = EPSILON * average_idf
self.idf[负IDF词] = eps  # 用平均IDF的25%替代负值
```

3. **相关性计算** (`get_score`)
```python
for word in query:
    if word in 文档词频:
        # 计算TF分量
        tf = doc_freqs[word]
        # 计算长度归一化因子
        norm_factor = 1 - b + b*(文档长度/平均长度)
        # 累加词项得分
        score += idf * (tf*(k1+1)) / (tf + k1*norm_factor)
```

### 参数作用说明

| 参数   | 典型值范围 | 功能说明                                                                 |
|--------|------------|--------------------------------------------------------------------------|
| k1     | 1.2-2.0    | 控制词频饱和度：值越大，高频词影响越大                                   |
| b      | 0.5-0.8    | 长度归一化强度：1表示完全归一化，0禁用长度调整                           |
| EPSILON| 0.2-0.3    | 负IDF修正系数：防止罕见词产生负权重，保持数值稳定性                      |

该实现完整包含了BM25的核心要素，通过预计算倒排索引和IDF值实现高效检索，适合中小规模语料库的实时搜索场景。







"""