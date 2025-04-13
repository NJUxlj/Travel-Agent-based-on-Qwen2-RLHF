# 自然语言生成评估指标详解

## 1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

### ROUGE-N
- **概念**：测量参考文本和生成文本之间的n元语法（n-gram）重叠程度。
- **适用场景**：摘要生成、回答生成质量评估。
- **数学公式**：
  - **Precision**: $\text{ROUGE-N}_p = \frac{\sum_{S \in \text{Refs}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Refs}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$
  - **Recall**: $\text{ROUGE-N}_r = \frac{\sum_{S \in \text{Refs}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \text{Cand}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$
  - **F1-Score**: $\text{ROUGE-N}_f = \frac{2 \times \text{ROUGE-N}_p \times \text{ROUGE-N}_r}{\text{ROUGE-N}_p + \text{ROUGE-N}_r}$
- **变体说明**：
  1. **ROUGE-1**：单个词匹配（unigrams）
     - 测量单词级别的重叠度
     - 适合评估内容召回率，但忽略语法结构
  
  2. **ROUGE-2**：双词组匹配（bigrams）
     - 测量连续两个词的重叠度
     - 更好地捕捉语法流畅度和结构性

### ROUGE-L
- **概念**：基于最长公共子序列(LCS)的评估方法，考虑序列级别的匹配。
- **优势**：自动考虑句子层面的匹配，无需预设n-gram大小。
- **数学公式**：
  - **LCS**：最长公共子序列长度
  - **Precision**: $\text{ROUGE-L}_p = \frac{\text{LCS}(X,Y)}{|Y|}$
  - **Recall**: $\text{ROUGE-L}_r = \frac{\text{LCS}(X,Y)}{|X|}$
  - **F1-Score**: $\text{ROUGE-L}_f = \frac{(1+\beta^2) \times \text{ROUGE-L}_p \times \text{ROUGE-L}_r}{\text{ROUGE-L}_p + \beta^2 \times \text{ROUGE-L}_r}$，其中通常β=1

- **推导思路**：
  1. 找出参考文本和生成文本中的最长公共子序列（顺序一致但可以不连续）
  2. 用该子序列长度分别与参考文本长度和生成文本长度相比，计算召回率和精确度

## 2. BLEU (Bilingual Evaluation Understudy)

- **概念**：最初为机器翻译设计，测量生成文本与一个或多个参考文本的相似度。
- **核心思想**：基于精确度(Precision)的度量，重点关注生成文本中与参考文本匹配的n-gram比例。
- **数学公式**：
  - **Modified Precision**: $p_n = \frac{\sum_{C \in \text{Candidates}} \sum_{\text{gram}_n \in C} \text{Count}_{\text{clip}}(\text{gram}_n)}{\sum_{C' \in \text{Candidates}} \sum_{\text{gram}'_n \in C'} \text{Count}(\text{gram}'_n)}$
    
    其中$\text{Count}_{\text{clip}}$表示截断计数（不超过参考文本中出现的次数）
  
  - **BP (Brevity Penalty)**: $BP = \begin{cases} 
    1 & \text{if } c > r \\
    e^{(1-r/c)} & \text{if } c \leq r
    \end{cases}$
    
    其中c是候选文本长度，r是参考文本长度
  
  - **BLEU**: $BLEU = BP \times \exp(\sum_{n=1}^{N} w_n \log(p_n))$

- **变体说明**：
  1. **BLEU-1** (weights=(1,0,0,0))：只考虑单个词匹配
     - 公式简化为：$BLEU-1 = BP \times p_1$
     - 适合评估词汇覆盖度
  
  2. **BLEU-2** (weights=(0.5,0.5,0,0))：同时考虑unigram和bigram
     - 公式：$BLEU-2 = BP \times p_1^{0.5} \times p_2^{0.5}$
     - 开始关注部分语法和词序
  
  3. **BLEU-4** (weights=(0.25,0.25,0.25,0.25))：考虑1-4元语法
     - 标准BLEU计算方式：$BLEU-4 = BP \times p_1^{0.25} \times p_2^{0.25} \times p_3^{0.25} \times p_4^{0.25}$
     - 更全面评估语法和语义连贯性

- **平滑处理(Smoothing)**：
  - 目的：处理n-gram零匹配问题
  - 方法1 (代码中使用)：为零计数添加微小值，避免完全惩罚
  - 计算：$p_n = \frac{\epsilon + \sum \text{Count}_{\text{match}}}{\epsilon + \sum \text{Count}_{\text{total}}}$，其中ε是小常数

## 3. BERTScore

- **概念**：使用预训练语言模型(如BERT)的词嵌入进行语义相似性评估。
- **优势**：
  - 捕捉语义相似性而非表面形式
  - 更接近人类对文本质量的判断
  - 对同义词和不同表达方式更为宽容

- **计算步骤**：
  1. 使用BERT模型为参考文本和生成文本中的每个token生成上下文化的嵌入表示
  2. 计算两个文本之间token级别的相似度
  3. 通过贪婪匹配找到最佳匹配关系

- **数学公式**：
  - **Token Embedding**: $x_i$表示参考文本的第i个token嵌入，$\hat{x}_j$表示生成文本的第j个token嵌入
  
  - **Cosine Similarity**: $sim(x_i, \hat{x}_j) = \frac{x_i^T \hat{x}_j}{||x_i|| \cdot ||\hat{x}_j||}$
  
  - **Precision**: $P_{BERT} = \frac{1}{|\hat{x}|} \sum_{\hat{x}_i \in \hat{x}} \max_{x_j \in x} sim(x_j, \hat{x}_i)$
  
  - **Recall**: $R_{BERT} = \frac{1}{|x|} \sum_{x_i \in x} \max_{\hat{x}_j \in \hat{x}} sim(x_i, \hat{x}_j)$
  
  - **F1**: $F_{BERT} = 2 \frac{P_{BERT} \cdot R_{BERT}}{P_{BERT} + R_{BERT}}$

- **Baseline Rescaling**：
  - 目的：通过与随机文本比较调整分数分布
  - 公式：$score_{rescaled} = \frac{score - baseline}{1 - baseline}$
  - 代码中使用：`rescale_with_baseline=True`

## 4. 评估指标比较

| 指标 | 优势 | 局限性 | 适用场景 |
|------|------|---------|----------|
| ROUGE | 易于理解，关注召回率，适合摘要 | 只关注词汇重叠，忽略语义 | 摘要评估，基本QA质量 |
| BLEU | 关注精确度，广泛应用，支持多参考 | 不考虑语义相关性，对词序敏感 | 机器翻译，生成任务初步评估 |
| BERTScore | 捕捉语义相似性，对同义表达宽容 | 计算开销大，需要预训练模型 | 语义理解要求高的生成评估 |

## 5. 计算注意事项

1. **标记化(Tokenization)**：
   - 所有指标依赖于文本标记化质量
   - 代码使用NLTK的word_tokenize进行标准化处理
   ```python
   reference_tokens = nltk.word_tokenize(reference.lower())
   ```

2. **归一化处理**：
   - 对ROUGE和BLEU，通常将结果转换为0-1之间的F1分数
   - 对BERTScore，使用baseline rescaling调整分布

3. **多参考处理**：
   - BLEU原生支持多参考文本评估
   - ROUGE可扩展到多参考文本
   - 实现：通常取单个参考与生成文本的最高分数

4. **平均聚合**：
   ```python
   aggregated_results[metric_name] = np.mean(values)
   ```
   - 对所有样本的指标取算术平均值，得到整体性能评估

这些评估指标各有优缺点，通常建议结合多种指标进行综合评估，以获得对生成文本质量更全面的了解。