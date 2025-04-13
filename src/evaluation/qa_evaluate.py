import numpy as np  
from transformers import pipeline  
from datasets import load_dataset  
from rouge_score import rouge_scorer  
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  
from bert_score import score as bert_score  
import torch  
from tqdm import tqdm
import nltk  

# 下载NLTK资源
try:  
    nltk.data.find('tokenizers/punkt')  
except LookupError:  
    nltk.download('punkt')  
    
    
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
import nltk  
from nltk.corpus import stopwords  

# 下载NLTK资源（如果尚未下载）  
try:  
    nltk.data.find('corpora/stopwords')  
except LookupError:  
    nltk.download('stopwords')  



from src.configs.config import (
    DATA_PATH
)


class QAEvaluator():
    
    def __init__(self, model, tokenizer, max_seq_length, eval_dataset=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device  = self.model.device
        self.max_seq_length = max_seq_length
        
        self.eval_dataset = eval_dataset

    def compute_metrics(self, eval_preds):
        """  
        评估模型在QA数据集上的性能  
        
        参数:  
            eval_preds: 评估预测结果，包含predictions和label_ids
        
        返回:  
            包含各种评测指标的字典  
        """  
        assert self.model is not None, "未提供Question-Answer评估需要的模型"
        
        # 使用提供的模型或默认模型  
        model_to_evaluate = self.model
        
        # 从Trainer获取评估数据集  
        if not hasattr(self, "eval_dataset") or self.eval_dataset is None:  
            print("警告：评估数据集未设置，无法计算指标, 加载示例数据 ~~~")  
            eval_dataset = {  
                    "Question": [  
                        "What are the best attractions to visit in Paris?",  
                        "How can I get from London to Edinburgh by train?",  
                        "What is the local cuisine in Thailand?",  
                        "What's the best time to visit Japan?"  
                    ],  
                    "Response": [  
                        "The top attractions in Paris include the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Montmartre.",  
                        "You can take a direct train from London King's Cross to Edinburgh Waverley. The journey takes about 4.5 hours.",  
                        "Thai cuisine features dishes like Pad Thai, Tom Yum Goong, Green Curry, and Mango Sticky Rice.",  
                        "Spring (March to May) and autumn (September to November) are the best times to visit Japan."  
                    ]  
                }  
        
        else:
            eval_dataset = self.eval_dataset 
        
                
        
        # 创建QA pipeline  
        qa_pipeline = pipeline(  
            "text-generation",  
            model=model_to_evaluate,  
            tokenizer=self.tokenizer,  
            max_length=self.max_seq_length,  
            do_sample=False  # 使用贪婪解码以确保可重复性  
        )  
        
        # 初始化指标计算器  
        rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)  
        smoothing = SmoothingFunction().method1  
        
        # 初始化结果存储  
        results = {  
            "rouge1_f1": [],  
            "rouge2_f1": [],  
            "rougeL_f1": [],  
            "bleu1": [],  
            "bleu2": [],  
            "bleu4": [],  
            "bert_score_precision": [],  
            "bert_score_recall": [],  
            "bert_score_f1": []  
        }  
        
        generated_responses = []  
        reference_responses = []  
        
        # 对每个问题生成回答并计算指标  
        for i, (question, reference) in enumerate(zip(eval_dataset["Question"], eval_dataset["Response"])):  
            # 格式化输入  
            formatted_input = f"Question: {question}\nAnswer:"  
            
            # 生成回答  
            generated_output = qa_pipeline(formatted_input)[0]["generated_text"]  
            
            # 提取生成的回答部分（去掉原始问题）  
            if "Answer:" in generated_output:  
                generated_answer = generated_output.split("Answer:", 1)[1].strip()  
            else:  
                generated_answer = generated_output.replace(formatted_input, "").strip()  
            
            # 存储生成的回答和参考回答  
            generated_responses.append(generated_answer)  
            reference_responses.append(reference)  
            
            # 计算ROUGE分数  
            rouge_scores = rouge_scorer_instance.score(reference, generated_answer)  
            results["rouge1_f1"].append(rouge_scores["rouge1"].fmeasure)  
            results["rouge2_f1"].append(rouge_scores["rouge2"].fmeasure)  
            results["rougeL_f1"].append(rouge_scores["rougeL"].fmeasure)  
            
            # 计算BLEU分数  
            reference_tokens = nltk.word_tokenize(reference.lower())  
            generated_tokens = nltk.word_tokenize(generated_answer.lower())  
            
            results["bleu1"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(1, 0, 0, 0),   
                                                smoothing_function=smoothing))  
            results["bleu2"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(0.5, 0.5, 0, 0),   
                                                smoothing_function=smoothing))  
            results["bleu4"].append(sentence_bleu([reference_tokens], generated_tokens,   
                                                weights=(0.25, 0.25, 0.25, 0.25),   
                                                smoothing_function=smoothing))  
            
            # 每处理100个样本打印一次进度  
            if (i+1) % 100 == 0:  
                print(f"已处理 {i+1}/{len(eval_dataset['Question'])} 个样本")  
        
        # 计算BERTScore (批量计算以提高效率)  
        try:  
            P, R, F1 = bert_score(generated_responses, reference_responses, lang="en", rescale_with_baseline=True)  
            results["bert_score_precision"] = P.tolist()  
            results["bert_score_recall"] = R.tolist()  
            results["bert_score_f1"] = F1.tolist()  
        except Exception as e:  
            print(f"计算BERTScore时出错: {e}")  
            # 使用占位值  
            results["bert_score_precision"] = [0.0] * len(generated_responses)  
            results["bert_score_recall"] = [0.0] * len(generated_responses)  
            results["bert_score_f1"] = [0.0] * len(generated_responses)  
        
        # 计算每个指标的平均值  
        aggregated_results = {}  
        for metric_name, values in results.items():  
            aggregated_results[metric_name] = np.mean(values)  
        
        # 添加样本级别的结果，以便详细分析  
        aggregated_results["sample_results"] = {  
            "questions": eval_dataset["Question"],  
            "references": reference_responses,  
            "generated": generated_responses,  
            "metrics": {k: v for k, v in results.items()}  
        }  
        
        # 计算额外的评估指标：问答正确率（如果数据集中包含多个可接受的答案）  
        if "Acceptable_Answers" in eval_dataset:  
            correct_answers = 0  
            for i, (gen_answer, acceptable_answers) in enumerate(zip(generated_responses, eval_dataset["Acceptable_Answers"])):  
                # 检查生成的答案是否与任何可接受的答案匹配  
                if any(self._answer_match(gen_answer, acc_answer) for acc_answer in acceptable_answers):  
                    correct_answers += 1  
            
            aggregated_results["answer_accuracy"] = correct_answers / len(generated_responses)  
        
        return aggregated_results  
    
    
    
    
    
    
    
    def _answer_match(self, generated_answer, reference_answer, threshold=0.7):  
        """  
        检查生成的答案是否与参考答案匹配  
        
        参数:  
            generated_answer: 生成的答案  
            reference_answer: 参考答案  
            threshold: 相似度阈值，超过此值认为匹配  
            
        返回:  
            布尔值，表示是否匹配  
        """  

        
        # 获取停用词  
        stop_words = set(stopwords.words('english'))  
        
        # 文本预处理  
        def preprocess(text):  
            tokens = nltk.word_tokenize(text.lower())  
            return ' '.join([w for w in tokens if w.isalnum() and w not in stop_words])  
        
        processed_gen = preprocess(generated_answer)  
        processed_ref = preprocess(reference_answer)  
        
        # 如果其中一个文本为空，则返回False  
        if not processed_gen or not processed_ref:  
            return False  
        
        # 计算TF-IDF特征  
        vectorizer = TfidfVectorizer()  
        try:  
            tfidf_matrix = vectorizer.fit_transform([processed_gen, processed_ref])  
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]  
            return similarity >= threshold  
        except:  
            # 如果向量化失败（例如，词汇表为空），则返回False  
            return False  