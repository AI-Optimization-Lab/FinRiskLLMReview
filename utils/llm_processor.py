import os
import json
import asyncio
import httpx
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import sys
import importlib.util
import pandas as pd

# 获取keywords.py中的关键词
def get_keywords():
    """从keywords.py中获取关键词列表"""
    try:
        # 获取项目根目录
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keywords_path = os.path.join(base_dir, "keywords.py")
        
        # 使用importlib动态导入keywords.py
        spec = importlib.util.spec_from_file_location("keywords_module", keywords_path)
        keywords_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keywords_module)
        
        # 返回keywords字典
        return keywords_module.keywords
    except Exception as e:
        print(f"加载关键词时出错: {str(e)}")
        return {}


class LLMProcessor:
    """LLM处理器，用于调用DeepSeek API进行论文关键词匹配判断"""
    
    def __init__(self, api_key: str = "", api_base: str = "https://api.deepseek.com"):
        """初始化LLM处理器
        
        Args:
            api_key: DeepSeek API密钥
            api_base: DeepSeek API基础URL
        """
        self.api_key = api_key
        self.api_base = api_base
        self.client = None
        self.system_prompt = ""
        self.user_prompt_template = ""
    
    def set_api_key(self, api_key: str) -> None:
        """设置API密钥
        
        Args:
            api_key: DeepSeek API密钥
        """
        self.api_key = api_key
    
    def set_prompts(self, system_prompt: str, user_prompt_template: str) -> None:
        """设置系统提示词和用户提示词模板
        
        Args:
            system_prompt: 系统提示词
            user_prompt_template: 用户提示词模板
        """
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
    
    def load_prompts_from_file(self, file_path: str) -> bool:
        """从文件加载提示词
        
        Args:
            file_path: 提示词文件路径
            
        Returns:
            加载成功返回True，否则返回False
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                self.system_prompt = prompts.get('system_prompt', '')
                self.user_prompt_template = prompts.get('user_prompt_template', '')
                return True
        except Exception as e:
            print(f"加载提示词文件 {file_path} 时出错: {str(e)}")
            return False
    
    def get_client(self) -> httpx.AsyncClient:
        """获取异步HTTP客户端
        
        Returns:
            异步HTTP客户端
        """
        if not self.client or self.client.is_closed:
            self.client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0
            )
        return self.client
    
    def format_user_prompt(self, title: str, abstract: str, keywords: List[str]) -> str:
        """格式化用户提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            keywords: 关键词列表
            
        Returns:
            格式化后的用户提示词
        """
        return self.user_prompt_template.format(
            title=title,
            abstract=abstract,
            keywords="\n".join([f"- {keyword}" for keyword in keywords])
        )
    
    async def call_llm_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """调用DeepSeek API
        
        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            API响应结果
        """
        client = self.get_client()
        
        try:
            # 构造API请求体
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,  # 低温度以获得更确定性的输出
                "max_tokens": 4096
            }
            
            # 发送请求
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            
            return {
                "success": True,
                "content": result["choices"][0]["message"]["content"],
                "full_response": result
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "full_response": None
            }
    
    def parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """解析LLM的响应内容，提取相关关键词和解释
        
        Args:
            response_content: LLM响应内容
            
        Returns:
            解析后的结果字典
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response_content)
            
            # 验证结果格式
            if "relevant_keywords" not in result or "explanations" not in result:
                return {
                    "success": False,
                    "error": "响应格式不正确",
                    "relevant_keywords": [],
                    "explanations": {},
                    "raw_response": response_content
                }
            
            return {
                "success": True,
                "relevant_keywords": result["relevant_keywords"],
                "explanations": result["explanations"],
                "raw_response": response_content
            }
            
        except json.JSONDecodeError:
            # 尝试从文本中提取JSON部分
            try:
                # 查找第一个{和最后一个}之间的内容
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}')
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx+1]
                    result = json.loads(json_str)
                    
                    if "relevant_keywords" not in result or "explanations" not in result:
                        return {
                            "success": False,
                            "error": "响应格式不正确",
                            "relevant_keywords": [],
                            "explanations": {},
                            "raw_response": response_content
                        }
                    
                    return {
                        "success": True,
                        "relevant_keywords": result["relevant_keywords"],
                        "explanations": result["explanations"],
                        "raw_response": response_content
                    }
            except:
                pass
            
            return {
                "success": False,
                "error": "无法解析JSON响应",
                "relevant_keywords": [],
                "explanations": {},
                "raw_response": response_content
            }
    
    async def process_paper(self, title: str, abstract: str, keywords: List[str]) -> Dict[str, Any]:
        """处理单篇论文
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            keywords: 关键词列表
            
        Returns:
            处理结果字典
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "未设置API密钥",
                "relevant_keywords": [],
                "explanations": {},
                "raw_response": None
            }
        
        if not self.system_prompt or not self.user_prompt_template:
            return {
                "success": False,
                "error": "未设置提示词",
                "relevant_keywords": [],
                "explanations": {},
                "raw_response": None
            }
        
        user_prompt = self.format_user_prompt(title, abstract, keywords)
        api_response = await self.call_llm_api(self.system_prompt, user_prompt)
        
        if not api_response["success"]:
            return {
                "success": False,
                "error": api_response["error"],
                "relevant_keywords": [],
                "explanations": {},
                "raw_response": None
            }
        
        parsed_result = self.parse_llm_response(api_response["content"])
        return parsed_result
    
    async def process_batch(self, 
                          papers: List[Dict[str, Any]], 
                          keywords: List[str],
                          on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                          max_concurrent: int = 5,
                          retry_count: int = 3,
                          retry_delay: int = 5) -> List[Dict[str, Any]]:
        """批量处理多篇论文
        
        Args:
            papers: 论文列表，每篇论文包含title和abstract
            keywords: 关键词列表
            on_progress: 进度回调函数，参数为(已处理数量, 总数量, 当前处理结果)
            max_concurrent: 最大并发请求数
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
            
        Returns:
            处理结果列表
        """
        results = []
        sem = asyncio.Semaphore(max_concurrent)
        total = len(papers)
        processed = 0
        
        async def process_with_semaphore(paper, index):
            nonlocal processed
            async with sem:
                for attempt in range(retry_count):
                    try:
                        result = await self.process_paper(
                            paper["title"], 
                            paper["abstract"], 
                            keywords
                        )
                        processed += 1
                        if on_progress:
                            on_progress(processed, total, result)
                        return {
                            "index": index,
                            "paper": paper,
                            "result": result
                        }
                    except Exception as e:
                        if attempt == retry_count - 1:
                            processed += 1
                            error_result = {
                                "success": False,
                                "error": str(e),
                                "relevant_keywords": [],
                                "explanations": {},
                                "raw_response": None
                            }
                            if on_progress:
                                on_progress(processed, total, error_result)
                            return {
                                "index": index,
                                "paper": paper,
                                "result": error_result
                            }
                        await asyncio.sleep(retry_delay)
        
        # 创建所有任务
        tasks = [process_with_semaphore(paper, i) for i, paper in enumerate(papers)]
        
        # 等待所有任务完成
        results_with_index = await asyncio.gather(*tasks)
        
        # 按原始顺序排序结果
        results_with_index.sort(key=lambda x: x["index"])
        results = [item["result"] for item in results_with_index]
        
        return results
    
    def process_dataframe(self, 
                        df: pd.DataFrame, 
                        keywords: List[str],
                        on_batch_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                        batch_size: int = 10,
                        max_concurrent: int = 5) -> pd.DataFrame:
        """处理DataFrame中的论文
        
        Args:
            df: 包含论文的DataFrame
            keywords: 关键词列表
            on_batch_progress: 批次进度回调函数
            batch_size: 批次大小
            max_concurrent: 最大并发请求数
            
        Returns:
            包含处理结果的DataFrame
        """
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        total_rows = len(df)
        result_rows = []
        
        # 分批处理
        for batch_start in range(0, total_rows, batch_size):
            batch_end = min(batch_start + batch_size, total_rows)
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            # 准备批次的论文数据
            papers = []
            for _, row in batch_df.iterrows():
                papers.append({
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "id": row["id"] if "id" in row else None,
                    "year": row["year"] if "year" in row else None,
                    "source": row["source"] if "source" in row else None,
                    "area": row["area"] if "area" in row else None,
                    "method": row["method"] if "method" in row else None
                })
            
            # 处理当前批次
            batch_results = loop.run_until_complete(self.process_batch(
                papers, keywords, on_batch_progress, max_concurrent
            ))
            
            # 将结果添加到列表
            for i, result in enumerate(batch_results):
                row_data = batch_df.iloc[i].to_dict()
                row_data.update({
                    "success": result["success"],
                    "relevant_keywords": json.dumps(result["relevant_keywords"], ensure_ascii=False),
                    "explanations": json.dumps(result["explanations"], ensure_ascii=False),
                    "raw_response": result["raw_response"]
                })
                result_rows.append(row_data)
        
        # 关闭事件循环
        loop.close()
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(result_rows)
        return result_df


if __name__ == "__main__":
    # 测试代码
    api_key = "your_api_key_here"  # 替换为实际API密钥
    processor = LLMProcessor(api_key)
    
    # 测试加载关键词
    keywords_dict = get_keywords()
    print(f"可用的关键词类别: {list(keywords_dict.keys())}")
    
    # 测试加载提示词
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
    default_prompts = os.path.join(prompts_dir, "default.json")
    if os.path.exists(default_prompts):
        processor.load_prompts_from_file(default_prompts)
        print("成功加载默认提示词模板")
    
    # 测试处理单篇论文
    async def test_process_paper():
        result = await processor.process_paper(
            "A Deep Learning Approach for Financial Risk Assessment",
            "This paper proposes a novel approach to financial risk assessment using deep learning techniques. We use convolutional neural networks to analyze market data and predict potential risks.",
            keywords_dict.get("deep learning", [])[:5]  # 只使用前5个关键词进行测试
        )
        print(f"处理结果: {result}")
    
    if api_key != "your_api_key_here":
        import asyncio
        asyncio.run(test_process_paper()) 