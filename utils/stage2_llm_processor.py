import os
import sys
import json
import asyncio
import httpx
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
import traceback

class Stage2LLMProcessor:
    """第二阶段LLM处理器，用于判断论文是否应用于金融领域"""
    
    def __init__(self, api_key: str = "", api_base: str = "https://api.deepseek.com"):
        """初始化第二阶段LLM处理器
        
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
    
    def format_user_prompt(self, title: str, abstract: str, stage1_keywords: List[str]) -> str:
        """格式化第二阶段用户提示词
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            stage1_keywords: 第一阶段识别的关键词列表
            
        Returns:
            格式化后的用户提示词
        """
        # 在使用format方法前，先将模板中的JSON示例中的花括号转义
        template = self.user_prompt_template.replace("{", "{{").replace("}", "}}")
        
        # 然后恢复真正的占位符
        template = template.replace("{{title}}", "{title}")
        template = template.replace("{{abstract}}", "{abstract}")
        template = template.replace("{{stage1_keywords}}", "{stage1_keywords}")
        
        return template.format(
            title=title,
            abstract=abstract,
            stage1_keywords=", ".join(stage1_keywords) if stage1_keywords else "无"
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
            error_details = traceback.format_exc()
            return {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "content": None,
                "full_response": None
            }
    
    def parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """解析LLM的响应内容，提取应用领域和判断理由
        
        Args:
            response_content: LLM响应内容
            
        Returns:
            解析后的结果字典
        """
        try:
            # 检查并清理可能的markdown代码块
            content = response_content.strip()
            if content.startswith("```json") or content.startswith("```"):
                # 移除markdown代码块标记
                content = content.replace("```json", "").replace("```", "").strip()
            
            # 尝试解析JSON响应
            result = json.loads(content)
            
            # 验证结果格式
            if "application_domains" not in result or "justification" not in result:
                return {
                    "success": False,
                    "error": "响应格式不正确",
                    "application_domains": ["None"],
                    "justification": "无法从响应中解析出所需的字段",
                    "raw_response": response_content
                }
            
            # 规范化领域名称（确保首字母大写）
            domains = []
            for domain in result["application_domains"]:
                if domain.lower() == "none":
                    domains.append("None")
                elif domain.lower() == "derivatives pricing" or domain.lower() == "衍生品定价":
                    domains.append("Derivatives Pricing")
                elif domain.lower() == "financial risk" or domain.lower() == "financial risk management" or domain.lower() == "金融风险" or domain.lower() == "金融风险管理":
                    domains.append("Financial Risk")
                elif domain.lower() == "portfolio" or domain.lower() == "portfolio management" or domain.lower() == "投资组合" or domain.lower() == "投资组合管理":
                    domains.append("Portfolio Management")
                else:
                    domains.append(domain)  # 保留原始名称
            
            return {
                "success": True,
                "application_domains": domains,
                "justification": result["justification"],
                "raw_response": response_content
            }
        
        except json.JSONDecodeError:
            # 如果不是JSON格式，尝试从文本中提取信息
            response_content = response_content.strip()
            
            # 简单处理：寻找关键词
            domains = []
            if "衍生品定价" in response_content or "derivatives pricing" in response_content.lower():
                domains.append("Derivatives Pricing")
            if "金融风险" in response_content or "financial risk" in response_content.lower():
                domains.append("Financial Risk")
            if "投资组合" in response_content or "portfolio" in response_content.lower():
                domains.append("Portfolio Management")
            
            # 如果没有找到任何领域关键词
            if not domains:
                domains = ["None"]
            
            return {
                "success": True,
                "application_domains": domains,
                "justification": "通过文本分析提取的领域（非JSON格式响应）",
                "raw_response": response_content
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "application_domains": ["None"],
                "justification": "解析响应时出错",
                "raw_response": response_content
            }
    
    async def process_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """处理单篇论文的领域分类
        
        Args:
            paper: 论文信息字典，包含标题、摘要和第一阶段关键词
            
        Returns:
            处理结果字典
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "未设置API密钥",
                "application_domains": ["None"],
                "justification": "无法进行处理，因为未设置API密钥",
                "raw_response": None
            }
        
        if not self.system_prompt or not self.user_prompt_template:
            return {
                "success": False,
                "error": "未设置提示词",
                "application_domains": ["None"],
                "justification": "无法进行处理，因为未设置提示词",
                "raw_response": None
            }
        
        # 从论文信息中提取所需字段
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        stage1_keywords = paper.get("relevant_keywords", [])
        
        # 格式化用户提示词
        user_prompt = self.format_user_prompt(title, abstract, stage1_keywords)
        
        # 调用API
        api_response = await self.call_llm_api(self.system_prompt, user_prompt)
        
        if not api_response["success"]:
            return {
                "success": False,
                "error": api_response["error"],
                "application_domains": ["None"],
                "justification": f"API调用失败: {api_response.get('error', '未知错误')}",
                "raw_response": None
            }
        
        # 解析响应
        parsed_result = self.parse_llm_response(api_response["content"])
        return parsed_result
    
    async def process_batch(self, 
                          papers: List[Dict[str, Any]], 
                          on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                          max_concurrent: int = 5,
                          retry_count: int = 3,
                          retry_delay: int = 5) -> List[Dict[str, Any]]:
        """批量处理多篇论文
        
        Args:
            papers: 论文列表，每篇论文包含title、abstract和stage1_keywords
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
                        result = await self.process_paper(paper)
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
                                "application_domains": ["None"],
                                "justification": f"处理失败: {str(e)}",
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
    
    def process_papers(self, 
                      papers: List[Dict[str, Any]], 
                      on_progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
                      max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """处理多篇论文（同步接口）
        
        Args:
            papers: 论文列表
            on_progress: 进度回调函数
            max_concurrent: 最大并发请求数
            
        Returns:
            处理结果列表
        """
        # 创建事件循环
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 执行批量处理
            results = loop.run_until_complete(self.process_batch(
                papers, on_progress, max_concurrent
            ))
            return results
        finally:
            # 关闭事件循环
            loop.close() 