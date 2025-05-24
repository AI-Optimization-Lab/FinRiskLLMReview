import os
import json
import hashlib
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import shutil

# 第二阶段缓存目录路径
STAGE2_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stage2_cache")

class Stage2CacheManager:
    """第二阶段缓存管理器，用于管理金融领域分类的缓存"""
    
    def __init__(self):
        """初始化第二阶段缓存管理器，确保缓存目录存在"""
        self.cache_dir = STAGE2_CACHE_DIR
        self.details_dir = os.path.join(self.cache_dir, "details")
        self.annotations_dir = os.path.join(self.cache_dir, "annotations")
        self.index_file = os.path.join(self.cache_dir, "index.json")
        
        # 确保缓存目录存在
        for directory in [self.cache_dir, self.details_dir, self.annotations_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def generate_cache_key(self, paper_id: str, title: str) -> str:
        """生成缓存键，用于唯一标识一个处理任务
        
        Args:
            paper_id: 论文ID或第一阶段缓存键
            title: 论文标题
            
        Returns:
            缓存键字符串
        """
        # 组合ID和标题
        combined = f"{paper_id}|{title}"
        # 生成MD5哈希作为缓存键
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get_index(self) -> List[Dict[str, Any]]:
        """获取索引文件内容
        
        Returns:
            索引列表，如果不存在则返回空列表
        """
        if not os.path.exists(self.index_file):
            return []
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取索引文件时出错: {str(e)}")
            return []
    
    def save_index(self, index_data: List[Dict[str, Any]]) -> bool:
        """保存索引文件
        
        Args:
            index_data: 索引数据列表
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存索引文件时出错: {str(e)}")
            return False
    
    def add_to_index(self, paper_info: Dict[str, Any]) -> bool:
        """将论文信息添加到索引中
        
        Args:
            paper_info: 论文信息字典，至少包含cache_key字段
            
        Returns:
            添加成功返回True，否则返回False
        """
        index_data = self.get_index()
        
        # 检查是否已存在相同cache_key的记录
        for i, item in enumerate(index_data):
            if item.get('cache_key') == paper_info.get('cache_key'):
                # 更新已存在的记录
                index_data[i] = paper_info
                return self.save_index(index_data)
        
        # 添加新记录
        paper_info['timestamp'] = datetime.now().isoformat()
        index_data.append(paper_info)
        return self.save_index(index_data)
    
    def remove_from_index(self, cache_key: str) -> bool:
        """从索引中移除指定缓存键的记录
        
        Args:
            cache_key: 缓存键
            
        Returns:
            移除成功返回True，否则返回False
        """
        index_data = self.get_index()
        initial_length = len(index_data)
        
        # 过滤掉要删除的记录
        index_data = [item for item in index_data if item.get('cache_key') != cache_key]
        
        # 如果长度没变，说明没有找到要删除的记录
        if len(index_data) == initial_length:
            return False
        
        return self.save_index(index_data)
    
    def save_detail(self, cache_key: str, detail: Dict[str, Any]) -> bool:
        """保存详细结果到单独文件
        
        Args:
            cache_key: 缓存键
            detail: 详细结果字典
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            detail['timestamp'] = datetime.now().isoformat()
            detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(detail, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存详细结果时出错: {str(e)}")
            return False
    
    def get_detail(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取指定缓存键的详细结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            详细结果字典，如果不存在则返回None
        """
        detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
        if not os.path.exists(detail_file):
            return None
        
        try:
            with open(detail_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取详细结果文件时出错: {str(e)}")
            return None
    
    def save_annotation(self, cache_key: str, annotation: Dict[str, Any]) -> bool:
        """保存人工标注结果
        
        Args:
            cache_key: 缓存键
            annotation: 标注结果字典
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            annotation['timestamp'] = datetime.now().isoformat()
            annotation_file = os.path.join(self.annotations_dir, f"{cache_key}.json")
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存标注结果时出错: {str(e)}")
            return False
    
    def get_annotation(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """获取指定缓存键的标注结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            标注结果字典，如果不存在则返回None
        """
        annotation_file = os.path.join(self.annotations_dir, f"{cache_key}.json")
        if not os.path.exists(annotation_file):
            return None
        
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取标注文件时出错: {str(e)}")
            return None
    
    def save_result(self, paper: Dict[str, Any], domain_result: Dict[str, Any]) -> Tuple[str, bool]:
        """保存处理结果，包括索引和详细结果
        
        Args:
            paper: 论文信息字典
            domain_result: 领域分类结果字典
            
        Returns:
            (缓存键, 保存成功标志)
        """
        try:
            # 生成缓存键
            paper_id = paper.get('id', '') or paper.get('cache_key', '')
            title = paper.get('title', '')
            cache_key = self.generate_cache_key(paper_id, title)
            
            # 准备索引数据
            index_item = {
                'cache_key': cache_key,
                'title': title,
                'abstract': paper.get('abstract', '')[:200] + '...',  # 摘要截断
                'year': paper.get('year', ''),
                'source': paper.get('source', ''),
                'area': paper.get('area', ''),
                'method': paper.get('method', ''),
                'stage1_cache_key': paper.get('cache_key', ''),
                'stage1_keywords': paper.get('relevant_keywords', []),
                'application_domains': domain_result.get('application_domains', []),
                'timestamp': datetime.now().isoformat()
            }
            
            # 准备详细结果数据
            detail_data = {
                'paper': paper,
                'domain_result': domain_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存索引和详细结果
            index_saved = self.add_to_index(index_item)
            detail_saved = self.save_detail(cache_key, detail_data)
            
            return cache_key, index_saved and detail_saved
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
            return "", False
    
    def has_processed(self, paper_id: str, title: str) -> bool:
        """检查论文是否已处理
        
        Args:
            paper_id: 论文ID或第一阶段缓存键
            title: 论文标题
            
        Returns:
            是否已处理
        """
        cache_key = self.generate_cache_key(paper_id, title)
        detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
        return os.path.exists(detail_file)
    
    def delete_result(self, cache_key: str) -> bool:
        """删除指定缓存键的结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            删除成功返回True，否则返回False
        """
        try:
            success = True
            
            # 从索引中删除
            index_success = self.remove_from_index(cache_key)
            
            # 删除详细结果文件
            detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
            if os.path.exists(detail_file):
                os.remove(detail_file)
            else:
                success = False
            
            # 删除标注文件
            annotation_file = os.path.join(self.annotations_dir, f"{cache_key}.json")
            if os.path.exists(annotation_file):
                os.remove(annotation_file)
            
            return success and index_success
        except Exception as e:
            print(f"删除结果时出错: {str(e)}")
            return False
    
    def clear_all_results(self) -> bool:
        """清空所有结果
        
        Returns:
            清空成功返回True，否则返回False
        """
        try:
            # 删除所有详细结果文件
            if os.path.exists(self.details_dir):
                shutil.rmtree(self.details_dir)
                os.makedirs(self.details_dir)
            
            # 删除所有标注文件
            if os.path.exists(self.annotations_dir):
                shutil.rmtree(self.annotations_dir)
                os.makedirs(self.annotations_dir)
            
            # 清空索引文件
            self.save_index([])
            
            return True
        except Exception as e:
            print(f"清空所有结果时出错: {str(e)}")
            return False
    
    def get_results_by_filter(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据筛选条件获取结果
        
        Args:
            filter_criteria: 筛选条件字典
            
        Returns:
            符合条件的结果列表
        """
        index_data = self.get_index()
        filtered_results = []
        
        for item in index_data:
            matches = True
            for key, value in filter_criteria.items():
                if key in item:
                    # 对于列表类型的字段，检查是否有交集
                    if isinstance(item[key], list) and isinstance(value, list):
                        if not set(value).intersection(set(item[key])):
                            matches = False
                            break
                    # 对于字符串等标量类型，检查是否相等
                    elif item[key] != value:
                        matches = False
                        break
            
            if matches:
                # 获取详细信息
                detail = self.get_detail(item.get('cache_key', ''))
                filtered_results.append({
                    'index': item,
                    'detail': detail
                })
        
        return filtered_results
    
    def get_stage1_cache_manager(self):
        """获取第一阶段缓存管理器实例
        
        Returns:
            第一阶段缓存管理器实例
        """
        from .cache_manager import CacheManager
        return CacheManager()
    
    def load_stage1_results(self, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """加载第一阶段处理结果作为第二阶段的输入
        
        Args:
            filter_criteria: 筛选条件字典
            
        Returns:
            第一阶段结果列表
        """
        stage1_cache_manager = self.get_stage1_cache_manager()
        
        # 获取第一阶段的所有结果
        if filter_criteria is None:
            filter_criteria = {}
        
        stage1_results = stage1_cache_manager.get_results_by_filter(filter_criteria)
        
        # 转换格式
        processed_results = []
        for result in stage1_results:
            metadata = result.get('metadata', {})
            stage1_result = result.get('result', {})
            
            # 只处理成功的结果
            if stage1_result.get('success'):
                # 确保关键词列表是列表而非字符串
                relevant_keywords = stage1_result.get('relevant_keywords', [])
                if isinstance(relevant_keywords, str):
                    try:
                        relevant_keywords = json.loads(relevant_keywords)
                    except:
                        relevant_keywords = []
                
                processed_item = {
                    'id': metadata.get('id', ''),
                    'title': metadata.get('title', ''),
                    'abstract': metadata.get('abstract', ''),
                    'year': metadata.get('year', ''),
                    'source': metadata.get('source', ''),
                    'area': metadata.get('area', ''),
                    'method': metadata.get('method', ''),
                    'cache_key': metadata.get('cache_key', ''),
                    'relevant_keywords': relevant_keywords,
                    'stage1_timestamp': metadata.get('timestamp', '')
                }
                
                # 检查是否已处理过
                if not self.has_processed(processed_item.get('id', '') or processed_item.get('cache_key', ''), 
                                         processed_item.get('title', '')):
                    processed_results.append(processed_item)
        
        return processed_results
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """获取领域统计信息
        
        Returns:
            统计信息字典
        """
        index_data = self.get_index()
        
        # 初始化统计数据
        stats = {
            'total': len(index_data),
            'domain_counts': {
                'derivatives_pricing': 0,
                'financial_risk': 0,
                'portfolio_management': 0,
                'multiple': 0,
                'none': 0
            },
            'yearly_domain_counts': {},
            'method_domain_counts': {}
        }
        
        # 统计各个领域的数量
        for item in index_data:
            domains = item.get('application_domains', [])
            
            # 计算领域计数
            if 'None' in domains or not domains:
                stats['domain_counts']['none'] += 1
            elif len(domains) > 1:
                stats['domain_counts']['multiple'] += 1
                # 同时也计入每个具体的领域
                for domain in domains:
                    if domain != 'None':
                        domain_key = domain.lower().replace(' ', '_')
                        if domain_key in stats['domain_counts']:
                            stats['domain_counts'][domain_key] += 1
            else:
                domain_key = domains[0].lower().replace(' ', '_')
                if domain_key in stats['domain_counts']:
                    stats['domain_counts'][domain_key] += 1
            
            # 按年份统计
            year = str(item.get('year', 'unknown'))
            if year not in stats['yearly_domain_counts']:
                stats['yearly_domain_counts'][year] = {
                    'derivatives_pricing': 0,
                    'financial_risk': 0,
                    'portfolio_management': 0,
                    'none': 0
                }
            
            # 按方法统计
            method = item.get('method', 'unknown')
            if method not in stats['method_domain_counts']:
                stats['method_domain_counts'][method] = {
                    'derivatives_pricing': 0,
                    'financial_risk': 0,
                    'portfolio_management': 0,
                    'none': 0
                }
            
            # 更新年份和方法的领域计数
            if 'None' in domains or not domains:
                stats['yearly_domain_counts'][year]['none'] += 1
                stats['method_domain_counts'][method]['none'] += 1
            else:
                for domain in domains:
                    if domain != 'None':
                        domain_key = domain.lower().replace(' ', '_')
                        if domain_key in stats['yearly_domain_counts'][year]:
                            stats['yearly_domain_counts'][year][domain_key] += 1
                        if domain_key in stats['method_domain_counts'][method]:
                            stats['method_domain_counts'][method][domain_key] += 1
        
        return stats
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """将处理结果导出为DataFrame
        
        Returns:
            包含处理结果的DataFrame
        """
        index_data = self.get_index()
        
        if not index_data:
            return pd.DataFrame()
        
        # 将索引数据转换为DataFrame
        df = pd.DataFrame(index_data)
        
        # 如果application_domains是列表，将其转换为字符串
        if 'application_domains' in df.columns:
            df['application_domains'] = df['application_domains'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # 如果stage1_keywords是列表，将其转换为字符串
        if 'stage1_keywords' in df.columns:
            df['stage1_keywords'] = df['stage1_keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        return df 