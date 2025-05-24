import os
import json
import hashlib
import glob
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import shutil
from pathlib import Path

# 缓存目录路径
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

class CacheManager:
    """缓存管理器，用于管理LLM处理结果的缓存"""
    
    def __init__(self):
        """初始化缓存管理器，确保缓存目录存在"""
        self.cache_dir = CACHE_DIR
        self.results_dir = os.path.join(self.cache_dir, "results")
        self.metadata_dir = os.path.join(self.cache_dir, "metadata")
        self.annotations_dir = os.path.join(self.cache_dir, "annotations")
        self.keywords_dir = os.path.join(self.cache_dir, "keywords")  # 添加关键词列表目录
        
        # 确保缓存目录存在
        for directory in [self.cache_dir, self.results_dir, self.metadata_dir, 
                         self.annotations_dir, self.keywords_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 索引文件路径
        self.index_file = os.path.join(self.cache_dir, "index.json")
        # 详细信息目录
        self.details_dir = os.path.join(self.cache_dir, "details")
        # 加载数据缓存
        self.loaded_data_file = os.path.join(self.cache_dir, "loaded_data.json")
        
        # 索引缓存
        self._index_cache = None
    
    def _get_index(self) -> List[Dict[str, Any]]:
        """获取索引文件内容
        
        Returns:
            索引列表，如果不存在则返回空列表
        """
        # 使用缓存避免频繁读取
        if self._index_cache is not None:
            return self._index_cache
            
        if not os.path.exists(self.index_file):
            self._index_cache = []
            return []
        
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self._index_cache = json.load(f)
                return self._index_cache
        except Exception as e:
            print(f"读取索引文件时出错: {str(e)}")
            self._index_cache = []
            return []
    
    def _save_index(self, index_data: List[Dict[str, Any]]) -> bool:
        """保存索引文件
        
        Args:
            index_data: 索引数据列表
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            # 更新缓存
            self._index_cache = index_data
            return True
        except Exception as e:
            print(f"保存索引文件时出错: {str(e)}")
            return False
    
    def _add_to_index(self, item_info: Dict[str, Any]) -> bool:
        """将信息添加到索引中
        
        Args:
            item_info: 信息字典，至少包含cache_key字段
            
        Returns:
            添加成功返回True，否则返回False
        """
        index_data = self._get_index()
        
        # 检查是否已存在相同cache_key的记录
        for i, item in enumerate(index_data):
            if item.get('cache_key') == item_info.get('cache_key'):
                # 更新已存在的记录
                index_data[i] = item_info
                return self._save_index(index_data)
        
        # 添加新记录
        item_info['timestamp'] = datetime.now().isoformat()
        index_data.append(item_info)
        return self._save_index(index_data)
    
    def _remove_from_index(self, cache_key: str) -> bool:
        """从索引中移除指定缓存键的记录
        
        Args:
            cache_key: 缓存键
            
        Returns:
            移除成功返回True，否则返回False
        """
        index_data = self._get_index()
        initial_length = len(index_data)
        
        # 过滤掉要删除的记录
        index_data = [item for item in index_data if item.get('cache_key') != cache_key]
        
        # 如果长度没变，说明没有找到要删除的记录
        if len(index_data) == initial_length:
            return False
        
        return self._save_index(index_data)
    
    def generate_cache_key(self, title: str, method: str = "", source: str = "") -> str:
        """生成缓存键，用于唯一标识一个处理任务
        
        Args:
            title: 论文标题
            method: 处理方法，可选
            source: 数据来源，可选
            
        Returns:
            缓存键字符串
        """
        # 组合标题、方法和来源
        combined = f"{title}|{method}|{source}"
        # 生成MD5哈希作为缓存键
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def generate_cache_key_from_abstract(self, title: str, abstract: str, keywords: List[str]) -> str:
        """旧版接口，生成缓存键，用于唯一标识一个处理任务
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            keywords: 关键词列表
            
        Returns:
            缓存键字符串
        """
        combined = f"{title}|{abstract}|{'|'.join(sorted(keywords))}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """根据缓存键获取缓存的结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的结果字典，如果不存在则返回None
        """
        cache_file = os.path.join(self.results_dir, f"{cache_key}.json")
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取缓存文件 {cache_file} 时出错: {str(e)}")
            return None
    
    def save_result(self, metadata: Dict[str, Any], result: Dict[str, Any]) -> Tuple[str, bool]:
        """保存处理结果，包括索引和详细结果
        
        Args:
            metadata: 元数据字典
            result: 处理结果字典
            
        Returns:
            (缓存键, 保存成功标志)
        """
        try:
            # 生成缓存键
            title = metadata.get('title', '')
            method = metadata.get('method', '')
            source = metadata.get('source', '')
            cache_key = self.generate_cache_key(title, method, source)
            
            # 准备索引数据
            index_item = {
                'cache_key': cache_key,
                'title': title,
                'abstract': metadata.get('abstract', '')[:200] + '...' if len(metadata.get('abstract', '')) > 200 else metadata.get('abstract', ''),
                'year': metadata.get('year', ''),
                'source': source,
                'area': metadata.get('area', ''),
                'method': method,
                'relevant_keywords': result.get('relevant_keywords', []),
                'timestamp': datetime.now().isoformat()
            }
            
            # 准备详细结果数据
            detail_data = {
                'metadata': metadata,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存索引
            index_saved = self._add_to_index(index_item)
            
            # 保存详细结果
            detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
            try:
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(detail_data, f, ensure_ascii=False, indent=2)
                detail_saved = True
            except Exception as e:
                print(f"保存详细结果时出错: {str(e)}")
                detail_saved = False
            
            return cache_key, index_saved and detail_saved
        except Exception as e:
            print(f"保存结果时出错: {str(e)}")
            return "", False
    
    def save_cached_result(self, metadata, results, cache_key=None):
        """兼容旧版API的方法，将结果保存到缓存文件中
        
        Args:
            metadata: 元数据字典
            results: 处理结果字典
            cache_key: 缓存键，如果为None则自动生成
            
        Returns:
            缓存键
        """
        if not cache_key:
            title = metadata.get('title', '')
            method = metadata.get('method', '')
            source = metadata.get('source', '')
            cache_key = self.generate_cache_key(title, method, source)
        
        # 转换为新格式并保存
        self.save_result(metadata, results)
        
        # 为兼容性保留旧格式文件
        try:
            old_cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(old_cache_file, 'w', encoding='utf-8') as f:
                # 准备旧格式数据
                old_format = {**results, 'metadata': metadata}
                json.dump(old_format, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存旧格式缓存文件时出错: {str(e)}")
        
        return cache_key
    
    def has_cached_result(self, title, method="", source=""):
        """检查是否已有缓存结果
        
        Args:
            title: 论文标题
            method: 处理方法，可选
            source: 数据来源，可选
            
        Returns:
            是否有缓存结果
        """
        cache_key = self.generate_cache_key(title, method, source)
        
        # 检查新版格式
        detail_file = os.path.join(self.details_dir, f"{cache_key}.json")
        if os.path.exists(detail_file):
            return True
        
        # 检查旧版格式
        old_cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        return os.path.exists(old_cache_file)
    
    def get_all_processed_items(self) -> List[Dict[str, Any]]:
        """获取所有已处理的项目的元数据
        
        Returns:
            元数据字典列表
        """
        metadata_files = [f for f in os.listdir(self.metadata_dir) if f.endswith('.json')]
        metadata_list = []
        
        for filename in metadata_files:
            try:
                with open(os.path.join(self.metadata_dir, filename), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata_list.append(metadata)
            except Exception as e:
                print(f"读取元数据文件 {filename} 时出错: {str(e)}")
        
        # 按时间戳排序
        metadata_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return metadata_list
    
    def get_results_by_filter(self, filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据筛选条件获取处理结果
        
        Args:
            filter_criteria: 筛选条件字典，例如 {'area': 'financial_risk', 'method': 'machine learning'}
            
        Returns:
            符合条件的结果和元数据字典列表
        """
        all_metadata = self.get_all_processed_items()
        filtered_metadata = []
        
        for metadata in all_metadata:
            matches = True
            for key, value in filter_criteria.items():
                if key in metadata and metadata[key] != value:
                    matches = False
                    break
            
            if matches:
                cache_key = metadata.get('cache_key')
                if cache_key:
                    result = self.get_cached_result(cache_key)
                    if result:
                        filtered_metadata.append({
                            'metadata': metadata,
                            'result': result
                        })
        
        return filtered_metadata
    
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
            print(f"读取标注文件 {annotation_file} 时出错: {str(e)}")
            return None
    
    def save_loaded_data(self, data: pd.DataFrame, source: str, file_paths: List[str]) -> bool:
        """保存加载后的数据到缓存
        
        Args:
            data: 加载的DataFrame数据
            source: 数据来源
            file_paths: 加载的文件路径列表
        
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            # 保存数据为parquet格式（更高效）
            data_file = os.path.join(self.cache_dir, "last_loaded_data.parquet")
            data.to_parquet(data_file, index=False)
            
            # 保存元数据
            metadata = {
                "source": source,
                "file_paths": file_paths,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": len(data)
            }
            metadata_file = os.path.join(self.cache_dir, "last_loaded_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存加载数据到缓存时出错: {str(e)}")
            return False
    
    def load_last_data(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """加载上次保存的缓存数据和元数据
        
        Returns:
            (数据DataFrame, 元数据字典)，无缓存时返回(None, None)
        """
        data_file = os.path.join(self.cache_dir, "last_loaded_data.parquet")
        metadata_file = os.path.join(self.cache_dir, "last_loaded_metadata.json")
        
        if not os.path.exists(data_file) or not os.path.exists(metadata_file):
            return None, None
        
        try:
            # 加载数据
            data = pd.read_parquet(data_file)
            # 加载元数据
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return data, metadata
        except Exception as e:
            print(f"加载缓存数据时出错: {str(e)}")
            return None, None
    
    def get_all_annotations(self) -> Dict[str, Dict[str, Any]]:
        """获取所有标注结果
        
        Returns:
            以缓存键为键的标注结果字典
        """
        annotation_files = [f for f in os.listdir(self.annotations_dir) if f.endswith('.json')]
        annotations = {}
        
        for filename in annotation_files:
            cache_key = filename.replace('.json', '')
            try:
                with open(os.path.join(self.annotations_dir, filename), 'r', encoding='utf-8') as f:
                    annotations[cache_key] = json.load(f)
            except Exception as e:
                print(f"读取标注文件 {filename} 时出错: {str(e)}")
        
        return annotations
    
    def save_dataframe(self, df: pd.DataFrame, name: str) -> str:
        """保存DataFrame到缓存目录
        
        Args:
            df: 要保存的DataFrame
            name: 保存的文件名前缀
            
        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.csv"
        file_path = os.path.join(self.cache_dir, filename)
        
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        return file_path
    
    def get_saved_dataframes(self) -> List[str]:
        """获取所有保存的DataFrame文件路径
        
        Returns:
            文件路径列表
        """
        return [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir) if f.endswith('.csv')]
    
    def load_dataframe(self, file_path: str) -> Optional[pd.DataFrame]:
        """从文件加载DataFrame
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的DataFrame，如果失败则返回None
        """
        try:
            return pd.read_csv(file_path, encoding='utf-8-sig')
        except Exception as e:
            print(f"加载DataFrame文件 {file_path} 时出错: {str(e)}")
            return None
    
    def save_keyword_list(self, name: str, keywords: List[str]) -> bool:
        """保存关键词列表
        
        Args:
            name: 关键词列表名称
            keywords: 关键词列表
            
        Returns:
            保存成功返回True，否则返回False
        """
        try:
            keywords_data = {
                "name": name,
                "keywords": keywords,
                "timestamp": datetime.now().isoformat()
            }
            filename = f"{name.replace(' ', '_')}.json"
            file_path = os.path.join(self.keywords_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(keywords_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存关键词列表时出错: {str(e)}")
            return False
    
    def get_keyword_list(self, name: str) -> Optional[List[str]]:
        """获取指定名称的关键词列表
        
        Args:
            name: 关键词列表名称
            
        Returns:
            关键词列表，如果不存在则返回None
        """
        filename = f"{name.replace(' ', '_')}.json"
        file_path = os.path.join(self.keywords_dir, filename)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("keywords", [])
        except Exception as e:
            print(f"读取关键词列表时出错: {str(e)}")
            return None
    
    def get_all_keyword_lists(self) -> Dict[str, List[str]]:
        """获取所有保存的关键词列表
        
        Returns:
            以列表名称为键的关键词列表字典
        """
        keyword_files = [f for f in os.listdir(self.keywords_dir) if f.endswith('.json')]
        keyword_lists = {}
        
        for filename in keyword_files:
            try:
                with open(os.path.join(self.keywords_dir, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    name = data.get("name", filename.replace('.json', '').replace('_', ' '))
                    keyword_lists[name] = data.get("keywords", [])
            except Exception as e:
                print(f"读取关键词列表文件 {filename} 时出错: {str(e)}")
        
        return keyword_lists
    
    def clear_all_results(self):
        """
        清空所有处理结果
        """
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
            os.makedirs(self.results_dir)
        return True
    
    def result_exists(self, cache_key):
        """
        检查指定的缓存键是否存在结果
        
        参数:
            cache_key: 缓存键
        
        返回:
            布尔值，表示是否存在
        """
        result_file = os.path.join(self.results_dir, f"{cache_key}.json")
        return os.path.exists(result_file)
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """将处理结果导出为DataFrame
        
        Returns:
            包含处理结果的DataFrame
        """
        index_data = self._get_index()
        
        if not index_data:
            return pd.DataFrame()
        
        # 将索引数据转换为DataFrame
        df = pd.DataFrame(index_data)
        
        # 如果relevant_keywords是列表，将其转换为字符串
        if 'relevant_keywords' in df.columns:
            df['relevant_keywords'] = df['relevant_keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # 如果application_domains是列表，将其转换为字符串
        if 'application_domains' in df.columns:
            df['application_domains'] = df['application_domains'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        return df
    
    def export_full_results(self, with_raw_response=False) -> pd.DataFrame:
        """导出所有结果的完整信息
        
        Args:
            with_raw_response: 是否包含原始响应内容
            
        Returns:
            包含完整结果信息的DataFrame
        """
        all_items = self.get_all_processed_items()
        if not all_items:
            return pd.DataFrame()
        
        rows = []
        for item in all_items:
            metadata = item.get('metadata', {})
            result = item.get('result', {})
            
            # 准备行数据
            row = {
                # 基本信息
                'cache_key': metadata.get('cache_key', ''),
                'title': metadata.get('title', ''),
                'year': metadata.get('year', ''),
                'source': metadata.get('source', ''),
                'area': metadata.get('area', ''),
                'method': metadata.get('method', ''),
                
                # 关键词匹配结果
                'relevant_keywords': ', '.join(result.get('relevant_keywords', [])),
                'has_matches': len(result.get('relevant_keywords', [])) > 0,
                
                # 领域分类结果
                'application_domains': ', '.join(result.get('application_domains', [])) if 'application_domains' in result else '',
                'justification': result.get('justification', ''),
                
                # 时间戳
                'timestamp': metadata.get('timestamp', '')
            }
            
            # 可选包含原始响应
            if with_raw_response:
                row['raw_response'] = result.get('raw_response', '')
            
            rows.append(row)
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # 测试代码
    cache_manager = CacheManager()
    
    # 测试生成缓存键
    test_key = cache_manager.generate_cache_key(
        "Test Title", 
        "Test Abstract", 
        ["keyword1", "keyword2"]
    )
    print(f"生成的缓存键: {test_key}")
    
    # 测试保存结果
    test_result = {
        "relevant_keywords": ["keyword1"],
        "explanations": {
            "keyword1": "This is relevant because..."
        }
    }
    
    test_metadata = {
        "id": 123,
        "title": "Test Title",
        "abstract": "Test Abstract",
        "year": 2023,
        "source": "CNKI",
        "area": "financial_risk",
        "method": "machine learning"
    }
    
    saved = cache_manager.save_result(test_metadata, test_result)
    print(f"保存结果: {'成功' if saved[1] else '失败'}")
    
    # 测试获取缓存结果
    cached_result = cache_manager.get_cached_result(test_key)
    print(f"获取缓存结果: {cached_result}")
    
    # 测试保存标注
    test_annotation = {
        "is_correct": True,
        "feedback": "Good analysis"
    }
    
    saved = cache_manager.save_annotation(test_key, test_annotation)
    print(f"保存标注: {'成功' if saved else '失败'}")
    
    # 测试获取标注
    annotation = cache_manager.get_annotation(test_key)
    print(f"获取标注: {annotation}")