import os
import pandas as pd
import glob
from typing import Dict, List, Optional, Tuple, Union

# 根目录路径
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'raw_data')

# 数据源目录
CNKI_DIR = os.path.join(BASE_DIR, "CNKI")
WOS_DIR = os.path.join(BASE_DIR, "WOS")

# 领域映射关系
AREA_MAPPING = {
    "pricing": "derivatives_pricing",  # 对应于pricing_ml.xls等文件
    "risk": "financial_risk",          # 对应于risk_ml.xls等文件
    "portfolio": "portfolio"           # 对应于portfolio_ml.xls等文件
}

# 方法映射关系
METHOD_MAPPING = {
    "ml": "machine learning",
    "dl": "deep learning",
    "llm": "LLMs"
}

class DataLoader:
    """数据加载器，用于加载和预处理CNKI和WOS的数据"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.cnki_files = self._get_data_files(CNKI_DIR)
        self.wos_files = self._get_data_files(WOS_DIR)
    
    def _get_data_files(self, directory: str) -> Dict[str, List[str]]:
        """获取目录下的所有xls和csv文件
        
        Args:
            directory: 目录路径
            
        Returns:
            包含xls和csv文件路径的字典
        """
        if not os.path.exists(directory):
            return {"xls": [], "csv": []}
        
        xls_files = glob.glob(os.path.join(directory, "*.xls"))
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        return {
            "xls": sorted(xls_files),
            "csv": sorted(csv_files)
        }
    
    def get_available_data_files(self) -> Dict[str, Dict[str, List[str]]]:
        """获取所有可用的数据文件
        
        Returns:
            包含CNKI和WOS数据文件的字典
        """
        return {
            "CNKI": self.cnki_files,
            "WOS": self.wos_files
        }
    
    def get_area_method_from_filename(self, filename: str) -> Tuple[str, str]:
        """从文件名中提取领域和方法信息
        
        Args:
            filename: 文件名
            
        Returns:
            (领域, 方法) 元组
        """
        basename = os.path.basename(filename)
        name_parts = os.path.splitext(basename)[0].split('_')
        
        # 文件名格式应该是 area_method.xls，例如 risk_ml.xls
        if len(name_parts) >= 2:
            area_key = name_parts[0].lower()
            method_key = name_parts[1].lower()
            
            area = AREA_MAPPING.get(area_key, area_key)
            method = METHOD_MAPPING.get(method_key, method_key)
            
            return area, method
        
        return None, None
    
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """加载单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            加载的DataFrame，加载失败返回None
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.xls' or file_ext == '.xlsx':
                df = pd.read_excel(file_path)
            elif file_ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                print(f"不支持的文件类型: {file_ext}")
                return None
            
            return df
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
            return None
    
    def preprocess_dataframe(self, df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
        """预处理DataFrame，统一不同数据源的列名
        
        Args:
            df: 原始DataFrame
            source: 数据源，'CNKI'或'WOS'
            
        Returns:
            预处理后的DataFrame，处理失败返回None
        """
        try:
            # 为不同来源设置不同的列名候选项
            if source == "CNKI":
                # CNKI的列名格式
                title_candidates = ["Title-题名", "题名", "Title", "论文题目"]
                abstract_candidates = ["Summary-摘要", "摘要", "Abstract", "Summary"]
                year_candidates = ["Year-年", "年", "Year", "出版年", "发表年份", "Publication Year"]
            else:
                # WOS的列名格式
                title_candidates = ["Article Title", "Title", "ArticleTitle"]
                abstract_candidates = ["Abstract", "Summary", "摘要"]
                year_candidates = ["Publication Year", "Year", "PublicationYear", "出版年"]
            
            # 初始化要查找的列
            title_col = None
            abstract_col = None
            year_col = None
            
            # 查找标题列
            for col in df.columns:
                # 检查列名是否在候选列表中（大小写不敏感，忽略空格）
                col_lower = col.lower().replace(" ", "")
                if any(cand.lower().replace(" ", "") in col_lower or 
                        col_lower in cand.lower().replace(" ", "") for cand in title_candidates):
                    title_col = col
                    break
            
            # 查找摘要列
            for col in df.columns:
                col_lower = col.lower().replace(" ", "")
                if any(cand.lower().replace(" ", "") in col_lower or 
                        col_lower in cand.lower().replace(" ", "") for cand in abstract_candidates):
                    abstract_col = col
                    break
            
            # 查找年份列
            for col in df.columns:
                col_lower = col.lower().replace(" ", "")
                if any(cand.lower().replace(" ", "") in col_lower or 
                        col_lower in cand.lower().replace(" ", "") for cand in year_candidates):
                    year_col = col
                    break
            
            # 检查是否找到所有需要的列
            if not title_col:
                print(f"警告: 无法找到标题列")
                print(f"可用的列: {list(df.columns)}")
                return None
            
            if not abstract_col:
                print(f"警告: 无法找到摘要列")
                print(f"可用的列: {list(df.columns)}")
                return None
            
            if not year_col:
                print(f"警告: 无法找到年份列")
                print(f"可用的列: {list(df.columns)}")
                return None
            
            # 创建标准化的DataFrame
            df_renamed = pd.DataFrame({
                "title": df[title_col],
                "abstract": df[abstract_col],
                "year": df[year_col],
                "source": source
            })
            
            # 预处理：填补空值
            df_renamed["title"] = df_renamed["title"].fillna("")
            df_renamed["abstract"] = df_renamed["abstract"].fillna("")
            
            # 将year转换为整数类型
            df_renamed["year"] = pd.to_numeric(df_renamed["year"], errors='coerce')
            df_renamed = df_renamed.dropna(subset=["year"])
            df_renamed["year"] = df_renamed["year"].astype(int)
            
            # 添加ID列用于追踪
            df_renamed["id"] = range(len(df_renamed))
            
            return df_renamed
        
        except Exception as e:
            print(f"预处理数据时出错: {str(e)}")
            return None
    
    def load_and_preprocess_file(self, file_path: str, source: str) -> Tuple[Optional[pd.DataFrame], str, str]:
        """加载并预处理单个文件
        
        Args:
            file_path: 文件路径
            source: 数据源，'CNKI'或'WOS'
            
        Returns:
            (处理后的DataFrame, 领域, 方法) 元组，处理失败返回(None, None, None)
        """
        # 从文件名获取领域和方法
        area, method = self.get_area_method_from_filename(file_path)
        if not area or not method:
            print(f"无法从文件名 {os.path.basename(file_path)} 中提取领域和方法信息，跳过此文件")
            return None, None, None
        
        # 加载文件
        df = self.load_file(file_path)
        if df is None:
            return None, None, None
        
        # 预处理数据
        df_processed = self.preprocess_dataframe(df, source)
        if df_processed is None:
            return None, None, None
        
        # 添加领域和方法信息
        df_processed["area"] = area
        df_processed["method"] = method
        
        return df_processed, area, method
    
    def load_multiple_files(self, file_paths: List[str], source: str) -> pd.DataFrame:
        """加载并预处理多个文件，合并结果
        
        Args:
            file_paths: 文件路径列表
            source: 数据源，'CNKI'或'WOS'
            
        Returns:
            合并后的DataFrame
        """
        all_dfs = []
        
        for file_path in file_paths:
            df, area, method = self.load_and_preprocess_file(file_path, source)
            if df is not None:
                all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    # 测试代码
    loader = DataLoader()
    
    available_files = loader.get_available_data_files()
    print("可用的CNKI数据文件:")
    for file_type, files in available_files["CNKI"].items():
        print(f"  {file_type}: {len(files)} 个文件")
        for file in files[:3]:  # 只打印前3个文件
            print(f"    - {os.path.basename(file)}")
    
    print("\n可用的WOS数据文件:")
    for file_type, files in available_files["WOS"].items():
        print(f"  {file_type}: {len(files)} 个文件")
        for file in files[:3]:  # 只打印前3个文件
            print(f"    - {os.path.basename(file)}")
    
    # 测试加载单个文件
    if available_files["CNKI"]["xls"]:
        test_file = available_files["CNKI"]["xls"][0]
        print(f"\n测试加载CNKI文件: {os.path.basename(test_file)}")
        df, area, method = loader.load_and_preprocess_file(test_file, "CNKI")
        if df is not None:
            print(f"成功加载! 领域: {area}, 方法: {method}")
            print(f"数据形状: {df.shape}")
            print("数据预览:")
            print(df.head(3)) 