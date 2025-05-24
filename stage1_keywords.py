import os
import sys
import json
import asyncio
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
import random

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from utils.data_loader import DataLoader
from utils.cache_manager import CacheManager
from utils.llm_processor import LLMProcessor, get_keywords

# 设置页面标题和配置
st.set_page_config(
    page_title="论文关键词匹配",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化数据加载器、缓存管理器和LLM处理器
@st.cache_resource
def get_data_loader():
    return DataLoader()

@st.cache_resource
def get_cache_manager():
    return CacheManager()

def get_llm_processor():
    api_key = st.session_state.get("api_key", "")
    processor = LLMProcessor(api_key=api_key)
    return processor

# 初始化Session状态
def init_session_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = None
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "user_prompt_template" not in st.session_state:
        st.session_state.user_prompt_template = ""
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = []
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "数据加载"
    if "annotation_results" not in st.session_state:
        st.session_state.annotation_results = {}
    # 添加视图控制状态
    if "show_detail_view" not in st.session_state:
        st.session_state.show_detail_view = False
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None
    # 添加关键词管理状态
    if "keyword_lists" not in st.session_state:
        st.session_state.keyword_lists = {}  # 保存的关键词列表集合
    if "to_select_keywords" not in st.session_state:
        st.session_state.to_select_keywords = []  # 批量选择临时存储
    if "to_delete_keywords" not in st.session_state:
        st.session_state.to_delete_keywords = []  # 批量删除临时存储
    # 添加处理状态控制
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False  # 是否正在处理数据
    if "current_processing" not in st.session_state:
        st.session_state.current_processing = None  # 当前正在处理的数据
    if "processed_items" not in st.session_state:
        st.session_state.processed_items = []  # 本次会话已处理的数据列表
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []  # 待处理队列
    if "display_page" not in st.session_state:
        st.session_state.display_page = {
            "unprocessed": 0,  # 未处理数据当前页码
            "processed": 0,    # 已处理数据当前页码
            "processing": 0,   # 正在处理数据当前页码
            "results_list": 0, # 结果列表当前页码
            "cached": 0,       # 缓存数据当前页码
            "page_size": 10    # 每页显示数量
        }
    # 添加数据加载缓存状态
    if "last_loaded_files" not in st.session_state:
        st.session_state.last_loaded_files = []  # 上次加载的文件列表
    if "last_loaded_source" not in st.session_state:
        st.session_state.last_loaded_source = ""  # 上次加载的数据源
    if "last_session_time" not in st.session_state:
        st.session_state.last_session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 上次会话时间
    if "to_delete_results" not in st.session_state:
        st.session_state.to_delete_results = []  # 待删除结果列表
    if "prompt_examples" not in st.session_state:
        st.session_state.prompt_examples = []  # 提示词示例
    if "confirm_clear_cache" not in st.session_state:
        st.session_state.confirm_clear_cache = False  # 确认清空缓存的状态

# 加载默认提示词
def load_default_prompts():
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    default_prompts = os.path.join(prompts_dir, "default.json")
    if os.path.exists(default_prompts):
        try:
            with open(default_prompts, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                st.session_state.system_prompt = prompts.get('system_prompt', '')
                st.session_state.user_prompt_template = prompts.get('user_prompt_template', '')
                st.session_state.prompt_examples = prompts.get('examples', [])
        except Exception as e:
            st.error(f"加载默认提示词时出错: {str(e)}")

# 增加缓存管理器中的数据加载缓存方法
def add_data_cache_methods():
    cache_manager = get_cache_manager()
    
    # 保存加载的数据
    def save_loaded_data(df, source, file_paths):
        """保存加载的数据到缓存"""
        if df is None or df.empty:
            return False
        
        try:
            # 创建缓存目录
            data_cache_dir = os.path.join(cache_manager.cache_dir, "data_cache")
            os.makedirs(data_cache_dir, exist_ok=True)
            
            # 保存数据
            cache_path = os.path.join(data_cache_dir, "last_loaded_data.pkl")
            df.to_pickle(cache_path)
            
            # 保存元数据
            metadata = {
                "source": source,
                "file_paths": file_paths,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rows": len(df)
            }
            with open(os.path.join(data_cache_dir, "last_loaded_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存加载数据到缓存时出错: {str(e)}")
            return False
    
    # 加载上次加载的数据
    def load_last_data():
        """从缓存加载上次加载的数据"""
        try:
            # 检查缓存文件
            data_cache_dir = os.path.join(cache_manager.cache_dir, "data_cache")
            cache_path = os.path.join(data_cache_dir, "last_loaded_data.pkl")
            metadata_path = os.path.join(data_cache_dir, "last_loaded_metadata.json")
            
            if not os.path.exists(cache_path) or not os.path.exists(metadata_path):
                return None, None
            
            # 加载元数据
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # 加载数据
            df = pd.read_pickle(cache_path)
            
            return df, metadata
        except Exception as e:
            print(f"从缓存加载数据时出错: {str(e)}")
            return None, None
    
    # API密钥缓存功能
    def save_api_key(api_key):
        """保存API密钥到缓存"""
        if not api_key:
            return False
            
        try:
            # 创建缓存目录
            api_cache_dir = os.path.join(cache_manager.cache_dir, "api_cache")
            os.makedirs(api_cache_dir, exist_ok=True)
            
            # 保存API密钥
            with open(os.path.join(api_cache_dir, "api_key.txt"), "w", encoding="utf-8") as f:
                f.write(api_key)
                
            return True
        except Exception as e:
            print(f"保存API密钥到缓存时出错: {str(e)}")
            return False
    
    def load_api_key():
        """从缓存加载API密钥"""
        try:
            # 检查缓存文件
            api_cache_dir = os.path.join(cache_manager.cache_dir, "api_cache")
            cache_path = os.path.join(api_cache_dir, "api_key.txt")
            
            if not os.path.exists(cache_path):
                return ""
                
            # 加载API密钥
            with open(cache_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
                
            return api_key
        except Exception as e:
            print(f"从缓存加载API密钥时出错: {str(e)}")
            return ""
    
    # 保存当前选择的关键词
    def save_current_keywords(keywords):
        """保存当前选择的关键词到缓存"""
        if not keywords:
            return False
        
        try:
            # 创建缓存目录
            keywords_cache_dir = os.path.join(cache_manager.cache_dir, "keywords_cache")
            os.makedirs(keywords_cache_dir, exist_ok=True)
            
            # 保存当前关键词
            with open(os.path.join(keywords_cache_dir, "current_keywords.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "keywords": keywords,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存当前关键词到缓存时出错: {str(e)}")
            return False
    
    # 加载上次选择的关键词
    def load_last_keywords():
        """从缓存加载上次选择的关键词"""
        try:
            # 检查缓存文件
            keywords_cache_dir = os.path.join(cache_manager.cache_dir, "keywords_cache")
            cache_path = os.path.join(keywords_cache_dir, "current_keywords.json")
            
            if not os.path.exists(cache_path):
                return []
            
            # 加载关键词
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            return data.get("keywords", [])
        except Exception as e:
            print(f"从缓存加载关键词时出错: {str(e)}")
            return []
    
    # 添加方法到cache_manager
    cache_manager.save_loaded_data = save_loaded_data
    cache_manager.load_last_data = load_last_data
    cache_manager.save_api_key = save_api_key
    cache_manager.load_api_key = load_api_key
    cache_manager.save_current_keywords = save_current_keywords
    cache_manager.load_last_keywords = load_last_keywords
    
    return cache_manager

# 修改数据加载页面
def render_data_loading_page():
    st.header("📊 数据加载")
    
    # 初始化增强版缓存管理器
    cache_manager = add_data_cache_methods()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("选择数据来源和文件")
        data_loader = get_data_loader()
        available_files = data_loader.get_available_data_files()
        
        # 检查缓存中的上一次加载的数据
        cached_data, metadata = cache_manager.load_last_data()
        if cached_data is not None and metadata is not None:
            st.info(f"发现上次加载的数据: {metadata['rows']}条记录，来源: {metadata['source']}，加载时间: {metadata['timestamp']}")
            
            if st.button("恢复上次加载的数据"):
                st.session_state.loaded_data = cached_data
                st.session_state.last_loaded_source = metadata['source']
                st.session_state.last_loaded_files = metadata['file_paths']
                st.success(f"已恢复上次加载的数据，共{len(cached_data)}条记录")
                # 重新加载页面以更新UI
                st.rerun()
        
        # 选择数据源
        data_source = st.radio("选择数据来源:", ["CNKI", "WOS"])
        
        # 显示可用文件
        available_file_types = available_files[data_source]
        file_type = st.radio("选择文件类型:", ["xls", "csv"])
        
        if available_file_types[file_type]:
            selected_files = st.multiselect(
                "选择要加载的文件:",
                [os.path.basename(f) for f in available_file_types[file_type]],
                key=f"{data_source}_{file_type}_selection"
            )
            
            selected_file_paths = [
                os.path.join(os.path.dirname(f), s) 
                for f in available_file_types[file_type] 
                for s in selected_files 
                if os.path.basename(f) == s
            ]
            
            if selected_file_paths and st.button("加载选中的文件"):
                try:
                    with st.spinner("正在加载数据..."):
                        df = data_loader.load_multiple_files(selected_file_paths, data_source)
                        if not df.empty:
                            st.session_state.loaded_data = df
                            st.session_state.last_loaded_source = data_source
                            st.session_state.last_loaded_files = selected_file_paths
                            
                            # 保存到缓存
                            cache_manager.save_loaded_data(df, data_source, selected_file_paths)
                            
                            st.success(f"成功加载 {len(df)} 条数据！")
                            time.sleep(1)  # 给用户时间看到成功消息
                            st.rerun()  # 重新加载页面以更新其他组件
                        else:
                            st.error("没有成功加载任何数据。")
                except Exception as e:
                    st.error(f"加载数据时出错: {str(e)}")
        else:
            st.info(f"没有找到 {data_source} 的 {file_type} 文件。")

    with col2:
        st.subheader("已加载的数据预览")
        if st.session_state.loaded_data is not None:
            df = st.session_state.loaded_data
            st.write(f"数据形状: {df.shape}")
            
            # 显示数据统计信息
            stats_tab1, stats_tab2 = st.tabs(["数据分布", "数据预览"])
            
            with stats_tab1:
                # 按领域和方法显示数据分布
                if 'area' in df.columns and 'method' in df.columns:
                    area_counts = df['area'].value_counts().reset_index()
                    area_counts.columns = ['area', 'count']
                    
                    method_counts = df['method'].value_counts().reset_index()
                    method_counts.columns = ['method', 'count']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("按领域分布")
                        fig = px.pie(area_counts, values='count', names='area', title='按领域分布')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("按方法分布")
                        fig = px.pie(method_counts, values='count', names='method', title='按方法分布')
                        st.plotly_chart(fig, use_container_width=True)
                    
                # 按年份显示数据分布
                if 'year' in df.columns:
                    year_counts = df['year'].value_counts().reset_index()
                    year_counts.columns = ['year', 'count']
                    year_counts = year_counts.sort_values('year')
                    
                    st.subheader("按年份分布")
                    fig = px.bar(year_counts, x='year', y='count', title='按年份分布')
                    st.plotly_chart(fig, use_container_width=True)
            
            with stats_tab2:
                st.dataframe(df.head(10))
                
                # 随机展示一条数据
                if st.button("随机展示一条数据"):
                    if len(df) > 0:  # 确保有数据可以展示
                        random_idx = random.randint(0, len(df) - 1)
                        random_row = df.iloc[random_idx]
                        
                        st.subheader("随机数据样例")
                        st.markdown(f"**标题**: {random_row['title']}")
                        st.markdown(f"**摘要**: {random_row['abstract']}")
                        st.markdown(f"**年份**: {random_row['year']}")
                        st.markdown(f"**领域**: {random_row['area']}")
                        st.markdown(f"**方法**: {random_row['method']}")
                    else:
                        st.error("数据集为空，无法展示随机数据。")
        else:
            st.info("请先加载数据。")

# 修改关键词管理页面
def render_keywords_management_page():
    st.header("🔑 关键词管理")
    
    # 初始化增强版缓存管理器
    cache_manager = add_data_cache_methods()
    
    # 如果还没有选择关键词，尝试从缓存恢复上次选择的关键词
    if not st.session_state.selected_keywords:
        last_keywords = cache_manager.load_last_keywords()
        if last_keywords:
            st.info(f"发现上次选择的{len(last_keywords)}个关键词")
            if st.button("恢复上次选择的关键词"):
                st.session_state.selected_keywords = last_keywords
                st.success(f"已恢复上次选择的{len(last_keywords)}个关键词")
                st.rerun()  # 立即刷新页面显示恢复的关键词
    
    # 获取关键词
    keywords_dict = get_keywords()
    
    if not keywords_dict:
        st.error("无法加载关键词，请检查keywords.py文件是否存在。")
        return
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("关键词选择")
        
        # 选择关键词类别的选项卡
        categories = list(keywords_dict.keys())
        category = st.selectbox("选择关键词类别:", categories)
        
        # 确保选择的类别存在于字典中
        if category in keywords_dict:
            keywords = keywords_dict[category]
            st.write(f"共 {len(keywords)} 个关键词")
            
            # 批量选择控制
            st.subheader("批量选择")
            batch_col1, batch_col2 = st.columns(2)
            
            with batch_col1:
                select_all = st.checkbox("全选", key=f"select_all_{category}")
            
            with batch_col2:
                if select_all:
                    st.session_state.to_select_keywords = keywords.copy()
                
                if st.button("批量添加", key=f"batch_add_{category}"):
                    # 将批量选择的关键词添加到已选列表
                    added_count = 0
                    for kw in st.session_state.to_select_keywords:
                        if kw not in st.session_state.selected_keywords:
                            st.session_state.selected_keywords.append(kw)
                            added_count += 1
                    
                    st.session_state.to_select_keywords = []  # 清空临时列表
                    
                    # 保存当前选择的关键词到缓存
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                    
                    if added_count > 0:
                        st.success(f"已添加{added_count}个关键词")
                        st.rerun()  # 立即刷新以更新界面
            
            # 显示关键词列表并提供单独选择功能
            st.subheader("关键词列表")
            
            # 使用容器显示关键词列表
            keyword_container = st.container()
            
            # 计算行数和列数
            keywords_count = len(keywords)
            num_columns = 4
            rows_per_column = (keywords_count + num_columns - 1) // num_columns  # 向上取整
            
            # 创建关键词控件列表
            keyword_controls = []
            
            # 为每个关键词创建控件集合
            for keyword_idx, keyword in enumerate(keywords):
                # 计算列索引 - 修改为列优先顺序（先填满第一列再填第二列）
                col_idx = keyword_idx // rows_per_column
                
                # 计算行索引
                row_idx = keyword_idx % rows_per_column
                
                # 检查是否已在已选列表中
                is_selected = keyword in st.session_state.selected_keywords
                is_in_batch = keyword in st.session_state.to_select_keywords
                
                # 添加到控件列表
                keyword_controls.append((row_idx, col_idx, keyword, is_selected, is_in_batch))
            
            # 按行排序控件
            keyword_controls.sort()
            
            # 创建4个列来显示关键词
            cols = st.columns(4)
            
            # 分配关键词到列
            for row_idx in range(rows_per_column):
                for col_idx in range(num_columns):
                    # 查找当前行列位置的关键词
                    current_controls = [
                        control for control in keyword_controls 
                        if control[0] == row_idx and control[1] == col_idx
                    ]
                    
                    if current_controls:
                        _, _, keyword, is_selected, is_in_batch = current_controls[0]
                        
                        # 在对应列中显示关键词
                        with cols[col_idx]:
                            # 显示关键词和添加按钮
                            batch_select = st.checkbox(
                                keyword, 
                                value=select_all or is_in_batch,
                                key=f"kw_{category}_{keyword}"
                            )
                            
                            add_disabled = is_selected
                            if st.button(
                                "➕", 
                                key=f"add_{category}_{keyword}", 
                                disabled=add_disabled
                            ):
                                st.session_state.selected_keywords.append(keyword)
                                # 保存当前选择的关键词到缓存
                                cache_manager.save_current_keywords(st.session_state.selected_keywords)
                                st.rerun()  # 立即刷新界面
                            
                            # 更新批量选择列表
                            if batch_select and keyword not in st.session_state.to_select_keywords:
                                st.session_state.to_select_keywords.append(keyword)
                            elif not batch_select and keyword in st.session_state.to_select_keywords:
                                st.session_state.to_select_keywords.remove(keyword)
        else:
            st.error(f"找不到类别'{category}'的关键词。")
    
    with col2:
        st.subheader("已选关键词")
        
        # 显示已选关键词数量
        num_selected = len(st.session_state.selected_keywords)
        st.write(f"已选择 {num_selected} 个关键词")
        
        # 保存关键词列表功能
        st.subheader("保存关键词列表")
        
        # 使用container代替嵌套列
        save_container = st.container()
        list_name = save_container.text_input("关键词列表名称:", placeholder="输入名称...")
        
        save_disabled = not list_name or num_selected == 0
        if save_container.button("保存关键词列表", disabled=save_disabled):
            try:
                # 保存到session_state
                st.session_state.keyword_lists[list_name] = st.session_state.selected_keywords.copy()
                # 持久化保存
                if cache_manager.save_keyword_list(list_name, st.session_state.selected_keywords):
                    st.success(f"已保存关键词列表：{list_name}")
                    # 同时保存当前选择的关键词
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                else:
                    st.error("保存关键词列表失败")
            except Exception as e:
                st.error(f"保存关键词列表时出错: {str(e)}")
        
        # 加载已保存的关键词列表
        saved_lists_from_cache = cache_manager.get_all_keyword_lists()
        
        # 合并内存中和缓存中的关键词列表
        for name, keywords in saved_lists_from_cache.items():
            if name not in st.session_state.keyword_lists:
                st.session_state.keyword_lists[name] = keywords
        
        if st.session_state.keyword_lists:
            saved_lists = list(st.session_state.keyword_lists.keys())
            selected_list = st.selectbox("加载已保存的列表:", [""] + saved_lists)
            
            if selected_list and st.button("加载列表"):
                try:
                    st.session_state.selected_keywords = st.session_state.keyword_lists[selected_list].copy()
                    # 保存当前选择的关键词到缓存
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                    st.success(f"已加载关键词列表：{selected_list}")
                    st.rerun()  # 立即刷新界面
                except Exception as e:
                    st.error(f"加载关键词列表时出错: {str(e)}")
        
        # 批量删除功能
        st.subheader("关键词管理")
        
        # 批量操作功能 - 使用container代替嵌套列
        delete_container = st.container()
        delete_all = delete_container.checkbox("全选删除")
        
        if delete_all:
            st.session_state.to_delete_keywords = st.session_state.selected_keywords.copy()
        
        delete_disabled = len(st.session_state.to_delete_keywords) == 0
        if delete_container.button("批量删除", disabled=delete_disabled):
            try:
                removed_count = 0
                for kw in st.session_state.to_delete_keywords:
                    if kw in st.session_state.selected_keywords:
                        st.session_state.selected_keywords.remove(kw)
                        removed_count += 1
                
                st.session_state.to_delete_keywords = []  # 清空临时列表
                # 保存当前选择的关键词到缓存
                cache_manager.save_current_keywords(st.session_state.selected_keywords)
                
                if removed_count > 0:
                    st.success(f"已删除{removed_count}个关键词")
                    st.rerun()  # 立即刷新界面
            except Exception as e:
                st.error(f"删除关键词时出错: {str(e)}")
        
        # 显示已选关键词
        st.subheader("已选关键词列表")
        
        if st.session_state.selected_keywords:
            # 使用容器而不是嵌套列
            selected_keywords_container = st.container()
            
            # 计算每组应显示的关键词数量
            num_selected = len(st.session_state.selected_keywords)
            num_columns = 4
            rows_per_column = (num_selected + num_columns - 1) // num_columns  # 向上取整
            
            # 创建关键词控件列表 - 修改为列优先顺序
            keyword_controls = []
            
            # 为每个关键词创建一个控件集合
            for keyword_idx, keyword in enumerate(st.session_state.selected_keywords):
                # 计算列索引 - 列优先顺序
                col_idx = keyword_idx // rows_per_column
                
                # 计算行索引
                row_idx = keyword_idx % rows_per_column
                
                # 添加到控件列表
                keyword_controls.append((row_idx, col_idx, keyword))
            
            # 按行列顺序排序控件
            keyword_controls.sort()
            
            # 创建4个列来显示关键词
            selected_cols = st.columns(4)
            
            # 显示控件
            for row_idx, col_idx, keyword in keyword_controls:
                with selected_cols[col_idx]:
                    # 显示复选框和删除按钮
                    is_in_delete_batch = keyword in st.session_state.to_delete_keywords
                    delete_select = st.checkbox(
                        keyword, 
                        value=delete_all or is_in_delete_batch,
                        key=f"del_{keyword}"
                    )
                    
                    if st.button("删除", key=f"remove_{keyword}"):
                        st.session_state.selected_keywords.remove(keyword)
                        # 保存当前选择的关键词到缓存
                        cache_manager.save_current_keywords(st.session_state.selected_keywords)
                        st.rerun()  # 立即刷新界面
                    
                    # 更新批量删除列表
                    if delete_select and keyword not in st.session_state.to_delete_keywords:
                        st.session_state.to_delete_keywords.append(keyword)
                    elif not delete_select and keyword in st.session_state.to_delete_keywords:
                        st.session_state.to_delete_keywords.remove(keyword)
        else:
            st.info("请先从左侧选择关键词。")
        
        # 添加自定义关键词
        st.subheader("添加自定义关键词")
        
        # 使用容器而非嵌套列
        custom_container = st.container()
        new_keyword = custom_container.text_input("输入关键词:")
        
        add_disabled = not new_keyword or new_keyword in st.session_state.selected_keywords
        if custom_container.button("添加自定义关键词", disabled=add_disabled, key="add_custom"):
            if new_keyword and new_keyword not in st.session_state.selected_keywords:
                st.session_state.selected_keywords.append(new_keyword)
                # 保存当前选择的关键词到缓存
                cache_manager.save_current_keywords(st.session_state.selected_keywords)
                st.success(f"已添加关键词：{new_keyword}")
                st.rerun()  # 立即刷新界面

# 提示词管理页面
def render_prompts_management_page():
    st.header("💬 提示词管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("编辑系统提示词")
        system_prompt = st.text_area("系统提示词:", st.session_state.system_prompt, height=300)
        
        st.subheader("编辑用户提示词模板")
        user_prompt_template = st.text_area("用户提示词模板:", st.session_state.user_prompt_template, height=200)
        
        if st.button("保存提示词"):
            st.session_state.system_prompt = system_prompt
            st.session_state.user_prompt_template = user_prompt_template
            st.success("提示词已更新。")
    
    with col2:
        st.subheader("提示词模板示例")
        if hasattr(st.session_state, 'prompt_examples'):
            examples = st.session_state.prompt_examples
            selected_example = st.selectbox("选择示例:", [""] + [ex['name'] for ex in examples])
            
            if selected_example:
                example = next((ex for ex in examples if ex['name'] == selected_example), None)
                if example:
                    st.text_area("示例系统提示词:", example['system_prompt'], height=200, disabled=True)
                    st.text_area("示例用户提示词模板:", example['user_prompt_template'], height=100, disabled=True)
                    
                    if st.button("使用此示例"):
                        st.session_state.system_prompt = example['system_prompt']
                        st.session_state.user_prompt_template = example['user_prompt_template']
        
        st.subheader("预览格式化后的用户提示词")
        if st.session_state.loaded_data is not None and not st.session_state.loaded_data.empty and st.session_state.selected_keywords:
            df = st.session_state.loaded_data
            random_idx = random.randint(0, len(df) - 1)
            random_row = df.iloc[random_idx]
            
            processor = get_llm_processor()
            processor.set_prompts(st.session_state.system_prompt, st.session_state.user_prompt_template)
            
            formatted_prompt = processor.format_user_prompt(
                random_row['title'],
                random_row['abstract'],
                st.session_state.selected_keywords
            )
            
            st.text_area("预览:", formatted_prompt, height=300, disabled=True)
        else:
            st.info("请先加载数据并选择关键词。")

# 数据分页函数
def paginate_dataframe(df, page_key, page_size_key=None):
    """
    将数据框分页显示
    
    参数:
        df: 要分页的数据框
        page_key: 页码在session_state中的键名
        page_size_key: 每页大小在session_state中的键名，如果为None则使用默认值
    
    返回:
        当前页的数据框
    """
    if df.empty:
        return df
    
    page = st.session_state.display_page.get(page_key, 0)
    page_size = st.session_state.display_page.get("page_size", 10)
    
    # 计算总页数
    total_pages = (len(df) + page_size - 1) // page_size
    
    # 确保页码在有效范围内
    page = max(0, min(page, total_pages - 1))
    
    # 计算当前页的起始和结束索引
    start_idx = page * page_size
    end_idx = min(start_idx + page_size, len(df))
    
    # 返回当前页的数据
    return df.iloc[start_idx:end_idx], page, total_pages, start_idx, end_idx

# 页面导航控件
def render_pagination_controls(page_key, total_pages, current_page):
    """
    渲染分页控件
    
    参数:
        page_key: 页码在session_state中的键名
        total_pages: 总页数
        current_page: 当前页码
    """
    if total_pages <= 1:
        return
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])
    
    with col1:
        if st.button("首页", key=f"first_{page_key}", disabled=current_page == 0):
            st.session_state.display_page[page_key] = 0
            st.rerun()
    
    with col2:
        if st.button("上一页", key=f"prev_{page_key}", disabled=current_page == 0):
            st.session_state.display_page[page_key] = max(0, current_page - 1)
            st.rerun()
    
    with col3:
        st.write(f"第 {current_page + 1} 页，共 {total_pages} 页")
    
    with col4:
        if st.button("下一页", key=f"next_{page_key}", disabled=current_page >= total_pages - 1):
            st.session_state.display_page[page_key] = min(total_pages - 1, current_page + 1)
            st.rerun()
    
    with col5:
        if st.button("末页", key=f"last_{page_key}", disabled=current_page >= total_pages - 1):
            st.session_state.display_page[page_key] = total_pages - 1
            st.rerun()
    
    # 允许直接跳转到指定页面
    jump_col1, jump_col2 = st.columns([3, 1])
    with jump_col1:
        jump_page = st.number_input(
            "跳转到页码:", 
            min_value=1, 
            max_value=total_pages,
            value=current_page + 1,
            key=f"jump_input_{page_key}"
        )
    
    with jump_col2:
        if st.button("跳转", key=f"jump_{page_key}"):
            st.session_state.display_page[page_key] = jump_page - 1
            st.rerun()

# 显示数据表格
def render_data_table(df, show_columns=None, title="数据列表", page_key=None):
    """
    渲染数据表格并提供分页功能
    
    参数:
        df: 要显示的数据框
        show_columns: 要显示的列
        title: 表格标题
        page_key: 页码在session_state中的键名
    """
    if df.empty:
        st.info(f"没有{title}数据")
        return
    
    # 设置要显示的列
    if show_columns is None:
        show_columns = ['title', 'year', 'area', 'method']
    
    # 确保所有指定的列都存在
    valid_columns = [col for col in show_columns if col in df.columns]
    
    if not valid_columns:
        st.error(f"数据中没有可显示的列")
        return
    
    # 如果有页码键，则使用分页显示
    if page_key is not None:
        current_df, current_page, total_pages, start_idx, end_idx = paginate_dataframe(df, page_key)
        st.write(f"{title} ({len(df)}条，显示第 {start_idx+1}-{end_idx} 条):")
        st.dataframe(current_df[valid_columns], use_container_width=True)
        
        # 渲染分页控件
        render_pagination_controls(page_key, total_pages, current_page)
    else:
        # 否则直接显示所有数据
        st.write(f"{title} ({len(df)}条):")
        st.dataframe(df[valid_columns], use_container_width=True)

# 检查文献是否已被处理
def is_paper_processed(title, abstract, cache_manager, selected_keywords):
    """
    检查文献是否已经被处理过
    
    参数:
        title: 文献标题
        abstract: 文献摘要
        cache_manager: 缓存管理器
        selected_keywords: 选中的关键词列表
    
    返回:
        布尔值，表示是否已处理
    """
    try:
        # 生成缓存键
        cache_key = cache_manager.generate_cache_key(title, abstract, selected_keywords)
        
        # 直接检查结果文件是否存在
        result_file = os.path.join(cache_manager.results_dir, f"{cache_key}.json")
        return os.path.exists(result_file)
    except Exception as e:
        print(f"检查文献是否已处理时出错: {str(e)}")
        return False

# 改进缓存数据获取函数
def get_processed_papers(cache_manager, selected_keywords=None, filter_criteria=None):
    """
    获取已处理的文献列表，支持关键词和其他条件筛选
    
    参数:
        cache_manager: 缓存管理器
        selected_keywords: 选中的关键词列表，如果为None则不按关键词筛选
        filter_criteria: 其他筛选条件，例如领域、方法等
    
    返回:
        已处理文献的DataFrame
    """
    # 获取所有已处理项目
    all_processed_items = cache_manager.get_all_processed_items()
    
    # 准备数据
    processed_data = []
    for item in all_processed_items:
        metadata = item
        
        # 应用筛选条件
        skip_item = False
        if filter_criteria:
            for key, value in filter_criteria.items():
                if metadata.get(key) != value:
                    skip_item = True
                    break
        
        if skip_item:
            continue
        
        # 如果指定了关键词，则只返回使用这些关键词处理的文献
        if selected_keywords is not None:
            # 获取处理该文献时使用的关键词
            used_keywords = metadata.get('keywords', [])
            # 如果没有关键词信息，则跳过
            if not used_keywords:
                continue
            # 检查是否是相同的关键词集合 (宽松匹配，只要有一个关键词相同就返回)
            if not set(used_keywords).intersection(set(selected_keywords)):
                continue
        
        # 获取关键结果信息
        cache_key = metadata.get('cache_key', '')
        result = cache_manager.get_cached_result(cache_key) if cache_key else {}
        
        # 提取关键信息
        processed_item = {
            'title': metadata.get('title', ''),
            'abstract': metadata.get('abstract', ''),
            'year': metadata.get('year', ''),
            'area': metadata.get('area', ''),
            'method': metadata.get('method', ''),
            'source': metadata.get('source', ''),
            'success': result.get('success', False) if result else False,
            'relevant_keywords': result.get('relevant_keywords', []) if result else [],
            'cache_key': cache_key
        }
        
        processed_data.append(processed_item)
    
    # 转换为DataFrame
    if processed_data:
        return pd.DataFrame(processed_data)
    else:
        return pd.DataFrame()

# LLM处理页面
def render_llm_processing_page():
    st.header("🤖 LLM处理")
    
    # 创建主要的列布局
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        st.subheader("API设置")
        
        # 获取缓存管理器
        cache_manager = get_cache_manager()
        cache_manager = add_data_cache_methods()
        
        # 尝试从缓存加载API密钥
        if not st.session_state.api_key:
            cached_api_key = cache_manager.load_api_key()
            if cached_api_key:
                st.session_state.api_key = cached_api_key
                
        api_key = st.text_input("DeepSeek API密钥:", st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            # 保存API密钥到缓存
            if api_key:
                cache_manager.save_api_key(api_key)
        
        if not st.session_state.api_key:
            st.warning("请输入DeepSeek API密钥。")
        
        # 检查是否处于处理状态
        is_processing = st.session_state.is_processing
        
        # 处理设置部分
        st.subheader("处理设置")
        if st.session_state.loaded_data is not None and not st.session_state.loaded_data.empty:
            # 获取缓存管理器
            cache_manager = get_cache_manager()
            # 增强版缓存管理器
            cache_manager = add_data_cache_methods()
            # 添加删除结果方法
            cache_manager = add_delete_result_method()
            
            # 提取标题和摘要以便筛选
            df = st.session_state.loaded_data
            
            # 筛选要处理的数据
            filter_section = st.container()
            
            with filter_section:
                if not is_processing:  # 只有在非处理状态才允许修改筛选条件
                    filter_col, process_col = st.columns(2)
                    
                    with filter_col:
                        # 按领域筛选
                        if 'area' in df.columns:
                            areas = ['全部'] + sorted(df['area'].unique().tolist())
                            selected_area = st.selectbox("按领域筛选:", areas, disabled=is_processing)
                        
                        # 按方法筛选
                        if 'method' in df.columns:
                            methods = ['全部'] + sorted(df['method'].unique().tolist())
                            selected_method = st.selectbox("按方法筛选:", methods, disabled=is_processing)
                        
                        # 按年份筛选
                        if 'year' in df.columns:
                            years = sorted(df['year'].unique().tolist())
                            min_year, max_year = min(years), max(years)
                            # 检查最小值和最大值是否相同
                            if min_year == max_year:
                                st.write(f"年份: {min_year}（所有文档年份相同）")
                                year_range = (min_year, min_year)
                            else:
                                year_range = st.slider(
                                    "按年份筛选:", 
                                    min_value=min_year, 
                                    max_value=max_year, 
                                    value=(min_year, max_year), 
                                    disabled=is_processing
                                )
                        
                        # 添加是否排除已处理数据的选项
                        exclude_processed = st.checkbox("排除已处理数据", value=True, disabled=is_processing)
                    
                    # 应用筛选
                    filtered_df = df.copy()
                    
                    if 'area' in df.columns and selected_area != '全部':
                        filtered_df = filtered_df[filtered_df['area'] == selected_area]
                    
                    if 'method' in df.columns and selected_method != '全部':
                        filtered_df = filtered_df[filtered_df['method'] == selected_method]
                    
                    # 应用年份筛选
                    if 'year' in df.columns and min_year != max_year:
                        # 确保year列是数值类型
                        filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
                        # 使用有效的年份数据进行筛选
                        filtered_df = filtered_df.dropna(subset=['year'])
                        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
                    
                    # 排除已处理的数据 - 修复版本
                    if exclude_processed:
                        # 获取所有已处理结果
                        processed_results = cache_manager.get_results_by_filter({})
                        processed_titles_set = set()
                        
                        # 收集所有已处理的标题，用集合提高查找效率
                        for item in processed_results:
                            title = item['metadata'].get('title', '')
                            if title:
                                processed_titles_set.add(title)
                        
                        # 过滤掉已处理的数据
                        if processed_titles_set:
                            filtered_df = filtered_df[~filtered_df['title'].isin(processed_titles_set)]
                    
                    with process_col:
                        st.write(f"筛选后数据条数: {len(filtered_df)}")
                        
                        # 设置处理参数
                        batch_size = st.slider("批次大小:", min_value=1, max_value=50, value=10, disabled=is_processing)
                        max_concurrent = st.slider("最大并发请求数:", min_value=1, max_value=50, value=5, disabled=is_processing)
                        
                        # 保存处理参数到会话状态
                        st.session_state.batch_size = batch_size
                        st.session_state.max_concurrent = max_concurrent
                        
                        # 随机抽样
                        sample_size = st.number_input("随机抽样数量 (0表示使用全部数据):", min_value=0, max_value=len(filtered_df), value=min(20, len(filtered_df)), disabled=is_processing)
                        
                        if sample_size > 0 and sample_size < len(filtered_df) and not is_processing:
                            filtered_df = filtered_df.sample(sample_size, random_state=42)
                            st.write(f"已随机抽取 {len(filtered_df)} 条数据。")
                
                # 显示处理状态
                status_container = st.container()
                
                # 检查是否已选择关键词
                error_msgs = []
                if not st.session_state.selected_keywords:
                    error_msgs.append("请先在关键词管理页面选择关键词。")
                # 检查提示词是否已设置
                if not st.session_state.system_prompt or not st.session_state.user_prompt_template:
                    error_msgs.append("请先在提示词管理页面设置提示词。")
                # 检查API密钥是否已设置
                if not st.session_state.api_key:
                    error_msgs.append("请先设置DeepSeek API密钥。")
                
                if error_msgs:
                    for msg in error_msgs:
                        st.warning(msg)
                elif not is_processing:
                    # 创建处理队列并存储在会话状态中
                    if st.button("开始处理", disabled=is_processing or filtered_df.empty):
                        # 将筛选后的数据转换为记录列表
                        st.session_state.processing_queue = filtered_df.to_dict('records')
                        st.session_state.is_processing = True
                        st.session_state.processed_items = []  # 清空已处理列表
                        # 重新加载页面以开始处理
                        st.rerun()
                else:
                    # 显示正在处理的状态
                    with status_container:
                        st.info(f"正在处理数据，队列中还有 {len(st.session_state.processing_queue)} 条数据待处理")
                        
                        if st.button("停止处理", key="stop_processing_btn"):
                            st.session_state.is_processing = False
                            st.success("已停止处理")
                            st.rerun()
            
            # 显示数据列表
            data_tabs = st.tabs(["待处理数据", "正在处理", "本次已处理数据", "缓存中的已处理数据"])
            
            with data_tabs[0]:
                # 待处理数据列表
                if is_processing:
                    # 如果正在处理，显示处理队列中的数据
                    queue_df = pd.DataFrame(st.session_state.processing_queue)
                    if not queue_df.empty:
                        render_data_table(queue_df, title="待处理数据", page_key="unprocessed")
                    else:
                        st.info("队列中没有待处理数据")
                else:
                    # 否则显示筛选后的数据
                    if not filtered_df.empty:
                        render_data_table(filtered_df, title="待处理数据", page_key="unprocessed")
                    else:
                        st.info("没有待处理数据")
            
            with data_tabs[1]:
                # 正在处理的数据
                if is_processing and st.session_state.processing_queue:
                    # 显示当前正在处理的批次
                    current_batch_size = min(st.session_state.batch_size, len(st.session_state.processing_queue))
                    batch_items = st.session_state.processing_queue[:current_batch_size]
                    current_df = pd.DataFrame(batch_items)
                    st.write(f"当前批处理 ({current_batch_size}条，最大并发: {st.session_state.max_concurrent})")
                    render_data_table(current_df, title="当前批次数据", page_key="processing")
                else:
                    st.info("没有正在处理的数据")
            
            with data_tabs[2]:
                # 本次已处理数据列表
                processed_df = pd.DataFrame(st.session_state.processed_items)
                if not processed_df.empty:
                    render_data_table(processed_df, title="本次已处理数据", page_key="processed")
                else:
                    st.info("本次会话尚未处理任何数据")
            
            with data_tabs[3]:
                # 获取所有处理结果，使用与结果查看页面一致的方法
                all_results = cache_manager.get_results_by_filter({})
                
                # 准备数据框
                if all_results:
                    cached_data = []
                    for item in all_results:
                        metadata = item['metadata']
                        result = item['result']
                        
                        relevant_keywords = result.get('relevant_keywords', [])
                        
                        cached_item = {
                            'title': metadata.get('title', ''),
                            'abstract': metadata.get('abstract', ''),
                            'year': metadata.get('year', ''),
                            'area': metadata.get('area', ''),
                            'method': metadata.get('method', ''),
                            'source': metadata.get('source', ''),
                            'success': result.get('success', False),
                            'relevant_keywords': relevant_keywords,
                            'num_keywords': len(relevant_keywords),
                            'cache_key': metadata.get('cache_key', '')
                        }
                        
                        cached_data.append(cached_item)
                    
                    cached_processed_df = pd.DataFrame(cached_data)
                    
                    # 添加缓存数据的筛选控件
                    cache_filter_col1, cache_filter_col2 = st.columns(2)
                    
                    with cache_filter_col1:
                        if 'area' in cached_processed_df.columns and not cached_processed_df['area'].empty:
                            cache_areas = ['全部'] + sorted(cached_processed_df['area'].unique().tolist())
                            cache_selected_area = st.selectbox("按领域筛选缓存:", cache_areas, key="cache_area_filter")
                        
                        if 'method' in cached_processed_df.columns and not cached_processed_df['method'].empty:
                            cache_methods = ['全部'] + sorted(cached_processed_df['method'].unique().tolist())
                            cache_selected_method = st.selectbox("按方法筛选缓存:", cache_methods, key="cache_method_filter")
                    
                    with cache_filter_col2:
                        if 'year' in cached_processed_df.columns and not cached_processed_df['year'].empty:
                            cache_years = sorted(cached_processed_df['year'].unique().tolist())
                            if cache_years:
                                cache_min_year, cache_max_year = min(cache_years), max(cache_years)
                                # 检查最小值和最大值是否相同
                                if cache_min_year == cache_max_year:
                                    st.write(f"年份: {cache_min_year}（所有文档年份相同）")
                                else:
                                    cache_year_range = st.slider(
                                        "按年份筛选缓存:", 
                                        min_value=cache_min_year, 
                                        max_value=cache_max_year, 
                                        value=(cache_min_year, cache_max_year),
                                        key="cache_year_filter"
                                    )
                    
                    # 应用缓存数据的筛选
                    filtered_cache_df = cached_processed_df.copy()
                    
                    if 'area' in cached_processed_df.columns and cache_selected_area != '全部':
                        filtered_cache_df = filtered_cache_df[filtered_cache_df['area'] == cache_selected_area]
                    
                    if 'method' in cached_processed_df.columns and cache_selected_method != '全部':
                        filtered_cache_df = filtered_cache_df[filtered_cache_df['method'] == cache_selected_method]
                    
                    if 'year' in cached_processed_df.columns and cache_years:
                        # 只有当最小值和最大值不同时才应用年份筛选
                        if cache_min_year != cache_max_year:
                            # 确保year列的数据类型是数值型
                            filtered_cache_df['year'] = pd.to_numeric(filtered_cache_df['year'], errors='coerce')
                            filtered_cache_df = filtered_cache_df[(filtered_cache_df['year'] >= cache_year_range[0]) & 
                                                             (filtered_cache_df['year'] <= cache_year_range[1])]
                    
                    # 添加删除功能
                    if "to_delete_results" not in st.session_state:
                        st.session_state.to_delete_results = []
                    
                    # 显示筛选后的缓存数据
                    col1, col2 = st.columns([9, 1])
                    with col1:
                        st.write(f"共找到 {len(filtered_cache_df)} 条缓存结果")
                    with col2:
                        if st.button("批量删除", key="batch_delete_cached", disabled=len(st.session_state.to_delete_results) == 0):
                            delete_count = 0
                            for cache_key in st.session_state.to_delete_results:
                                if cache_manager.delete_result(cache_key):
                                    delete_count += 1
                            if delete_count > 0:
                                st.success(f"成功删除{delete_count}条结果")
                                st.session_state.to_delete_results = []
                                time.sleep(1)
                                st.rerun()
                    
                    # 显示数据表格，添加复选框用于删除操作
                    if not filtered_cache_df.empty:
                        # 使用通用分页函数替代直接实现
                        current_df, current_page, total_pages, start_idx, end_idx = paginate_dataframe(filtered_cache_df, "cached")
                        
                        st.write(f"筛选结果 ({len(filtered_cache_df)}条，显示第 {start_idx+1}-{end_idx} 条):")
                        
                        # 分页导航
                        render_pagination_controls("cached", total_pages, current_page)
                        
                        # 全选/取消全选按钮
                        select_all = st.checkbox("全选当前页", key="select_all_cached")
                        
                        # 显示数据表格
                        for i, row in current_df.iterrows():
                            col1, col2 = st.columns([1, 11])
                            
                            cache_key = row['cache_key']
                            with col1:
                                is_selected = cache_key in st.session_state.to_delete_results
                                if st.checkbox("", value=is_selected or select_all, key=f"select_cached_{cache_key}"):
                                    if cache_key not in st.session_state.to_delete_results:
                                        st.session_state.to_delete_results.append(cache_key)
                                else:
                                    if cache_key in st.session_state.to_delete_results:
                                        st.session_state.to_delete_results.remove(cache_key)
                            
                            with col2:
                                with st.expander(f"{row['title'][:50]}..."):
                                    st.write(f"**年份**: {row['year']}")
                                    st.write(f"**领域**: {row['area']}")
                                    st.write(f"**方法**: {row['method']}")
                                    st.write(f"**关键词数量**: {row['num_keywords']}")
                                    
                                    # 显示关键词
                                    if row['num_keywords'] > 0:
                                        st.write(f"**关键词**: {', '.join(row['relevant_keywords'])}")
                                    
                                    # 单独删除按钮
                                    if st.button("删除", key=f"delete_cached_{cache_key}"):
                                        if cache_manager.delete_result(cache_key):
                                            st.success("已删除")
                                            if cache_key in st.session_state.to_delete_results:
                                                st.session_state.to_delete_results.remove(cache_key)
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error("删除失败")
                    else:
                        st.info("没有符合筛选条件的缓存数据")
                    
                    # 提供清空缓存的选项
                    if st.button("清空所有缓存", key="clear_cache_btn"):
                        if st.session_state.get("confirm_clear_cache", False):
                            cache_manager.clear_all_results()
                            st.success("所有缓存已清空")
                            st.session_state.confirm_clear_cache = False
                            st.rerun()
                        else:
                            st.session_state.confirm_clear_cache = True
                            st.warning("确定要清空所有缓存吗？这将删除所有已处理的结果。点击再次确认。")
                else:
                    st.info("缓存中没有已处理数据")
            
            # 如果正在处理，则启动处理逻辑
            if is_processing and st.session_state.processing_queue:
                # 保存处理参数到会话状态，以便重新加载页面后仍能访问
                if "batch_size" not in st.session_state:
                    st.session_state.batch_size = 10  # 默认值
                if "max_concurrent" not in st.session_state:
                    st.session_state.max_concurrent = 5  # 默认值
                
                # 创建进度显示区域
                progress_container = st.container()
                
                with progress_container:
                    # 创建LLM处理器
                    processor = get_llm_processor()
                    processor.set_prompts(st.session_state.system_prompt, st.session_state.user_prompt_template)
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 计算总进度
                    total_items = len(st.session_state.processing_queue) + len(st.session_state.processed_items)
                    current_progress = len(st.session_state.processed_items) / total_items if total_items > 0 else 0
                    progress_bar.progress(current_progress)
                    status_text.text(f"总进度: {len(st.session_state.processed_items)}/{total_items} ({current_progress*100:.1f}%)")
                    
                    # 创建处理状态区域
                    processing_status = st.empty()
                    item_progress = st.empty()
                    item_progress.text("正在处理中...")
                    
                    # 执行处理
                    try:
                        # 确定本次批处理的数量
                        current_batch_size = min(st.session_state.batch_size, len(st.session_state.processing_queue))
                        processing_status.info(f"正在并行处理 {current_batch_size} 条数据，最大并发请求数: {st.session_state.max_concurrent}")
                        
                        # 准备批处理数据
                        batch_items = st.session_state.processing_queue[:current_batch_size]
                        batch_df = pd.DataFrame(batch_items)
                        
                        # 处理进度回调函数
                        def on_progress(processed, total, result):
                            progress = processed / total
                            item_progress.progress(progress)
                            item_progress.text(f"批处理进度: {processed}/{total} ({progress*100:.1f}%)")
                        
                        # 批量处理数据
                        with st.spinner("正在处理数据批次..."):
                            result_df = processor.process_dataframe(
                                batch_df,
                                st.session_state.selected_keywords,
                                on_progress,
                                current_batch_size,  # 使用设置的批次大小
                                st.session_state.max_concurrent      # 使用设置的最大并发数
                            )
                            
                            # 将结果添加到已处理列表
                            for _, row in result_df.iterrows():
                                processed_item = row.to_dict()
                                st.session_state.processed_items.append(processed_item)
                                
                                # 缓存结果
                                if processed_item.get('success'):
                                    cache_key = cache_manager.generate_cache_key(
                                        processed_item['title'],
                                        processed_item['abstract'],
                                        st.session_state.selected_keywords
                                    )
                                    
                                    # 解析结果
                                    result = {
                                        "success": processed_item['success'],
                                        "relevant_keywords": json.loads(processed_item['relevant_keywords']) if isinstance(processed_item['relevant_keywords'], str) else processed_item['relevant_keywords'],
                                        "explanations": json.loads(processed_item['explanations']) if isinstance(processed_item['explanations'], str) else processed_item['explanations'],
                                        "raw_response": processed_item['raw_response']
                                    }
                                    
                                    # 准备元数据
                                    metadata = {
                                        "id": processed_item.get('id'),
                                        "title": processed_item['title'],
                                        "abstract": processed_item['abstract'],
                                        "year": int(processed_item['year']) if not pd.isna(processed_item['year']) else None,
                                        "source": processed_item['source'],
                                        "area": processed_item['area'],
                                        "method": processed_item['method'],
                                        "keywords": st.session_state.selected_keywords,  # 添加选择的关键词
                                        "cache_key": cache_key  # 添加缓存键
                                    }
                                    
                                    # 保存到缓存
                                    cache_manager.save_result(cache_key, result, metadata)
                        
                        # 从处理队列中移除已处理项
                        st.session_state.processing_queue = st.session_state.processing_queue[current_batch_size:]
                        
                        # 如果队列为空，则处理完成
                        if not st.session_state.processing_queue:
                            st.session_state.is_processing = False
                            st.success(f"处理完成! 共处理 {len(st.session_state.processed_items)} 条数据。")
                        else:
                            status_text.text(f"总进度: {len(st.session_state.processed_items)}/{total_items} ({len(st.session_state.processed_items)/total_items*100:.1f}%)")
                        
                        # 重新加载页面以更新状态
                        time.sleep(1)  # 稍微延迟以确保用户能看到状态
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"处理数据时出错: {str(e)}")
                        # 发生错误时也将当前项从队列中移除，以防止无限循环
                        if st.session_state.processing_queue:
                            st.session_state.processing_queue.pop(0)
                        # 清除当前处理项
                        st.session_state.current_processing = None
                        # 重新加载页面
                        time.sleep(2)  # 稍微延迟以确保用户能看到错误
                        st.rerun()
        else:
            st.warning("请先加载数据。")
    
    with main_col2:
        st.subheader("处理状态")
        
        # 显示当前处理的统计信息
        stats_container = st.container()
        with stats_container:
            if is_processing or st.session_state.processed_items:
                total_count = len(st.session_state.processing_queue) + len(st.session_state.processed_items)
                processed_count = len(st.session_state.processed_items)
                remaining_count = len(st.session_state.processing_queue)
                
                progress_percentage = (processed_count / total_count) * 100 if total_count > 0 else 0
                
                st.metric("总数据量", total_count)
                st.metric("已处理", processed_count)
                st.metric("待处理", remaining_count)
                st.metric("完成百分比", f"{progress_percentage:.1f}%")
                
                # 当前处理状态
                if is_processing and st.session_state.current_processing:
                    st.subheader("当前处理中")
                    current = st.session_state.current_processing
                    st.markdown(f"**标题**: {current['title'][:50]}...")
                    st.markdown(f"**年份**: {current.get('year', 'N/A')}")
                    st.markdown(f"**领域**: {current.get('area', 'N/A')}")
                    st.markdown(f"**方法**: {current.get('method', 'N/A')}")
            else:
                st.info("尚未开始处理数据")
        
        # 显示关键词匹配统计信息
        if st.session_state.processed_items:
            st.subheader("关键词匹配统计")
            
            # 统计关键词匹配情况
            keyword_matches = {}
            for item in st.session_state.processed_items:
                if item.get('success'):
                    relevant_keywords = json.loads(item['relevant_keywords']) if isinstance(item['relevant_keywords'], str) else item['relevant_keywords']
                    for kw in relevant_keywords:
                        keyword_matches[kw] = keyword_matches.get(kw, 0) + 1
            
            if keyword_matches:
                keyword_df = pd.DataFrame({
                    "keyword": list(keyword_matches.keys()),
                    "count": list(keyword_matches.values())
                }).sort_values("count", ascending=False)
                
                # 显示关键词匹配频率图表
                fig = px.bar(keyword_df, x="keyword", y="count", title="关键词匹配频率")
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示详细的统计表格
                st.write("关键词匹配详情:")
                st.dataframe(keyword_df)

# 添加删除缓存结果的方法到缓存管理器
def add_delete_result_method():
    cache_manager = get_cache_manager()
    
    # 删除单个缓存结果
    def delete_result(cache_key):
        """删除特定的缓存结果"""
        if not cache_key:
            return False
            
        try:
            # 检查结果文件是否存在
            result_path = os.path.join(cache_manager.results_dir, f"{cache_key}.json")
            metadata_path = os.path.join(cache_manager.metadata_dir, f"{cache_key}.json")
            
            deleted = False
            
            # 删除结果文件
            if os.path.exists(result_path):
                os.remove(result_path)
                deleted = True
            
            # 删除元数据文件
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                deleted = True
            
            # 如果有标注，也删除标注
            annotation_path = os.path.join(cache_manager.annotations_dir, f"{cache_key}.json")
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
                deleted = True
            
            return deleted
        except Exception as e:
            print(f"删除缓存结果时出错: {str(e)}")
            return False
    
    # 清空所有结果的增强版本
    def clear_all_results():
        """清空所有处理结果"""
        try:
            # 原始方法只删除了results_dir中的文件
            # 增强版本同时删除元数据和标注
            results_count = 0
            
            # 获取所有结果文件
            if os.path.exists(cache_manager.results_dir):
                for filename in os.listdir(cache_manager.results_dir):
                    if filename.endswith('.json'):
                        # 提取cache_key
                        cache_key = filename[:-5]  # 去掉.json后缀
                        if delete_result(cache_key):
                            results_count += 1
            
            return results_count
        except Exception as e:
            print(f"清空所有结果时出错: {str(e)}")
            return 0
    
    # 添加方法到cache_manager
    cache_manager.delete_result = delete_result
    # 增强clear_all_results方法
    cache_manager.clear_all_results = clear_all_results
    
    return cache_manager

# 添加缺失的结果查看页面函数
def render_results_view_page():
    st.header("📋 结果查看")
    
    # 获取缓存管理器
    cache_manager = get_cache_manager()
    # 添加删除结果方法
    cache_manager = add_delete_result_method()
    
    # 获取所有已处理的项目
    all_items = cache_manager.get_all_processed_items()
    
    if not all_items:
        st.info("暂无处理结果。请先在LLM处理页面处理数据。")
        return
    
    # 筛选功能
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("筛选条件")
        
        # 提取领域和方法列表
        areas = sorted(set(item.get('area') for item in all_items if item.get('area')))
        methods = sorted(set(item.get('method') for item in all_items if item.get('method')))
        sources = sorted(set(item.get('source') for item in all_items if item.get('source')))
        
        # 按领域筛选
        selected_area = st.selectbox("领域:", ["全部"] + areas)
        # 按方法筛选
        selected_method = st.selectbox("方法:", ["全部"] + methods)
        # 按数据源筛选
        selected_source = st.selectbox("数据源:", ["全部"] + sources)
        
        # 按标注筛选
        annotations = cache_manager.get_all_annotations()
        has_annotations = bool(annotations)
        
        if has_annotations:
            annotation_filter = st.radio("标注状态:", ["全部", "已标注", "未标注"])
        
        # 应用筛选
        filter_criteria = {}
        if selected_area != "全部":
            filter_criteria['area'] = selected_area
        if selected_method != "全部":
            filter_criteria['method'] = selected_method
        if selected_source != "全部":
            filter_criteria['source'] = selected_source
        
        # 获取筛选后的结果
        filtered_results = cache_manager.get_results_by_filter(filter_criteria)
        
        # 应用标注筛选
        if has_annotations and annotation_filter != "全部":
            if annotation_filter == "已标注":
                filtered_results = [r for r in filtered_results if r['metadata']['cache_key'] in annotations]
            else:  # 未标注
                filtered_results = [r for r in filtered_results if r['metadata']['cache_key'] not in annotations]
        
        st.write(f"共找到 {len(filtered_results)} 条结果")
        
        # 视图控制按钮
        view_col1, view_col2 = st.columns(2)
        
        # 随机查看按钮
        with view_col1:
            if st.button("随机查看一条"):
                if filtered_results:
                    random_idx = random.randint(0, len(filtered_results) - 1)
                    st.session_state.selected_result = filtered_results[random_idx]
                    st.session_state.show_detail_view = True
                    st.rerun()  # 立即更新UI
        
        # 查看列表按钮
        with view_col2:
            if st.button("查看列表"):
                st.session_state.show_detail_view = False
                st.session_state.selected_result = None
                st.rerun()  # 立即更新UI
        
        # 添加删除选中项功能
        if "to_delete_results" not in st.session_state:
            st.session_state.to_delete_results = []
        
        st.subheader("批量操作")
        if len(st.session_state.to_delete_results) > 0:
            st.write(f"已选择 {len(st.session_state.to_delete_results)} 条结果待删除")
            
        if st.button("删除选中项", disabled=len(st.session_state.to_delete_results) == 0):
            try:
                delete_count = 0
                for cache_key in st.session_state.to_delete_results:
                    # 删除缓存结果文件
                    if cache_manager.delete_result(cache_key):
                        delete_count += 1
                
                if delete_count > 0:
                    st.success(f"成功删除{delete_count}条结果")
                    st.session_state.to_delete_results = []  # 清空选择
                    time.sleep(1)
                    # 如果当前正在详情视图并且删除了该结果，返回列表视图
                    if st.session_state.show_detail_view and st.session_state.selected_result:
                        deleted_key = st.session_state.selected_result['metadata'].get('cache_key', '')
                        if deleted_key in st.session_state.to_delete_results:
                            st.session_state.show_detail_view = False
                            st.session_state.selected_result = None
                    
                    st.rerun()
                else:
                    st.error("删除失败")
            except Exception as e:
                st.error(f"删除结果时出错: {str(e)}")
    
    with col2:
        # 查看模式切换
        view_mode = st.radio("查看模式:", ["详情视图", "列表视图"], horizontal=True,
                            index=0 if st.session_state.show_detail_view else 1)
        
        # 根据选择更新视图状态
        if st.session_state.show_detail_view != (view_mode == "详情视图"):
            st.session_state.show_detail_view = (view_mode == "详情视图")
            # 如果切换到列表视图，清除选择的结果
            if view_mode == "列表视图":
                st.session_state.selected_result = None
            st.rerun()  # 立即更新UI
        
        # 根据当前状态显示详情或列表
        if st.session_state.show_detail_view and st.session_state.selected_result:
            # 显示详情视图
            st.subheader("结果详情")
            
            # 添加返回列表按钮
            if st.button("返回列表"):
                st.session_state.show_detail_view = False
                st.session_state.selected_result = None
                st.rerun()
            else:
                selected = st.session_state.selected_result
                metadata = selected['metadata']
                result = selected['result']
                
                st.markdown(f"**标题**: {metadata.get('title', '')}")
                st.markdown(f"**摘要**: {metadata.get('abstract', '')}")
                st.markdown(f"**年份**: {metadata.get('year', '')}")
                st.markdown(f"**领域**: {metadata.get('area', '')}")
                st.markdown(f"**方法**: {metadata.get('method', '')}")
                
                # 显示关键词匹配结果
                st.subheader("关键词匹配结果")
                
                relevant_keywords = result.get('relevant_keywords', [])
                explanations = result.get('explanations', {})
                
                if relevant_keywords:
                    for keyword in relevant_keywords:
                        explanation = explanations.get(keyword, "")
                        st.markdown(f"**{keyword}**: {explanation}")
                else:
                    reason = explanations.get('reason', "未提供原因")
                    st.markdown(f"**无匹配关键词**: {reason}")
                
                # 显示标注界面
                st.subheader("人工标注")
                
                cache_key = metadata.get('cache_key')
                annotation = cache_manager.get_annotation(cache_key) if cache_key else None
                
                is_correct = st.radio(
                    "LLM判断是否正确:",
                    ["正确", "部分正确", "不正确"],
                    index=0 if not annotation else (0 if annotation.get('is_correct') == "正确" else 
                                                  1 if annotation.get('is_correct') == "部分正确" else 2)
                )
                
                feedback = st.text_area(
                    "标注反馈:",
                    value="" if not annotation else annotation.get('feedback', ""),
                    height=100
                )
                
                if st.button("保存标注"):
                    try:
                        annotation_data = {
                            "is_correct": is_correct,
                            "feedback": feedback
                        }
                        
                        if cache_manager.save_annotation(cache_key, annotation_data):
                            st.success("标注已保存。")
                            # 更新session状态中的标注结果
                            if 'annotation_results' not in st.session_state:
                                st.session_state.annotation_results = {}
                            st.session_state.annotation_results[cache_key] = annotation_data
                        else:
                            st.error("保存标注时出错。")
                    except Exception as e:
                        st.error(f"保存标注时出错: {str(e)}")
        else:
            # 显示结果列表
            st.subheader("结果列表")
            
            if filtered_results:
                # 使用通用分页函数
                # 将列表转换为DataFrame以使用通用分页函数
                results_df = pd.DataFrame([r['metadata'] for r in filtered_results])
                current_df, current_page, total_pages, start_idx, end_idx = paginate_dataframe(results_df, "results_list")
                
                # 对应的当前页结果项
                current_page_items = [filtered_results[i] for i in range(start_idx, end_idx)]
                
                st.write(f"结果 ({len(filtered_results)}条，显示第 {start_idx+1}-{end_idx} 条):")
                
                # 分页导航
                render_pagination_controls("results_list", total_pages, current_page)
                
                # 全选/取消全选按钮
                select_all = st.checkbox("全选当前页", key="select_all_results")
                if select_all:
                    # 将当前页所有项添加到待删除列表
                    for item in current_page_items:
                        cache_key = item['metadata'].get('cache_key', '')
                        if cache_key and cache_key not in st.session_state.to_delete_results:
                            st.session_state.to_delete_results.append(cache_key)
                
                # 显示结果列表
                for i, item in enumerate(current_page_items):
                    metadata = item['metadata']
                    result = item['result']
                    cache_key = metadata.get('cache_key', '')
                    
                    # 创建包含复选框的行
                    col1, col2 = st.columns([1, 11])
                    
                    # 选择复选框
                    with col1:
                        is_selected = cache_key in st.session_state.to_delete_results
                        if st.checkbox("", value=is_selected or select_all, key=f"select_{cache_key}"):
                            if cache_key not in st.session_state.to_delete_results:
                                st.session_state.to_delete_results.append(cache_key)
                        else:
                            if cache_key in st.session_state.to_delete_results:
                                st.session_state.to_delete_results.remove(cache_key)
                    
                    # 显示数据行和按钮
                    with col2:
                        with st.expander(f"结果 {start_idx + i + 1}: {metadata.get('title', '')[:50]}..."):
                            if st.button("查看详情", key=f"view_{i}"):
                                st.session_state.selected_result = item
                                st.session_state.show_detail_view = True
                                st.rerun()
                            
                            # 显示基本信息
                            st.markdown(f"**年份**: {metadata.get('year', '')}")
                            st.markdown(f"**领域**: {metadata.get('area', '')}")
                            st.markdown(f"**方法**: {metadata.get('method', '')}")
                            
                            # 显示关键词统计
                            relevant_keywords = result.get('relevant_keywords', [])
                            if relevant_keywords:
                                st.markdown(f"**相关关键词数量**: {len(relevant_keywords)}")
                                st.markdown(f"**关键词**: {', '.join(relevant_keywords)}")
                            else:
                                st.markdown("**无匹配关键词**")
                            
                            # 删除按钮
                            if st.button("删除", key=f"delete_{i}"):
                                try:
                                    if cache_manager.delete_result(cache_key):
                                        st.success("已删除")
                                        if cache_key in st.session_state.to_delete_results:
                                            st.session_state.to_delete_results.remove(cache_key)
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("删除失败")
                                except Exception as e:
                                    st.error(f"删除结果时出错: {str(e)}")
                
                # 检查当前页是否有需要从删除列表中移除的项
                if not select_all:
                    current_page_keys = [item['metadata'].get('cache_key', '') for item in current_page_items]
                    for cache_key in list(st.session_state.to_delete_results):  # 使用副本遍历
                        if cache_key in current_page_keys:
                            # 这个键在当前页面上，但没有被选中（因为不是全选状态）
                            # 检查对应的复选框是否未被选中
                            checkbox_key = f"select_{cache_key}"
                            if checkbox_key in st.session_state and not st.session_state[checkbox_key]:
                                st.session_state.to_delete_results.remove(cache_key)
                
                if len(filtered_results) > 10:
                    st.info(f"当前显示第 {start_idx + 1} - {end_idx} 条，共 {len(filtered_results)} 条结果")
            else:
                st.info("没有找到符合条件的结果")

# 统计分析页面
def render_statistics_page():
    st.header("📊 统计分析")
    
    # 获取缓存管理器
    cache_manager = get_cache_manager()
    
    # 获取所有已处理的项目
    all_items = cache_manager.get_all_processed_items()
    
    if not all_items:
        st.info("暂无处理结果。请先在LLM处理页面处理数据。")
        return
    
    # 统计分析部分
    st.subheader("处理结果统计")
    
    # 确保获取的数据格式正确
    # 将结果转换为标准格式
    processed_results = []
    for item in cache_manager.get_results_by_filter({}):
        if 'metadata' in item and 'result' in item:
            processed_results.append(item)
    
    if not processed_results:
        st.info("没有找到格式正确的处理结果。")
        return
    
    # 统计成功率
    success_rate = sum(1 for item in processed_results if item['result'].get('success', False)) / len(processed_results)
    st.write(f"成功率: {success_rate:.2%}")
    
    # 统计关键词数量
    total_keywords = sum(len(item['result'].get('relevant_keywords', [])) for item in processed_results)
    st.write(f"总关键词数量: {total_keywords}")
    
    # 文章总数
    total_papers = len(processed_results)
    st.write(f"文章总数: {total_papers}")
    
    # 统计不同领域和方法的处理结果
    area_counts = {}
    method_counts = {}
    
    for item in processed_results:
        metadata = item['metadata']
        area = metadata.get('area', '未知')
        method = metadata.get('method', '未知')
        
        area_counts[area] = area_counts.get(area, 0) + 1
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # 转换为DataFrame
    area_df = pd.DataFrame({'area': list(area_counts.keys()), 'count': list(area_counts.values())})
    method_df = pd.DataFrame({'method': list(method_counts.keys()), 'count': list(method_counts.values())})
    
    # 排序
    if not area_df.empty:
        area_df = area_df.sort_values('count', ascending=False)
    if not method_df.empty:
        method_df = method_df.sort_values('count', ascending=False)
    
    # 添加合计行
    if not area_df.empty:
        total_row = pd.DataFrame({'area': ['总计'], 'count': [area_df['count'].sum()]})
        area_df = pd.concat([area_df, total_row]).reset_index(drop=True)
    
    if not method_df.empty:
        total_row = pd.DataFrame({'method': ['总计'], 'count': [method_df['count'].sum()]})
        method_df = pd.concat([method_df, total_row]).reset_index(drop=True)
    
    # 改进列名
    if not area_df.empty:
        area_df.columns = ['领域', '文章数量']
    if not method_df.empty:
        method_df.columns = ['方法', '文章数量']
    
    st.subheader("按领域统计")
    if not area_df.empty:
        st.dataframe(area_df, use_container_width=True)
    else:
        st.info("没有领域数据")
    
    st.subheader("按方法统计")
    if not method_df.empty:
        st.dataframe(method_df, use_container_width=True)
    else:
        st.info("没有方法数据")
    
    # 按关键词绘制发文量逐年累计图
    st.subheader("关键词发文量逐年累计图")
    
    # 准备用于绘图的数据
    # 从处理结果中提取年份、领域和关键词信息
    plot_data = []
    
    for item in processed_results:
        if 'metadata' not in item or 'result' not in item:
            continue
        
        metadata = item['metadata']
        result = item['result']
        
        year = metadata.get('year')
        method = metadata.get('method', '未知')
        source = metadata.get('source', '未知')
        relevant_keywords = result.get('relevant_keywords', [])
        
        # 跳过没有年份或关键词的数据
        if year is None or not relevant_keywords:
            continue
        
        # 转换为整数年份
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        
        # 添加到绘图数据中
        plot_data.append({
            'year': year,
            'method': method,
            'source': source,
            'keywords': relevant_keywords
        })
    
    if not plot_data:
        st.info("没有足够的数据绘制图表。")
        return
    
    # 转换为DataFrame以便统计
    plot_df = pd.DataFrame(plot_data)
    
    # 提取唯一年份和方法
    years = sorted(plot_df['year'].unique())
    methods = ["machine learning", "deep learning", "LLMs"]
    sources = sorted(plot_df['source'].unique())
    
    # 用户控制
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        # 选择数据源
        selected_sources = st.multiselect(
            "选择数据源:",
            options=sources,
            default=sources
        )
        
        # 选择图表类型
        chart_type = st.radio(
            "图表类型:",
            ["年度发文量", "累计发文量"],
            index=1  # 默认选择累计发文量
        )
    
    with control_col2:
        # 为每个方法选择年份范围
        st.write("年份范围筛选:")
        min_year, max_year = min(years), max(years)
        
        # 机器学习年份范围
        ml_min_year, ml_max_year = st.slider(
            "机器学习年份范围:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="ml_year_range"
        )
        
        # 深度学习年份范围
        dl_min_year, dl_max_year = st.slider(
            "深度学习年份范围:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="dl_year_range"
        )
        
        # LLM年份范围
        llm_min_year, llm_max_year = st.slider(
            "LLM年份范围:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            key="llm_year_range"
        )
    
    # 年份范围映射
    year_ranges = {
        "machine learning": (ml_min_year, ml_max_year),
        "deep learning": (dl_min_year, dl_max_year),
        "LLMs": (llm_min_year, llm_max_year)
    }
    
    # 筛选数据
    if selected_sources:
        plot_df = plot_df[plot_df['source'].isin(selected_sources)]
    
    if plot_df.empty:
        st.info("筛选后没有数据可显示。")
        return
    
    # 定义方法的配色和名称
    method_properties = {
        "machine learning": {
            "color": "#FF5733",  # 橙红色
            "display_name": "Machine Learning",
            "marker_symbol": "circle",
            "line_dash": "solid",
            "opacity": 0.7
        },
        "deep learning": {
            "color": "#3498DB",  # 蓝色
            "display_name": "Deep Learning",
            "marker_symbol": "square",
            "line_dash": "solid",
            "opacity": 0.7
        },
        "LLMs": {
            "color": "#2ECC71",  # 绿色
            "display_name": "Large Language Models",
            "marker_symbol": "star",
            "line_dash": "solid",
            "opacity": 0.7
        }
    }
    
    # 创建图表
    fig = px.line(title=f"关键词发文量{'累计' if chart_type == '累计发文量' else '年度'}图")
    
    # 处理每个方法的数据
    for method in methods:
        # 筛选当前方法的年份范围
        min_year, max_year = year_ranges[method]
        year_range = list(range(min_year, max_year + 1))
        
        # 如果年份范围为空，则跳过
        if not year_range:
            continue
        
        # 计算每年的文章数量，并考虑论文重叠问题
        yearly_counts = {year: 0 for year in year_range}
        
        for _, row in plot_df.iterrows():
            # 如果年份不在范围内，跳过
            if row['year'] not in year_range:
                continue
            
            # 获取当前论文的关键词和方法
            keywords = row['keywords']
            paper_method = row['method']
            
            # 判断是否应该计入当前方法
            should_count = False
            
            if method == "LLMs":
                # LLM论文只计入LLM方法
                should_count = "LLMs" in keywords or paper_method == "LLMs"
            elif method == "deep learning":
                # 深度学习论文计入深度学习方法，但排除LLM论文
                should_count = (("deep learning" in keywords or paper_method == "deep learning") and 
                               not ("LLMs" in keywords or paper_method == "LLMs"))
            elif method == "machine learning":
                # 机器学习论文计入机器学习方法，但排除深度学习和LLM论文
                should_count = (("machine learning" in keywords or paper_method == "machine learning") and 
                               not ("deep learning" in keywords or paper_method == "deep learning") and
                               not ("LLMs" in keywords or paper_method == "LLMs"))
            
            if should_count:
                yearly_counts[row['year']] += 1
        
        # 将计数转换为列表
        years_list = list(yearly_counts.keys())
        counts_list = list(yearly_counts.values())
        
        # 如果是累计图，计算累计值
        if chart_type == "累计发文量":
            cumulative_counts = []
            running_sum = 0
            for count in counts_list:
                running_sum += count
                cumulative_counts.append(running_sum)
            counts_list = cumulative_counts
        
        # 获取方法属性
        props = method_properties[method]
        
        # 添加到图表
        fig.add_trace(go.Scatter(
            x=years_list,
            y=counts_list,
            mode='lines+markers',
            name=props["display_name"],
            line=dict(color=props["color"], dash=props["line_dash"]),
            marker=dict(symbol=props["marker_symbol"], size=8),
            opacity=props["opacity"]
        ))
    
    # 设置图表布局
    fig.update_layout(
        xaxis_title="年份",
        yaxis_title="论文数量",
        legend_title="方法",
        template="plotly_white",
        height=500
    )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加下载图表按钮
    st.download_button(
        label="下载图表 (HTML)",
        data=fig.to_html(),
        file_name=f"keyword_trend_{'cumulative' if chart_type == '累计发文量' else 'yearly'}.html",
        mime="text/html"
    )

def main():
    # 初始化Session状态
    init_session_state()
    
    # 初始化缓存管理器和方法
    cache_manager = add_data_cache_methods()
    cache_manager = add_delete_result_method()
    
    # 加载默认提示词
    if not st.session_state.system_prompt or not st.session_state.user_prompt_template:
        load_default_prompts()
    
    # 加载已保存的关键词列表
    saved_keyword_lists = cache_manager.get_all_keyword_lists()
    for name, keywords in saved_keyword_lists.items():
        st.session_state.keyword_lists[name] = keywords
    
    # 更新会话时间
    if st.session_state.last_session_time:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.last_session_time = current_time
    
    # 设置页面标题
    st.title("📚 大模型论文关键词匹配")
    
    # 侧边栏导航
    page = st.sidebar.radio(
        "导航",
        ["📊 数据加载", "🔑 关键词管理", "🤖 LLM处理", "📝 结果查看", "📈 统计分析", "⚙️ 提示词管理"]
    )
    
    # 显示当前数据状态
    with st.sidebar.expander("当前状态", expanded=False):
        if st.session_state.loaded_data is not None:
            st.write(f"已加载 {len(st.session_state.loaded_data)} 条数据")
        else:
            st.write("未加载数据")
        
        if st.session_state.selected_keywords:
            st.write(f"已选择 {len(st.session_state.selected_keywords)} 个关键词")
        else:
            st.write("未选择关键词")
        
        # 从缓存管理器获取处理结果数量
        cache_manager = get_cache_manager()
        processed_count = len(cache_manager.get_results_by_filter({}))
        st.write(f"已处理 {processed_count} 条结果")
    
    try:
        # 根据选择的页面渲染相应的内容
        if page == "📊 数据加载":
            render_data_loading_page()
        elif page == "🔑 关键词管理":
            render_keywords_management_page()
        elif page == "🤖 LLM处理":
            render_llm_processing_page()
        elif page == "📝 结果查看":
            render_results_view_page()
        elif page == "📈 统计分析":
            render_statistics_page()
        elif page == "⚙️ 提示词管理":
            render_prompts_management_page()
        
        # 显示页脚
        st.markdown("---")
        st.markdown("📚 **大模型论文关键词匹配**")
    
    except Exception as e:
        st.error(f"应用程序发生错误: {str(e)}")
        import traceback
        st.exception(traceback.format_exc())


if __name__ == "__main__":
    main() 