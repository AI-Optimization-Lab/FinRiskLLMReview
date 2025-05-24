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

# 导入自定义模块 - 第二阶段处理相关
from utils.stage2_cache_manager import Stage2CacheManager
from utils.stage2_llm_processor import Stage2LLMProcessor

# 导入原始模块 - 第一阶段处理相关
from utils.data_loader import DataLoader
from utils.cache_manager import CacheManager

# 设置页面标题和配置
st.set_page_config(
    page_title="应用领域分类",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化数据加载器和缓存管理器
@st.cache_resource
def get_stage1_cache_manager():
    return CacheManager()

@st.cache_resource
def get_stage2_cache_manager():
    return Stage2CacheManager()

def get_stage2_llm_processor():
    api_key = st.session_state.get("api_key", "")
    processor = Stage2LLMProcessor(api_key=api_key)
    return processor

# 初始化Session状态
def init_session_state():
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "stage1_results" not in st.session_state:
        st.session_state.stage1_results = None
    if "stage1_filter_criteria" not in st.session_state:
        st.session_state.stage1_filter_criteria = {}
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "user_prompt_template" not in st.session_state:
        st.session_state.user_prompt_template = ""
    if "processing_results" not in st.session_state:
        st.session_state.processing_results = []
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "加载数据"
    if "annotation_results" not in st.session_state:
        st.session_state.annotation_results = {}
    
    # 添加视图控制状态
    if "show_detail_view" not in st.session_state:
        st.session_state.show_detail_view = False
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None
    
    # 添加处理状态控制
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False  # 是否正在处理数据
    if "current_processing" not in st.session_state:
        st.session_state.current_processing = None  # 当前正在处理的数据
    if "processed_items" not in st.session_state:
        st.session_state.processed_items = []  # 本次会话已处理的数据列表
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []  # 待处理队列
    
    # 分页控制
    if "display_page" not in st.session_state:
        st.session_state.display_page = {
            "unprocessed": 0,  # 未处理数据当前页码
            "processed": 0,    # 已处理数据当前页码
            "processing": 0,   # 正在处理数据当前页码
            "results_list": 0, # 结果列表当前页码
            "cached": 0,       # 缓存数据当前页码
            "page_size": 10    # 每页显示数量
        }
    
    # 其他控制状态
    if "to_delete_results" not in st.session_state:
        st.session_state.to_delete_results = []  # 待删除结果列表
    if "confirm_clear_cache" not in st.session_state:
        st.session_state.confirm_clear_cache = False  # 确认清空缓存的状态

# 加载默认提示词
def load_default_prompts():
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    default_prompts = os.path.join(prompts_dir, "stage2_prompts.json")
    if os.path.exists(default_prompts):
        try:
            with open(default_prompts, 'r', encoding='utf-8') as f:
                prompts = json.load(f)
                st.session_state.system_prompt = prompts.get('system_prompt', '')
                st.session_state.user_prompt_template = prompts.get('user_prompt_template', '')
                st.session_state.prompt_examples = prompts.get('examples', [])
        except Exception as e:
            st.error(f"加载默认提示词时出错: {str(e)}")

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
        return df, 0, 0, 0, 0
    
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
        show_columns = list(df.columns)
    
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

# 数据加载页面
def render_data_loading_page():
    st.header("📊 第一阶段数据加载")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("加载第一阶段处理结果")
        
        # 获取第一阶段和第二阶段的缓存管理器
        stage1_cache_manager = get_stage1_cache_manager()
        stage2_cache_manager = get_stage2_cache_manager()
        
        # 显示第一阶段已处理的数据总量
        stage1_results = stage1_cache_manager.get_all_processed_items()
        if stage1_results:
            st.info(f"第一阶段共处理了 {len(stage1_results)} 条数据")
            
            # 筛选条件
            st.subheader("筛选条件")
            
            # 提取所有可能的领域和方法
            areas = sorted(set(item.get('area') for item in stage1_results if item.get('area')))
            methods = sorted(set(item.get('method') for item in stage1_results if item.get('method')))
            sources = sorted(set(item.get('source') for item in stage1_results if item.get('source')))
            
            # 创建筛选控件
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # 按领域筛选
                selected_area = st.selectbox("按领域筛选:", ["全部"] + areas)
                
                # 按方法筛选
                selected_method = st.selectbox("按方法筛选:", ["全部"] + methods)
            
            with filter_col2:
                # 按数据源筛选
                selected_source = st.selectbox("按数据源筛选:", ["全部"] + sources)
                
                # 按关键词筛选
                include_keywords = st.text_input("包含关键词 (以逗号分隔):")
            
            # 排除已处理数据选项
            exclude_processed = st.checkbox("排除已在第二阶段处理的数据", value=True)
            
            # 添加筛选有无关键词匹配的选项
            keywords_filter_option = st.radio(
                "关键词匹配筛选:",
                ["全部论文", "仅显示有匹配关键词的论文", "仅显示无匹配关键词的论文"],
                index=0
            )
            
            # 根据用户输入构建筛选条件
            filter_criteria = {}
            
            if selected_area != "全部":
                filter_criteria['area'] = selected_area
            
            if selected_method != "全部":
                filter_criteria['method'] = selected_method
            
            if selected_source != "全部":
                filter_criteria['source'] = selected_source
            
            # 保存筛选条件到会话状态
            st.session_state.stage1_filter_criteria = filter_criteria
            
            # 加载按钮
            if st.button("加载符合条件的数据"):
                with st.spinner("正在加载第一阶段数据..."):
                    # 加载数据
                    stage1_results = stage2_cache_manager.load_stage1_results(filter_criteria)
                    
                    # 如果需要排除已处理的数据
                    if exclude_processed:
                        # 以下逻辑已经在Stage2CacheManager.load_stage1_results中处理
                        # 所以这里不需要额外的代码，只需要显示信息
                        pass
                    
                    # 如果有关键词筛选
                    if include_keywords:
                        keywords_list = [kw.strip() for kw in include_keywords.split(",") if kw.strip()]
                        if keywords_list:
                            # 过滤包含特定关键词的数据
                            filtered_results = []
                            for item in stage1_results:
                                relevant_keywords = item.get('relevant_keywords', [])
                                if any(kw in relevant_keywords for kw in keywords_list):
                                    filtered_results.append(item)
                            stage1_results = filtered_results
                    
                    # 应用关键词匹配筛选选项
                    if keywords_filter_option == "仅显示有匹配关键词的论文":
                        stage1_results = [item for item in stage1_results if item.get('relevant_keywords', [])]
                    elif keywords_filter_option == "仅显示无匹配关键词的论文":
                        stage1_results = [item for item in stage1_results if not item.get('relevant_keywords', [])]
                    
                    # 保存结果到会话状态
                    if stage1_results:
                        st.session_state.stage1_results = stage1_results
                        st.success(f"成功加载 {len(stage1_results)} 条符合条件的数据")
                        st.rerun()
                    else:
                        st.warning("没有找到符合条件的数据")
        else:
            st.warning("第一阶段没有处理结果，请先运行第一阶段处理。")
    
    with col2:
        st.subheader("已加载数据预览")
        
        if st.session_state.stage1_results:
            # 将结果转换为DataFrame以便显示
            data = []
            for item in st.session_state.stage1_results:
                data_item = {
                    'title': item.get('title', ''),
                    'year': item.get('year', ''),
                    'area': item.get('area', ''),
                    'method': item.get('method', ''),
                    'source': item.get('source', ''),
                    'keywords_count': len(item.get('relevant_keywords', [])),
                    'cache_key': item.get('cache_key', '')
                }
                data.append(data_item)
            
            df = pd.DataFrame(data)
            
            # 显示统计信息
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
                # 显示数据表格，使用分页
                show_columns = ['title', 'year', 'area', 'method', 'keywords_count']
                render_data_table(df, show_columns=show_columns, title="第一阶段处理结果", page_key="unprocessed")
                
                # 随机展示一条数据
                if st.button("随机展示一条数据"):
                    if len(df) > 0:
                        random_idx = random.randint(0, len(df) - 1)
                        random_row = df.iloc[random_idx]
                        random_item = next((item for item in st.session_state.stage1_results if item.get('cache_key') == random_row['cache_key']), None)
                        
                        if random_item:
                            st.subheader("随机数据样例")
                            st.markdown(f"**标题**: {random_item.get('title', '')}")
                            st.markdown(f"**摘要**: {random_item.get('abstract', '')}")
                            st.markdown(f"**年份**: {random_item.get('year', '')}")
                            st.markdown(f"**领域**: {random_item.get('area', '')}")
                            st.markdown(f"**方法**: {random_item.get('method', '')}")
                            st.markdown(f"**关键词**: {', '.join(random_item.get('relevant_keywords', []))}")
                    else:
                        st.error("数据集为空，无法展示随机数据。")
        else:
            st.info("请先加载第一阶段处理结果。")

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
                        st.success("已应用示例提示词")
                        st.rerun()
        
        st.subheader("预览格式化后的用户提示词")
        if st.session_state.stage1_results:
            # 随机选择一篇论文用于预览
            random_idx = random.randint(0, len(st.session_state.stage1_results) - 1)
            random_paper = st.session_state.stage1_results[random_idx]
            
            processor = get_stage2_llm_processor()
            processor.set_prompts(st.session_state.system_prompt, st.session_state.user_prompt_template)
            
            formatted_prompt = processor.format_user_prompt(
                random_paper.get('title', ''),
                random_paper.get('abstract', ''),
                random_paper.get('relevant_keywords', [])
            )
            
            st.text_area("预览:", formatted_prompt, height=300, disabled=True)
        else:
            st.info("请先加载第一阶段处理结果。") 

# LLM处理页面
def render_llm_processing_page():
    st.header("🤖 金融领域分类处理")
    
    # 创建主要的列布局
    main_col1, main_col2 = st.columns([2, 1])
    
    with main_col1:
        st.subheader("API设置")
        
        # 获取缓存管理器
        stage2_cache_manager = get_stage2_cache_manager()
        
        # 尝试从缓存加载API密钥
        api_key = st.text_input("DeepSeek API密钥:", st.session_state.api_key, type="password")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
        
        if not st.session_state.api_key:
            st.warning("请输入DeepSeek API密钥。")
        
        # 检查是否处于处理状态
        is_processing = st.session_state.is_processing
        
        # 处理设置部分
        st.subheader("处理设置")
        if st.session_state.stage1_results is not None and len(st.session_state.stage1_results) > 0:
            # 提取数据以便筛选
            papers = st.session_state.stage1_results
            
            # 筛选设置
            filter_section = st.container()
            
            with filter_section:
                if not is_processing:  # 只有在非处理状态才允许修改筛选条件
                    filter_col, process_col = st.columns(2)
                    
                    with filter_col:
                        # 设置处理参数
                        batch_size = st.slider("批次大小:", min_value=1, max_value=50, value=10, disabled=is_processing)
                        max_concurrent = st.slider("最大并发请求数:", min_value=1, max_value=50, value=5, disabled=is_processing)
                        
                        # 保存处理参数到会话状态
                        st.session_state.batch_size = batch_size
                        st.session_state.max_concurrent = max_concurrent
                    
                    with process_col:
                        # 随机抽样
                        sample_size = st.number_input("随机抽样数量 (0表示使用全部数据):", min_value=0, max_value=len(papers), value=min(20, len(papers)), disabled=is_processing)
                        
                        if sample_size > 0 and sample_size < len(papers) and not is_processing:
                            # 随机抽取
                            filtered_papers = random.sample(papers, sample_size)
                            st.write(f"已随机抽取 {len(filtered_papers)} 条数据。")
                        else:
                            filtered_papers = papers
                
                # 显示处理状态
                status_container = st.container()
                
                # 检查是否已设置提示词
                error_msgs = []
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
                    if st.button("开始处理", disabled=is_processing or len(filtered_papers) == 0):
                        # 将筛选后的数据转换为处理队列
                        st.session_state.processing_queue = filtered_papers
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
                    queue_data = []
                    for item in st.session_state.processing_queue:
                        queue_data.append({
                            'title': item.get('title', ''),
                            'year': item.get('year', ''),
                            'area': item.get('area', ''),
                            'method': item.get('method', ''),
                            'keywords_count': len(item.get('relevant_keywords', []))
                        })
                    
                    queue_df = pd.DataFrame(queue_data)
                    if not queue_df.empty:
                        render_data_table(queue_df, title="待处理数据", page_key="unprocessed")
                    else:
                        st.info("队列中没有待处理数据")
                else:
                    # 否则显示筛选后的数据
                    filtered_data = []
                    for item in filtered_papers:
                        filtered_data.append({
                            'title': item.get('title', ''),
                            'year': item.get('year', ''),
                            'area': item.get('area', ''),
                            'method': item.get('method', ''),
                            'keywords_count': len(item.get('relevant_keywords', []))
                        })
                    
                    filtered_df = pd.DataFrame(filtered_data)
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
                    
                    current_data = []
                    for item in batch_items:
                        current_data.append({
                            'title': item.get('title', ''),
                            'year': item.get('year', ''),
                            'area': item.get('area', ''),
                            'method': item.get('method', ''),
                            'keywords_count': len(item.get('relevant_keywords', []))
                        })
                    
                    current_df = pd.DataFrame(current_data)
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
                # 获取所有缓存的处理结果
                index_data = stage2_cache_manager.get_index()
                
                if index_data:
                    # 添加缓存数据的筛选控件
                    cache_filter_col1, cache_filter_col2 = st.columns(2)
                    
                    # 提取领域和方法列表
                    cached_areas = set()
                    cached_methods = set()
                    cached_domains = set()
                    
                    for item in index_data:
                        if 'area' in item and item['area']:
                            cached_areas.add(item['area'])
                        if 'method' in item and item['method']:
                            cached_methods.add(item['method'])
                        if 'application_domains' in item:
                            for domain in item['application_domains']:
                                cached_domains.add(domain)
                    
                    with cache_filter_col1:
                        # 按领域筛选
                        cache_selected_area = st.selectbox("按领域筛选缓存:", ["全部"] + sorted(list(cached_areas)), key="cache_area_filter")
                        
                        # 按方法筛选
                        cache_selected_method = st.selectbox("按方法筛选缓存:", ["全部"] + sorted(list(cached_methods)), key="cache_method_filter")
                    
                    with cache_filter_col2:
                        # 按应用领域筛选
                        cache_selected_domain = st.selectbox("按应用领域筛选:", ["全部"] + sorted(list(cached_domains)), key="cache_domain_filter")
                        
                        # 包含关键词
                        cache_keywords = st.text_input("包含关键词:", key="cache_keywords_filter")
                    
                    # 应用筛选
                    filtered_index = index_data.copy()
                    
                    # 按领域筛选
                    if cache_selected_area != "全部":
                        filtered_index = [item for item in filtered_index if item.get('area') == cache_selected_area]
                    
                    # 按方法筛选
                    if cache_selected_method != "全部":
                        filtered_index = [item for item in filtered_index if item.get('method') == cache_selected_method]
                    
                    # 按应用领域筛选
                    if cache_selected_domain != "全部":
                        filtered_index = [item for item in filtered_index if cache_selected_domain in item.get('application_domains', [])]
                    
                    # 按关键词筛选
                    if cache_keywords:
                        keywords_list = [kw.strip() for kw in cache_keywords.split(",") if kw.strip()]
                        if keywords_list:
                            # 过滤包含特定关键词的数据
                            filtered_index = [
                                item for item in filtered_index 
                                if any(kw in ', '.join(item.get('stage1_keywords', [])) for kw in keywords_list)
                            ]
                    
                    # 添加删除功能
                    if "to_delete_results" not in st.session_state:
                        st.session_state.to_delete_results = []
                    
                    # 显示筛选后的缓存数据
                    col1, col2 = st.columns([9, 1])
                    with col1:
                        st.write(f"共找到 {len(filtered_index)} 条缓存结果")
                    
                    with col2:
                        if st.button("批量删除", key="batch_delete_cached", disabled=len(st.session_state.to_delete_results) == 0):
                            delete_count = 0
                            for cache_key in st.session_state.to_delete_results:
                                if stage2_cache_manager.delete_result(cache_key):
                                    delete_count += 1
                            
                            if delete_count > 0:
                                st.success(f"成功删除{delete_count}条结果")
                                st.session_state.to_delete_results = []
                                time.sleep(1)
                                st.rerun()
                    
                    # 转换为DataFrame以便分页显示
                    cached_data = []
                    for item in filtered_index:
                        cached_item = {
                            'title': item.get('title', ''),
                            'year': item.get('year', ''),
                            'area': item.get('area', ''),
                            'method': item.get('method', ''),
                            'application_domains': ', '.join(item.get('application_domains', [])),
                            'cache_key': item.get('cache_key', '')
                        }
                        cached_data.append(cached_item)
                    
                    cached_df = pd.DataFrame(cached_data)
                    
                    # 显示数据表格，使用分页
                    show_columns = ['title', 'year', 'area', 'method', 'application_domains']
                    current_df, current_page, total_pages, start_idx, end_idx = paginate_dataframe(cached_df, "cached")
                    
                    st.write(f"筛选结果 ({len(cached_df)}条，显示第 {start_idx+1}-{end_idx} 条):")
                    
                    # 分页控件
                    render_pagination_controls("cached", total_pages, current_page)
                    
                    # 全选当前页
                    select_all = st.checkbox("全选当前页", key="select_all_cached")
                    
                    # 显示当前页数据
                    for i, row in current_df.iterrows():
                        col1, col2 = st.columns([1, 11])
                        
                        cache_key = row['cache_key']
                        
                        # 显示选择框
                        with col1:
                            is_selected = cache_key in st.session_state.to_delete_results
                            if st.checkbox("", value=is_selected or select_all, key=f"select_cached_{cache_key}"):
                                if cache_key not in st.session_state.to_delete_results:
                                    st.session_state.to_delete_results.append(cache_key)
                            else:
                                if cache_key in st.session_state.to_delete_results:
                                    st.session_state.to_delete_results.remove(cache_key)
                        
                        # 显示论文信息
                        with col2:
                            with st.expander(f"{row['title'][:50]}..."):
                                st.write(f"**年份**: {row['year']}")
                                st.write(f"**领域**: {row['area']}")
                                st.write(f"**方法**: {row['method']}")
                                st.write(f"**应用领域**: {row['application_domains']}")
                                
                                # 获取详细信息
                                detail_data = stage2_cache_manager.get_detail(cache_key)
                                if detail_data:
                                    domain_result = detail_data.get('domain_result', {})
                                    justification = domain_result.get('justification', '')
                                    if justification:
                                        st.write(f"**判断理由**: {justification}")
                                
                                # 单独删除按钮
                                if st.button("删除", key=f"delete_cached_{cache_key}"):
                                    if stage2_cache_manager.delete_result(cache_key):
                                        st.success("已删除")
                                        if cache_key in st.session_state.to_delete_results:
                                            st.session_state.to_delete_results.remove(cache_key)
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("删除失败")
                    
                    # 清空缓存的选项
                    if st.button("清空所有缓存", key="clear_cache_btn"):
                        if st.session_state.get("confirm_clear_cache", False):
                            stage2_cache_manager.clear_all_results()
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
                    processor = get_stage2_llm_processor()
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
                        
                        # 处理进度回调函数
                        def on_progress(processed, total, result):
                            progress = processed / total
                            item_progress.progress(progress)
                            item_progress.text(f"批处理进度: {processed}/{total} ({progress*100:.1f}%)")
                        
                        # 批量处理数据
                        with st.spinner("正在处理数据批次..."):
                            results = processor.process_papers(
                                batch_items,
                                on_progress,
                                st.session_state.max_concurrent
                            )
                            
                            # 保存处理结果
                            for i, result in enumerate(results):
                                paper = batch_items[i]
                                
                                # 将结果添加到已处理列表
                                processed_item = {
                                    'title': paper.get('title', ''),
                                    'abstract': paper.get('abstract', '')[:100] + '...',  # 摘要截断
                                    'year': paper.get('year', ''),
                                    'area': paper.get('area', ''),
                                    'method': paper.get('method', ''),
                                    'source': paper.get('source', ''),
                                    'cache_key': paper.get('cache_key', ''),
                                    'stage1_keywords': paper.get('relevant_keywords', []),
                                    'application_domains': result.get('application_domains', ['None']),
                                    'justification': result.get('justification', '')
                                }
                                
                                st.session_state.processed_items.append(processed_item)
                                
                                # 缓存结果
                                stage2_cache_manager.save_result(paper, result)
                        
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
                        import traceback
                        st.error(traceback.format_exc())
                        
                        # 发生错误时也将当前项从队列中移除，以防止无限循环
                        if st.session_state.processing_queue:
                            st.session_state.processing_queue.pop(0)
                        # 清除当前处理项
                        st.session_state.current_processing = None
                        # 重新加载页面
                        time.sleep(2)  # 稍微延迟以确保用户能看到错误
                        st.rerun()
        else:
            st.warning("请先加载第一阶段处理结果。")
    
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
                    st.markdown(f"**标题**: {current.get('title', '')[:50]}...")
                    st.markdown(f"**年份**: {current.get('year', 'N/A')}")
                    st.markdown(f"**领域**: {current.get('area', 'N/A')}")
                    st.markdown(f"**方法**: {current.get('method', 'N/A')}")
            else:
                st.info("尚未开始处理数据")
        
        # 显示领域分类统计信息
        if st.session_state.processed_items:
            st.subheader("领域分类统计")
            
            # 统计领域分类情况
            domain_counts = {
                "Derivatives Pricing": 0,
                "Financial Risk": 0,
                "Portfolio Management": 0,
                "None": 0,
                "Multiple": 0
            }
            
            for item in st.session_state.processed_items:
                domains = item.get('application_domains', ['None'])
                
                if len(domains) > 1 and 'None' not in domains:
                    domain_counts["Multiple"] += 1
                    # 同时计入各个具体领域
                    for domain in domains:
                        if domain in domain_counts:
                            domain_counts[domain] += 1
                elif len(domains) == 1:
                    domain = domains[0]
                    if domain in domain_counts:
                        domain_counts[domain] += 1
                    else:
                        domain_counts["None"] += 1
            
            # 显示领域分类饼图
            domain_data = pd.DataFrame({
                "domain": list(domain_counts.keys()),
                "count": list(domain_counts.values())
            }).sort_values("count", ascending=False)
            
            fig = px.pie(domain_data, values="count", names="domain", title="领域分类分布")
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示详细统计表格
            st.write("领域分类详情:")
            st.dataframe(domain_data)

# 结果可视化和分析页面
def render_result_analysis_page():
    st.header("📊 金融应用领域分析")
    
    # 获取缓存中的数据
    stage2_cache_manager = get_stage2_cache_manager()
    index_data = stage2_cache_manager.get_index()
    
    if not index_data:
        st.warning("缓存中没有处理结果，请先处理数据。")
        return
    
    # 获取领域统计信息
    domain_stats = stage2_cache_manager.get_domain_statistics()
    
    # 创建分析仪表盘
    dashboard_container = st.container()
    
    with dashboard_container:
        st.subheader("金融应用领域分布")
        
        # 显示总体领域分布
        domain_counts = domain_stats['domain_counts']
        domain_df = pd.DataFrame({
            'domain': list(domain_counts.keys()),
            'count': list(domain_counts.values())
        })
        domain_df = domain_df.sort_values('count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(domain_df, values='count', names='domain', title='金融应用领域分布')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(domain_df, x='domain', y='count', title='金融应用领域分布')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("按年份的金融应用领域分布")
        
        # 按年份显示领域分布趋势
        yearly_domain_counts = domain_stats['yearly_domain_counts']
        yearly_data = []
        
        for year, counts in yearly_domain_counts.items():
            for domain, count in counts.items():
                yearly_data.append({
                    'year': year,
                    'domain': domain,
                    'count': count
                })
        
        if yearly_data:
            yearly_df = pd.DataFrame(yearly_data)
            
            # 筛选有效的年份（去除unknown等）
            valid_years = [str(year) for year in range(1900, 2100)]
            valid_yearly_df = yearly_df[yearly_df['year'].isin(valid_years)]
            
            if not valid_yearly_df.empty:
                valid_yearly_df['year'] = pd.to_numeric(valid_yearly_df['year'])
                valid_yearly_df = valid_yearly_df.sort_values('year')
                
                fig = px.line(valid_yearly_df, x='year', y='count', color='domain', title='按年份的金融应用领域趋势')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("没有足够的按年份数据进行趋势分析")
        
        st.subheader("按研究方法的金融应用领域分布")
        
        # 按方法显示领域分布
        method_domain_counts = domain_stats['method_domain_counts']
        method_data = []
        
        # 过滤掉方法值为"unknown"的数据
        for method, counts in method_domain_counts.items():
            if method.lower() != "unknown" and method.strip():
                for domain, count in counts.items():
                    if count > 0:  # 只包含计数大于0的数据
                        method_data.append({
                            'method': method,
                            'domain': domain,
                            'count': count
                        })
        
        if method_data:
            method_df = pd.DataFrame(method_data)
            
            st.write("按研究方法和金融应用领域的分布热力图")
            
            # 创建交叉表
            heatmap_data = pd.pivot_table(
                method_df, 
                values='count', 
                index=['method'], 
                columns=['domain'],
                fill_value=0
            )
            
            # 绘制热力图
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="金融应用领域", y="研究方法", color="数量"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 堆叠条形图
            fig = px.bar(
                method_df, 
                x='method', 
                y='count', 
                color='domain',
                title='按研究方法的金融应用领域分布',
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有足够的按研究方法的数据进行分析")
    
    # 导出数据部分
    export_container = st.container()
    
    with export_container:
        st.subheader("导出处理结果")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # 导出所有处理结果
            if st.button("导出所有处理结果"):
                # 将索引数据转换为DataFrame
                export_df = stage2_cache_manager.export_to_dataframe()
                
                if not export_df.empty:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"financial_domain_analysis_{timestamp}.csv"
                    csv_data = export_df.to_csv(index=False)
                    
                    st.download_button(
                        label="下载CSV文件",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv"
                    )
                else:
                    st.warning("没有数据可导出")
        
        with export_col2:
            # 导出统计信息
            if st.button("导出统计信息"):
                # 创建统计数据
                stats_data = {
                    "总数据量": domain_stats['total'],
                    "领域分布": domain_stats['domain_counts'],
                    "年份分布": domain_stats['yearly_domain_counts'],
                    "研究方法分布": domain_stats['method_domain_counts']
                }
                
                # 转换为JSON字符串
                stats_json = json.dumps(stats_data, indent=2, ensure_ascii=False)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"financial_domain_stats_{timestamp}.json"
                
                st.download_button(
                    label="下载统计信息(JSON)",
                    data=stats_json,
                    file_name=filename,
                    mime="application/json"
                )
    
    # 详细查看结果
    result_detail_container = st.container()
    
    with result_detail_container:
        st.subheader("浏览详细结果")
        
        # 先将索引数据转换为DataFrame以方便筛选
        index_df = pd.DataFrame(index_data)
        
        # 创建筛选控件
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # 提取所有可能的领域、方法和应用领域
            if 'area' in index_df.columns:
                areas = ["全部"] + sorted(index_df['area'].unique().tolist())
                selected_area = st.selectbox("按领域筛选:", areas)
            
            if 'method' in index_df.columns:
                methods = ["全部"] + sorted(index_df['method'].unique().tolist())
                selected_method = st.selectbox("按方法筛选:", methods)
        
        with filter_col2:
            # 提取所有可能的应用领域
            application_domains = ["全部", "Derivatives Pricing", "Financial Risk", "Portfolio Management", "None"]
            selected_domain = st.selectbox("按金融应用领域筛选:", application_domains)
            
            # 按标题关键词筛选
            title_keyword = st.text_input("标题包含:", key="detail_title_keyword")
        
        # 筛选数据
        filtered_index = index_df.copy()
        
        if selected_area != "全部":
            filtered_index = filtered_index[filtered_index['area'] == selected_area]
        
        if selected_method != "全部":
            filtered_index = filtered_index[filtered_index['method'] == selected_method]
        
        if selected_domain != "全部":
            # 对于应用领域，需要检查列表中是否包含所选域
            if 'application_domains' in filtered_index.columns:
                filtered_index = filtered_index[filtered_index['application_domains'].apply(
                    lambda x: selected_domain in (x if isinstance(x, list) else [])
                )]
        
        if title_keyword:
            if 'title' in filtered_index.columns:
                filtered_index = filtered_index[filtered_index['title'].str.contains(title_keyword, case=False, na=False)]
        
        # 显示筛选后的结果
        if not filtered_index.empty:
            # 按应用领域分类显示
            domain_tabs = st.tabs(["全部", "Derivatives Pricing", "Financial Risk", "Portfolio Management", "None"])
            
            # 所有结果
            with domain_tabs[0]:
                # 分页显示所有结果
                render_data_table(filtered_index, 
                                 show_columns=['title', 'year', 'area', 'method', 'application_domains'],
                                 title="筛选结果",
                                 page_key="results_list")
                
                # 点击查看详情
                st.write("点击下方按钮查看随机样例:")
                if st.button("随机显示一篇论文详情", key="random_detail_all"):
                    random_idx = random.randint(0, len(filtered_index) - 1)
                    random_row = filtered_index.iloc[random_idx]
                    cache_key = random_row.get('cache_key', '')
                    
                    if cache_key:
                        detail = stage2_cache_manager.get_detail(cache_key)
                        if detail:
                            paper = detail.get('paper', {})
                            domain_result = detail.get('domain_result', {})
                            
                            st.subheader("论文详情")
                            st.markdown(f"**标题**: {paper.get('title', '')}")
                            st.markdown(f"**摘要**: {paper.get('abstract', '')}")
                            st.markdown(f"**年份**: {paper.get('year', '')}")
                            st.markdown(f"**领域**: {paper.get('area', '')}")
                            st.markdown(f"**方法**: {paper.get('method', '')}")
                            st.markdown(f"**第一阶段关键词**: {', '.join(paper.get('relevant_keywords', []))}")
                            
                            st.subheader("领域分类结果")
                            st.markdown(f"**应用领域**: {', '.join(domain_result.get('application_domains', ['None']))}")
                            st.markdown(f"**判断理由**: {domain_result.get('justification', '')}")
                        else:
                            st.warning("未找到该论文的详细信息")
            
            # 按领域分类显示
            domain_list = ["Derivatives Pricing", "Financial Risk", "Portfolio Management", "None"]
            
            for i, domain in enumerate(domain_list):
                with domain_tabs[i+1]:
                    # 提取此应用领域的数据
                    domain_df = filtered_index[filtered_index['application_domains'].apply(
                        lambda x: domain in (x if isinstance(x, list) else [])
                    )]
                    
                    if not domain_df.empty:
                        render_data_table(domain_df, 
                                         show_columns=['title', 'year', 'area', 'method'],
                                         title=f"{domain}领域论文",
                                         page_key=f"results_{domain.lower().replace(' ', '_')}")
                        
                        # 点击查看详情
                        if st.button(f"随机显示一篇{domain}领域论文详情", key=f"random_detail_{domain.lower().replace(' ', '_')}"):
                            random_idx = random.randint(0, len(domain_df) - 1)
                            random_row = domain_df.iloc[random_idx]
                            cache_key = random_row.get('cache_key', '')
                            
                            if cache_key:
                                detail = stage2_cache_manager.get_detail(cache_key)
                                if detail:
                                    paper = detail.get('paper', {})
                                    domain_result = detail.get('domain_result', {})
                                    
                                    st.subheader("论文详情")
                                    st.markdown(f"**标题**: {paper.get('title', '')}")
                                    st.markdown(f"**摘要**: {paper.get('abstract', '')}")
                                    st.markdown(f"**年份**: {paper.get('year', '')}")
                                    st.markdown(f"**领域**: {paper.get('area', '')}")
                                    st.markdown(f"**方法**: {paper.get('method', '')}")
                                    st.markdown(f"**第一阶段关键词**: {', '.join(paper.get('relevant_keywords', []))}")
                                    
                                    st.subheader("领域分类结果")
                                    st.markdown(f"**应用领域**: {', '.join(domain_result.get('application_domains', ['None']))}")
                                    st.markdown(f"**判断理由**: {domain_result.get('justification', '')}")
                                else:
                                    st.warning("未找到该论文的详细信息")
                    else:
                        st.info(f"没有{domain}领域的论文")
        else:
            st.info("没有符合筛选条件的结果") 

# 应用程序主函数
def main():
    # 初始化应用程序状态
    init_session_state()
    
    # 加载默认提示词
    if not st.session_state.system_prompt or not st.session_state.user_prompt_template:
        load_default_prompts()
    
    # 创建侧边栏
    st.sidebar.title("论文金融应用领域分类")
    st.sidebar.subheader("第二阶段：应用领域分类")
    
    # 菜单选项
    menu_options = ["加载数据", "提示词管理", "LLM处理", "结果分析"]
    
    # 创建侧边栏菜单
    selected_menu = st.sidebar.radio("导航", menu_options)
    
    # 更新当前选项卡
    st.session_state.current_tab = selected_menu
    
    
    # 根据选择的菜单显示相应的页面
    if selected_menu == "加载数据":
        render_data_loading_page()
    elif selected_menu == "提示词管理":
        render_prompts_management_page()
    elif selected_menu == "LLM处理":
        render_llm_processing_page()
    elif selected_menu == "结果分析":
        render_result_analysis_page()

# 启动应用程序
if __name__ == "__main__":
    main() 