import streamlit as st
import os
import sys
from datetime import datetime
import pandas as pd
import json
from collections import defaultdict
import hashlib
import pickle
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from utils.data_loader import DataLoader
from utils.cache_manager import CacheManager
from utils.llm_processor import LLMProcessor, get_keywords

# 添加第二阶段缓存管理器
from utils.stage2_cache_manager import Stage2CacheManager

# ========== Session初始化 ==========
def init_session_state():
    # 参考原有初始化，简化版
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = None
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = []
    if "keyword_subsets" not in st.session_state:
        st.session_state.keyword_subsets = {"ML": [], "DL": [], "LLM": []}
    if "results_cache" not in st.session_state:
        st.session_state.results_cache = None
    if "stage2_results_cache" not in st.session_state:
        st.session_state.stage2_results_cache = None
    if "last_session_time" not in st.session_state:
        st.session_state.last_session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "show_detail_view" not in st.session_state:
        st.session_state.show_detail_view = False
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None
    if "filter_no_match" not in st.session_state:
        st.session_state.filter_no_match = False
    if "deduplication" not in st.session_state:
        st.session_state.deduplication = True

# ========== 数据加载页面 ==========
def render_data_loading_page():
    st.header("📊 数据加载")
    cache_manager = CacheManager()
    data_loader = DataLoader()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("选择数据来源和文件")
        available_files = data_loader.get_available_data_files()
        # 检查缓存中的上一次加载的数据
        cached_data, metadata = None, None
        if hasattr(cache_manager, 'load_last_data'):
            try:
                cached_data, metadata = cache_manager.load_last_data()
            except Exception:
                cached_data, metadata = None, None
        # 检查缓存中的上一次加载的数据
        cached_data, metadata = cache_manager.load_last_data()
        if cached_data is not None and metadata is not None:
            st.info(f"发现上次加载的数据: {metadata['rows']}条记录，来源: {metadata['source']}，加载时间: {metadata['timestamp']}")
            if st.button("恢复上次加载的数据"):
                st.session_state.loaded_data = cached_data
                st.session_state.last_loaded_source = metadata['source']
                st.session_state.last_loaded_files = metadata['file_paths']
                st.success(f"已恢复上次加载的数据，共{len(cached_data)}条记录")
                st.rerun()
        # ====== 支持多数据源同时加载 ======
        data_sources = st.multiselect("选择数据来源:", ["CNKI", "WOS"], default=["CNKI"])
        file_type = st.radio("选择文件类型:", ["xls", "csv"])
        selected_file_paths = []
        for data_source in data_sources:
            available_file_types = available_files[data_source]
            if available_file_types[file_type]:
                selected_files = st.multiselect(
                    f"选择要加载的{data_source}文件:",
                    [os.path.basename(f) for f in available_file_types[file_type]],
                    key=f"{data_source}_{file_type}_selection"
                )
                selected_file_paths += [
                    os.path.join(os.path.dirname(f), s)
                    for f in available_file_types[file_type]
                    for s in selected_files
                    if os.path.basename(f) == s
                ]
        if selected_file_paths and st.button("加载选中的文件"):
            try:
                with st.spinner("正在加载数据..."):
                    dfs = []
                    for data_source in data_sources:
                        ds_files = [f for f in selected_file_paths if data_source in f]
                        if ds_files:
                            df = data_loader.load_multiple_files(ds_files, data_source)
                            if not df.empty:
                                dfs.append(df)
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                        st.session_state.loaded_data = df
                        st.session_state.last_loaded_source = ','.join(data_sources)
                        st.session_state.last_loaded_files = selected_file_paths
                        # 保存加载数据到缓存
                        cache_manager.save_loaded_data(df, ','.join(data_sources), selected_file_paths)
                        st.success(f"成功加载 {len(df)} 条数据！")
                        import time; time.sleep(1)
                        st.rerun()
                    else:
                        st.error("没有成功加载任何数据。")
            except Exception as e:
                st.error(f"加载数据时出错: {str(e)}")
        else:
            st.info(f"请至少选择一个数据源的 {file_type} 文件。")
    with col2:
        st.subheader("已加载的数据预览")
        if st.session_state.loaded_data is not None:
            df = st.session_state.loaded_data
            st.write(f"数据形状: {df.shape}")
            # ====== 新增：方法间重复文献统计表格及独特文献数 ======
            if all(col in df.columns for col in ['title', 'method']):
                ml_titles = set(df[df['method'] == 'machine learning']['title'])
                dl_titles = set(df[df['method'] == 'deep learning']['title'])
                llm_titles = set(df[df['method'] == 'LLMs']['title'])
                all_titles = set(df['title'])
                # 方法交叉
                ml_dl = ml_titles & dl_titles
                dl_llm = dl_titles & llm_titles
                ml_llm = ml_titles & llm_titles
                ml_dl_llm = ml_titles & dl_titles & llm_titles
                # 独特文献数（去除所有方法间重复）
                unique_titles = ml_titles | dl_titles | llm_titles
                # 统计表格
                cross_table = pd.DataFrame({
                    '(ML∩DL)': [len(ml_dl)],
                    '(DL∩LLM)': [len(dl_llm)],
                    '(ML∩LLM)': [len(ml_llm)],
                    '(ML∩DL∩LLM)': [len(ml_dl_llm)]
                }, index=['方法交叉文献数'])
                st.markdown("**三方法交叉重复文献统计**")
                st.dataframe(cross_table)
                # 总文献数、独特文献数
                st.markdown(f"**所有文献总数：{len(df)}**")
                st.markdown(f"**去除方法间重复后的独特文献数：{len(unique_titles)}**")
            # ====== 原有数据分布与预览 ======
            stats_tab1, stats_tab2 = st.tabs(["数据分布", "数据预览"])
            with stats_tab1:
                if 'area' in df.columns and 'method' in df.columns:
                    area_counts = df['area'].value_counts().reset_index()
                    area_counts.columns = ['area', 'count']
                    method_counts = df['method'].value_counts().reset_index()
                    method_counts.columns = ['method', 'count']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("按领域分布")
                        import plotly.express as px
                        fig = px.pie(area_counts, values='count', names='area', title='按领域分布')
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.subheader("按方法分布")
                        fig = px.pie(method_counts, values='count', names='method', title='按方法分布')
                        st.plotly_chart(fig, use_container_width=True)
                if 'year' in df.columns:
                    year_counts = df['year'].value_counts().reset_index()
                    year_counts.columns = ['year', 'count']
                    year_counts = year_counts.sort_values('year')
                    st.subheader("按年份分布")
                    fig = px.bar(year_counts, x='year', y='count', title='按年份分布')
                    st.plotly_chart(fig, use_container_width=True)
            with stats_tab2:
                st.dataframe(df.head(10))
                if st.button("随机展示一条数据"):
                    if len(df) > 0:
                        import random
                        random_idx = random.randint(0, len(df) - 1)
                        random_row = df.iloc[random_idx]
                        st.subheader("随机数据样例")
                        st.markdown(f"**标题**: {random_row['title']}")
                        st.markdown(f"**摘要**: {random_row['abstract']}")
                        st.markdown(f"**年份**: {random_row['year']}")
                        st.markdown(f"**领域**: {random_row['area']}")
                        st.markdown(f"**方法**: {random_row['method']}")
                        st.markdown(f"**处理阶段**: {'第二阶段' if random_row.get('stage') == 2 else '第一阶段'}")
                    else:
                        st.error("数据集为空，无法展示随机数据。")
        else:
            st.info("请先加载数据。")

# ========== 关键词管理页面 ==========
def render_keywords_management_page():
    st.header("🔑 关键词管理")
    cache_manager = CacheManager()
    # 如果还没有选择关键词，尝试从缓存恢复上次选择的关键词
    if not st.session_state.selected_keywords:
        if hasattr(cache_manager, 'load_last_keywords'):
            last_keywords = cache_manager.load_last_keywords()
            if last_keywords:
                st.info(f"发现上次选择的{len(last_keywords)}个关键词")
                if st.button("恢复上次选择的关键词"):
                    st.session_state.selected_keywords = last_keywords
                    st.success(f"已恢复上次选择的{len(last_keywords)}个关键词")
                    st.rerun()
    keywords_dict = get_keywords()
    if not keywords_dict:
        st.error("无法加载关键词，请检查keywords.py文件是否存在。")
        return
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("关键词选择")
        categories = list(keywords_dict.keys())
        category = st.selectbox("选择关键词类别:", categories)
        if category in keywords_dict:
            keywords = keywords_dict[category]
            st.write(f"共 {len(keywords)} 个关键词")
            st.subheader("批量选择")
            batch_col1, batch_col2 = st.columns(2)
            with batch_col1:
                select_all = st.checkbox("全选", key=f"select_all_{category}")
            with batch_col2:
                if select_all:
                    st.session_state.to_select_keywords = keywords.copy()
                if st.button("批量添加", key=f"batch_add_{category}"):
                    added_count = 0
                    for kw in st.session_state.to_select_keywords:
                        if kw not in st.session_state.selected_keywords:
                            st.session_state.selected_keywords.append(kw)
                            added_count += 1
                    st.session_state.to_select_keywords = []
                    if hasattr(cache_manager, 'save_current_keywords'):
                        cache_manager.save_current_keywords(st.session_state.selected_keywords)
                    if added_count > 0:
                        st.success(f"已添加{added_count}个关键词")
                        st.rerun()
            st.subheader("关键词列表")
            keyword_container = st.container()
            keywords_count = len(keywords)
            num_columns = 4
            rows_per_column = (keywords_count + num_columns - 1) // num_columns
            keyword_controls = []
            for keyword_idx, keyword in enumerate(keywords):
                col_idx = keyword_idx // rows_per_column
                row_idx = keyword_idx % rows_per_column
                is_selected = keyword in st.session_state.selected_keywords
                is_in_batch = keyword in st.session_state.to_select_keywords
                keyword_controls.append((row_idx, col_idx, keyword, is_selected, is_in_batch))
            keyword_controls.sort()
            cols = st.columns(4)
            for row_idx in range(rows_per_column):
                for col_idx in range(num_columns):
                    current_controls = [
                        control for control in keyword_controls
                        if control[0] == row_idx and control[1] == col_idx
                    ]
                    if current_controls:
                        _, _, keyword, is_selected, is_in_batch = current_controls[0]
                        with cols[col_idx]:
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
                                if hasattr(cache_manager, 'save_current_keywords'):
                                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                                st.rerun()
                            if batch_select and keyword not in st.session_state.to_select_keywords:
                                st.session_state.to_select_keywords.append(keyword)
                            elif not batch_select and keyword in st.session_state.to_select_keywords:
                                st.session_state.to_select_keywords.remove(keyword)
        else:
            st.error(f"找不到类别'{category}'的关键词。")
        # ====== 关键词子集一键保存/加载 ======
        st.markdown("---")
        st.subheader("关键词子集一键保存/加载")
        ml_keywords = keywords_dict.get("machine learning", [])
        dl_keywords = keywords_dict.get("deep learning", [])
        llm_keywords = keywords_dict.get("LLMs", [])
        col_ml, col_dl, col_llm = st.columns(3)
        with col_ml:
            if st.button("保存ML子集"):
                st.session_state.keyword_lists["ML子集"] = ml_keywords
                cache_manager.save_keyword_list("ML子集", ml_keywords)
                st.success("已保存ML子集")
            if st.button("加载ML子集"):
                st.session_state.selected_keywords = ml_keywords.copy()
                cache_manager.save_current_keywords(ml_keywords)
                st.success("已加载ML子集")
                st.rerun()
        with col_dl:
            if st.button("保存DL子集"):
                st.session_state.keyword_lists["DL子集"] = dl_keywords
                cache_manager.save_keyword_list("DL子集", dl_keywords)
                st.success("已保存DL子集")
            if st.button("加载DL子集"):
                st.session_state.selected_keywords = dl_keywords.copy()
                cache_manager.save_current_keywords(dl_keywords)
                st.success("已加载DL子集")
                st.rerun()
        with col_llm:
            if st.button("保存LLM子集"):
                st.session_state.keyword_lists["LLM子集"] = llm_keywords
                cache_manager.save_keyword_list("LLM子集", llm_keywords)
                st.success("已保存LLM子集")
            if st.button("加载LLM子集"):
                st.session_state.selected_keywords = llm_keywords.copy()
                cache_manager.save_current_keywords(llm_keywords)
                st.success("已加载LLM子集")
                st.rerun()
    with col2:
        st.subheader("已选关键词")
        num_selected = len(st.session_state.selected_keywords)
        st.write(f"已选择 {num_selected} 个关键词")
        st.subheader("保存关键词列表")
        save_container = st.container()
        list_name = save_container.text_input("关键词列表名称:", placeholder="输入名称...")
        save_disabled = not list_name or num_selected == 0
        if save_container.button("保存关键词列表", disabled=save_disabled):
            try:
                st.session_state.keyword_lists[list_name] = st.session_state.selected_keywords.copy()
                if hasattr(cache_manager, 'save_keyword_list'):
                    cache_manager.save_keyword_list(list_name, st.session_state.selected_keywords)
                st.success(f"已保存关键词列表：{list_name}")
                if hasattr(cache_manager, 'save_current_keywords'):
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
            except Exception as e:
                st.error(f"保存关键词列表时出错: {str(e)}")
        saved_lists_from_cache = cache_manager.get_all_keyword_lists()
        for name, keywords in saved_lists_from_cache.items():
            if name not in st.session_state.keyword_lists:
                st.session_state.keyword_lists[name] = keywords
        if st.session_state.keyword_lists:
            saved_lists = list(st.session_state.keyword_lists.keys())
            selected_list = st.selectbox("加载已保存的列表:", [""] + saved_lists)
            if selected_list and st.button("加载列表"):
                try:
                    st.session_state.selected_keywords = st.session_state.keyword_lists[selected_list].copy()
                    if hasattr(cache_manager, 'save_current_keywords'):
                        cache_manager.save_current_keywords(st.session_state.selected_keywords)
                    st.success(f"已加载关键词列表：{selected_list}")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载关键词列表时出错: {str(e)}")
        st.subheader("关键词管理")
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
                st.session_state.to_delete_keywords = []
                if hasattr(cache_manager, 'save_current_keywords'):
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                if removed_count > 0:
                    st.success(f"已删除{removed_count}个关键词")
                    st.rerun()
            except Exception as e:
                st.error(f"删除关键词时出错: {str(e)}")
        st.subheader("已选关键词列表")
        if st.session_state.selected_keywords:
            selected_keywords_container = st.container()
            num_selected = len(st.session_state.selected_keywords)
            num_columns = 4
            rows_per_column = (num_selected + num_columns - 1) // num_columns
            keyword_controls = []
            for keyword_idx, keyword in enumerate(st.session_state.selected_keywords):
                col_idx = keyword_idx // rows_per_column
                row_idx = keyword_idx % rows_per_column
                keyword_controls.append((row_idx, col_idx, keyword))
            keyword_controls.sort()
            selected_cols = st.columns(4)
            for row_idx, col_idx, keyword in keyword_controls:
                with selected_cols[col_idx]:
                    is_in_delete_batch = keyword in st.session_state.to_delete_keywords
                    delete_select = st.checkbox(
                        keyword,
                        value=delete_all or is_in_delete_batch,
                        key=f"del_{keyword}"
                    )
                    if st.button("删除", key=f"remove_{keyword}"):
                        st.session_state.selected_keywords.remove(keyword)
                        if hasattr(cache_manager, 'save_current_keywords'):
                            cache_manager.save_current_keywords(st.session_state.selected_keywords)
                        st.rerun()
                    if delete_select and keyword not in st.session_state.to_delete_keywords:
                        st.session_state.to_delete_keywords.append(keyword)
                    elif not delete_select and keyword in st.session_state.to_delete_keywords:
                        st.session_state.to_delete_keywords.remove(keyword)
        else:
            st.info("请先从左侧选择关键词。")
        st.subheader("添加自定义关键词")
        custom_container = st.container()
        new_keyword = custom_container.text_input("输入关键词:")
        add_disabled = not new_keyword or new_keyword in st.session_state.selected_keywords
        if custom_container.button("添加自定义关键词", disabled=add_disabled, key="add_custom"):
            if new_keyword and new_keyword not in st.session_state.selected_keywords:
                st.session_state.selected_keywords.append(new_keyword)
                if hasattr(cache_manager, 'save_current_keywords'):
                    cache_manager.save_current_keywords(st.session_state.selected_keywords)
                st.success(f"已添加关键词：{new_keyword}")
                st.rerun()

# ========== 结果查看页面 ==========
def render_results_view_page():
    st.header("📋 结果查看")
    from utils.cache_manager import CacheManager
    import pandas as pd
    import random
    import time
    
    # 初始化第一阶段和第二阶段缓存管理器
    cache_manager = CacheManager()
    stage2_cache_manager = Stage2CacheManager()
    
    # 加载第一阶段结果
    if "results_cache" not in st.session_state or st.session_state.results_cache is None:
        st.session_state.results_cache = cache_manager.get_results_by_filter({})
    
    # 加载第二阶段结果
    if "stage2_results_cache" not in st.session_state or st.session_state.stage2_results_cache is None:
        try:
            # 获取第二阶段索引数据
            stage2_index = stage2_cache_manager.get_index()
            
            # 将索引数据转换为与第一阶段缓存格式兼容的格式
            stage2_results = []
            
            for item in stage2_index:
                try:
                    cache_key = item.get('cache_key', '')
                    if not cache_key:
                        continue
                        
                    # 获取详细信息
                    detail = stage2_cache_manager.get_detail(cache_key)
                    if detail:
                        # 转换为第一阶段兼容格式
                        metadata = {
                            'title': item.get('title', ''),
                            'abstract': detail.get('paper', {}).get('abstract', ''),
                            'year': item.get('year', ''),
                            'area': item.get('area', ''),
                            'method': item.get('method', ''),
                            'source': item.get('source', ''),
                            'cache_key': cache_key,
                            'timestamp': item.get('timestamp', ''),
                            'stage': 2,  # 标记为第二阶段数据
                            'stage1_cache_key': item.get('stage1_cache_key', '') or detail.get('paper', {}).get('cache_key', '')  # 添加关联到第一阶段的缓存键
                        }
                        
                        # 获取第一阶段的处理结果，以便获取关键词解释
                        stage1_explanations = {}
                        if 'paper' in detail and 'cache_key' in detail['paper']:
                            stage1_cache_key = detail['paper'].get('cache_key', '')
                            if stage1_cache_key:
                                try:
                                    # 从第一阶段缓存中获取解释内容
                                    stage1_result = cache_manager.get_cached_result(stage1_cache_key)
                                    if stage1_result and 'explanations' in stage1_result:
                                        stage1_explanations = stage1_result.get('explanations', {})
                                except Exception:
                                    pass  # 忽略获取第一阶段解释的错误
                        
                        result = {
                            'relevant_keywords': item.get('stage1_keywords', []),
                            'application_domains': item.get('application_domains', []),
                            'justification': detail.get('domain_result', {}).get('justification', ''),
                            'explanations': stage1_explanations,  # 添加从第一阶段获取的关键词解释
                            'success': True
                        }
                        
                        stage2_results.append({'metadata': metadata, 'result': result})
                except Exception:
                    continue  # 忽略单个项目的处理错误
            
            st.session_state.stage2_results_cache = stage2_results
            
        except Exception:
            st.session_state.stage2_results_cache = []
    
    # 合并第一阶段和第二阶段结果
    all_results = []
    
    # 添加第一阶段结果
    if st.session_state.results_cache:
        # 标记第一阶段数据
        for item in st.session_state.results_cache:
            if 'metadata' in item:
                item['metadata']['stage'] = 1
        all_results.extend(st.session_state.results_cache)
    
    # 添加第二阶段结果
    if st.session_state.stage2_results_cache:
        all_results.extend(st.session_state.stage2_results_cache)
    
    # 提取元数据列表用于筛选
    all_items = [r['metadata'] for r in all_results]
    
    if not all_items:
        st.info("暂无处理结果。请先在LLM处理页面处理数据。")
        return

    # ====== 新增：搜索框 ======
    search_title = st.text_input("按标题关键字搜索：", "")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("筛选条件")
        areas = sorted(set(item.get('area') for item in all_items if item.get('area')))
        methods = sorted(set(item.get('method') for item in all_items if item.get('method')))
        sources = sorted(set(item.get('source') for item in all_items if item.get('source')))
        selected_area = st.selectbox("领域:", ["全部"] + areas)
        selected_method = st.selectbox("方法:", ["全部"] + methods)
        selected_source = st.selectbox("数据源:", ["全部"] + sources)
        
        # 添加处理阶段筛选
        stages = ["全部", "第一阶段", "第二阶段"]
        selected_stage = st.selectbox("处理阶段:", stages)
        
        # 添加第二阶段领域分类结果筛选
        # 收集所有可能的应用领域值
        all_domains = []
        for result_item in all_results:
            if 'result' in result_item and 'application_domains' in result_item['result']:
                domains = result_item['result'].get('application_domains', [])
                if domains and domains != ["None"]:
                    all_domains.extend(domains)
        # 去重并排序
        unique_domains = sorted(set(all_domains))
        # 添加"无应用领域"选项
        domain_options = ["全部", "无应用领域"] + unique_domains
        if unique_domains:
            selected_domain = st.selectbox("应用领域:", domain_options)
        
        # ====== 关键词匹配结果筛选 ======
        match_filter = st.radio("关键词匹配结果:", ["全部", "有匹配", "无匹配"])
        annotations = cache_manager.get_all_annotations()
        has_annotations = bool(annotations)
        if has_annotations:
            annotation_filter = st.radio("标注状态:", ["全部", "已标注", "未标注"])
            
        # 构建筛选条件
        filter_criteria = {}
        if selected_area != "全部":
            filter_criteria['area'] = selected_area
        if selected_method != "全部":
            filter_criteria['method'] = selected_method
        if selected_source != "全部":
            filter_criteria['source'] = selected_source
            
        # 处理阶段筛选
        if selected_stage == "第一阶段":
            filtered_results = [r for r in all_results if r['metadata'].get('stage') == 1]
        elif selected_stage == "第二阶段":
            filtered_results = [r for r in all_results if r['metadata'].get('stage') == 2]
        else:
            filtered_results = all_results.copy()
            
        # 按属性筛选
        filtered_results = [r for r in filtered_results if all(
            (k not in r['metadata'] or r['metadata'][k] == v) for k, v in filter_criteria.items()
        )]
        
        # 添加领域筛选
        if 'selected_domain' in locals() and selected_domain != "全部":
            if selected_domain == "无应用领域":
                # 筛选没有应用领域的文献
                filtered_results = [r for r in filtered_results if 
                                   'result' not in r or 
                                   'application_domains' not in r['result'] or 
                                   not r['result']['application_domains'] or 
                                   r['result']['application_domains'] == ["None"]]
            else:
                # 筛选包含特定应用领域的文献
                filtered_results = [r for r in filtered_results if 
                                   'result' in r and 
                                   'application_domains' in r['result'] and 
                                   selected_domain in r['result']['application_domains']]
        
        # ====== 新增：按title关键字筛选 ======
        if search_title.strip():
            filtered_results = [r for r in filtered_results if search_title.strip().lower() in r['metadata'].get('title', '').lower()]
        # 关键词匹配结果筛选
        if match_filter == "有匹配":
            filtered_results = [r for r in filtered_results if r['result'].get('relevant_keywords', [])]
        elif match_filter == "无匹配":
            filtered_results = [r for r in filtered_results if not r['result'].get('relevant_keywords', [])]
        # 标注筛选
        if has_annotations and annotation_filter != "全部":
            if annotation_filter == "已标注":
                filtered_results = [r for r in filtered_results if r['metadata']['cache_key'] in annotations]
            else:
                filtered_results = [r for r in filtered_results if r['metadata']['cache_key'] not in annotations]
        st.write(f"共找到 {len(filtered_results)} 条结果")
        view_col1, view_col2 = st.columns(2)
        with view_col1:
            if st.button("随机查看一条"):
                if filtered_results:
                    random_idx = random.randint(0, len(filtered_results) - 1)
                    st.session_state.selected_result = filtered_results[random_idx]
                    st.session_state.show_detail_view = True
                    st.rerun()
        with view_col2:
            if st.button("查看列表"):
                st.session_state.show_detail_view = False
                st.session_state.selected_result = None
                st.rerun()
        if "to_delete_results" not in st.session_state:
            st.session_state.to_delete_results = []
        st.subheader("批量操作")
        if len(st.session_state.to_delete_results) > 0:
            st.write(f"已选择 {len(st.session_state.to_delete_results)} 条结果待删除")
        if st.button("删除选中项", disabled=len(st.session_state.to_delete_results) == 0):
            try:
                delete_count = 0
                stage1_deleted = False
                stage2_deleted = False
                
                # 查找每个要删除的缓存键属于哪个阶段
                for cache_key in st.session_state.to_delete_results:
                    # 查找对应的数据项
                    data_item = next((item for item in all_results if item['metadata'].get('cache_key') == cache_key), None)
                    
                    if data_item:
                        # 根据阶段选择对应的缓存管理器
                        if data_item['metadata'].get('stage') == 2:
                            if stage2_cache_manager.delete_result(cache_key):
                                delete_count += 1
                                stage2_deleted = True
                        else:
                            if cache_manager.delete_result(cache_key):
                                delete_count += 1
                                stage1_deleted = True
                
                if delete_count > 0:
                    st.success(f"成功删除{delete_count}条结果")
                    st.session_state.to_delete_results = []
                    # 根据删除的数据阶段刷新对应的缓存
                    if stage1_deleted:
                        st.session_state.results_cache = None
                    if stage2_deleted:
                        st.session_state.stage2_results_cache = None
                    
                    time.sleep(1)
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
        view_mode = st.radio("查看模式:", ["详情视图", "列表视图"], horizontal=True,
                            index=0 if st.session_state.show_detail_view else 1)
        if st.session_state.show_detail_view != (view_mode == "详情视图"):
            st.session_state.show_detail_view = (view_mode == "详情视图")
            if view_mode == "列表视图":
                st.session_state.selected_result = None
            st.rerun()
        if st.session_state.show_detail_view and st.session_state.selected_result:
            st.subheader("结果详情")
            if st.button("返回列表"):
                st.session_state.show_detail_view = False
                st.session_state.selected_result = None
                st.rerun()
            else:
                selected = st.session_state.selected_result
                metadata = selected['metadata']
                result = selected['result']

                # 基本信息展示
                st.markdown("### 论文基本信息")
                st.markdown(f"**标题**: {metadata.get('title', '')}")
                with st.expander("显示完整摘要"):
                    st.markdown(f"{metadata.get('abstract', '')}")
                
                # 两列显示年份、领域、方法等信息
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**年份**: {metadata.get('year', '')}")
                    st.markdown(f"**读取时领域**: {metadata.get('area', '')}")
                    st.markdown(f"**读取时方法**: {metadata.get('method', '')}")
                    st.markdown(f"**数据来源**: {metadata.get('source', '')}")
                    st.markdown(f"**处理阶段**: {'第二阶段' if metadata.get('stage') == 2 else '第一阶段'}")
                
                with col2:
                    # 根据关键词匹配确定领域和方法
                    relevant_keywords = result.get('relevant_keywords', [])
                    if relevant_keywords:
                        # 如果有导入keywords.py，则使用它来分类关键词
                        from utils.llm_processor import get_keywords
                        keywords_dict = get_keywords()
                        
                        # 默认类别
                        match_method = "未匹配"
                        
                        # 检查关键词属于哪个方法类别
                        ml_keywords = set(keywords_dict.get("machine learning", []))
                        dl_keywords = set(keywords_dict.get("deep learning", []))
                        llm_keywords = set(keywords_dict.get("LLMs", []))
                        
                        # 与关键词列表求交集，确定方法分类
                        relevant_kw_set = set(relevant_keywords)
                        ml_match = relevant_kw_set.intersection(ml_keywords)
                        dl_match = relevant_kw_set.intersection(dl_keywords)
                        llm_match = relevant_kw_set.intersection(llm_keywords)
                        
                        if llm_match:
                            match_method = "LLMs"
                        elif dl_match:
                            match_method = "deep learning"
                        elif ml_match:
                            match_method = "machine learning"
                        
                        st.markdown(f"**匹配后方法**: {match_method}")
                    else:
                        st.markdown("**匹配后方法**: 无匹配关键词")
                    
                    # 显示匹配后的应用领域
                    application_domains = result.get('application_domains', [])
                    if application_domains and application_domains != ["None"]:
                        st.markdown(f"**匹配后领域**: {', '.join(application_domains)}")
                    else:
                        st.markdown("**匹配后领域**: 未应用于特定金融领域")
                    
                    # 时间信息
                    st.markdown(f"**处理时间**: {metadata.get('timestamp', '').split('T')[0] if 'timestamp' in metadata else '未知'}")
                
                # 匹配的关键词和解释
                st.markdown("### 关键词匹配结果")
                relevant_keywords = result.get('relevant_keywords', [])
                
                # 确保relevant_keywords是列表，处理可能的JSON字符串情况
                if isinstance(relevant_keywords, str):
                    try:
                        import json
                        relevant_keywords = json.loads(relevant_keywords)
                    except:
                        relevant_keywords = []
                
                # 确保explanations正确获取，处理可能的JSON字符串情况
                explanations = result.get('explanations', {})
                # 如果explanations是JSON字符串，则解析它
                if isinstance(explanations, str):
                    try:
                        import json
                        explanations = json.loads(explanations)
                    except:
                        explanations = {}
                
                # 处理第二阶段数据的特殊情况
                if metadata.get('stage') == 2 and (not explanations or len(explanations) == 0):
                    # 尝试从第一阶段缓存获取原始的explanations
                    stage1_cache_key = metadata.get('stage1_cache_key', '')
                    if stage1_cache_key:
                        stage1_result = cache_manager.get_cached_result(stage1_cache_key)
                        if stage1_result:
                            if isinstance(stage1_result, dict) and 'explanations' in stage1_result:
                                explanations = stage1_result.get('explanations', {})
                            elif isinstance(stage1_result, dict) and 'result' in stage1_result and 'explanations' in stage1_result['result']:
                                explanations = stage1_result['result'].get('explanations', {})
                
                if not relevant_keywords:
                    reason = explanations.get('reason', "未提供原因")
                    st.warning(f"**无匹配关键词**: {reason}")
                else:
                    # 创建表格以展示关键词和解释
                    keyword_data = []
                    for keyword in relevant_keywords:
                        keyword_type = "未知"
                        if keyword in ml_keywords:
                            keyword_type = "机器学习"
                        elif keyword in dl_keywords:
                            keyword_type = "深度学习"
                        elif keyword in llm_keywords:
                            keyword_type = "大语言模型"
                        
                        explanation = explanations.get(keyword, "")
                        keyword_data.append({
                            "关键词": keyword,
                            "类型": keyword_type,
                            "解释": explanation
                        })
                    
                    # 显示关键词表格
                    if keyword_data:
                        st.dataframe(pd.DataFrame(keyword_data), use_container_width=True)
                    
                    # 分类显示关键词
                    matches_by_type = {
                        "机器学习": [k for k in relevant_keywords if k in ml_keywords],
                        "深度学习": [k for k in relevant_keywords if k in dl_keywords],
                        "大语言模型": [k for k in relevant_keywords if k in llm_keywords],
                        "其他": [k for k in relevant_keywords if k not in ml_keywords and k not in dl_keywords and k not in llm_keywords]
                    }
                    
                    # 显示分类结果
                    for category, keywords in matches_by_type.items():
                        if keywords:
                            with st.expander(f"{category} ({len(keywords)}个关键词)"):
                                for kw in keywords:
                                    exp = explanations.get(kw, "无解释")
                                    st.markdown(f"**{kw}**: {exp}")
                
                # 显示论文领域分类信息
                if "application_domains" in result:
                    st.markdown("### 论文领域分类")
                    domains = result.get("application_domains", [])
                    
                    # 确保domains是列表，处理可能的JSON字符串情况
                    if isinstance(domains, str):
                        try:
                            import json
                            domains = json.loads(domains)
                        except:
                            domains = []
                    
                    justification = result.get("justification", "无解释")
                    
                    if domains and domains != ["None"]:
                        st.success(f"**应用领域**: {', '.join(domains)}")
                    else:
                        st.warning("论文未应用于特定金融领域")
                    
                    # 创建领域分类结果表格
                    domain_data = [{
                        "应用领域": ', '.join(domains) if domains and domains != ["None"] else "未应用于特定金融领域",
                        "判断理由": justification
                    }]
                    
                    # 显示领域分类表格
                    st.dataframe(pd.DataFrame(domain_data), use_container_width=True)
                    
                    # 详细显示判断理由
                    with st.expander("查看详细判断理由"):
                        st.markdown(f"{justification}")
                
                # 人工标注部分
                st.markdown("### 人工标注")
                cache_key = metadata.get('cache_key')
                annotation = cache_manager.get_annotation(cache_key) if cache_key else None
                is_correct = st.radio(
                    "LLM判断是否正确:",
                    ["正确", "部分正确", "不正确"],
                    index=0 if not annotation else (0 if annotation.get('is_correct') == "正确" else 1 if annotation.get('is_correct') == "部分正确" else 2)
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
                            if 'annotation_results' not in st.session_state:
                                st.session_state.annotation_results = {}
                            st.session_state.annotation_results[cache_key] = annotation_data
                        else:
                            st.error("保存标注时出错。")
                    except Exception as e:
                        st.error(f"保存标注时出错: {str(e)}")
        else:
            st.subheader("结果列表")
            if filtered_results:
                results_df = pd.DataFrame([r['metadata'] for r in filtered_results])
                def paginate_dataframe(df, page_key, page_size_key=None):
                    if df.empty:
                        return df, 0, 0, 0, 0
                    page = st.session_state.display_page.get(page_key, 0)
                    page_size = st.session_state.display_page.get("page_size", 10)
                    total_pages = (len(df) + page_size - 1) // page_size
                    page = max(0, min(page, total_pages - 1))
                    start_idx = page * page_size
                    end_idx = min(start_idx + page_size, len(df))
                    return df.iloc[start_idx:end_idx], page, total_pages, start_idx, end_idx
                def render_pagination_controls(page_key, total_pages, current_page):
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
                current_df, current_page, total_pages, start_idx, end_idx = paginate_dataframe(results_df, "results_list")
                current_page_items = [filtered_results[i] for i in range(start_idx, end_idx)]
                st.write(f"结果 ({len(filtered_results)}条，显示第 {start_idx+1}-{end_idx} 条):")
                render_pagination_controls("results_list", total_pages, current_page)
                select_all = st.checkbox("全选当前页", key="select_all_results")
                if select_all:
                    for item in current_page_items:
                        cache_key = item['metadata'].get('cache_key', '')
                        if cache_key and cache_key not in st.session_state.to_delete_results:
                            st.session_state.to_delete_results.append(cache_key)
                for i, item in enumerate(current_page_items):
                    metadata = item['metadata']
                    result = item['result']
                    cache_key = metadata.get('cache_key', '')
                    col1, col2 = st.columns([1, 11])
                    with col1:
                        is_selected = cache_key in st.session_state.to_delete_results
                        if st.checkbox("", value=is_selected or select_all, key=f"select_{cache_key}"):
                            if cache_key not in st.session_state.to_delete_results:
                                st.session_state.to_delete_results.append(cache_key)
                        else:
                            if cache_key in st.session_state.to_delete_results:
                                st.session_state.to_delete_results.remove(cache_key)
                    with col2:
                        with st.expander(f"结果 {start_idx + i + 1}: {metadata.get('title', '')[:50]}..."):
                            if st.button("查看详情", key=f"view_{i}"):
                                st.session_state.selected_result = item
                                st.session_state.show_detail_view = True
                                st.rerun()
                            st.markdown(f"**年份**: {metadata.get('year', '')}")
                            st.markdown(f"**读取时领域**: {metadata.get('area', '')}")
                            st.markdown(f"**读取时方法**: {metadata.get('method', '')}")
                            
                            # 添加匹配后的方法和领域分类信息
                            relevant_keywords = result.get('relevant_keywords', [])
                            
                            # 确定匹配后的方法分类
                            if relevant_keywords:
                                # 导入关键词分类
                                from utils.llm_processor import get_keywords
                                keywords_dict = get_keywords()
                                
                                # 获取各方法的关键词集合
                                ml_keywords = set(keywords_dict.get("machine learning", []))
                                dl_keywords = set(keywords_dict.get("deep learning", []))
                                llm_keywords = set(keywords_dict.get("LLMs", []))
                                
                                # 与关键词列表求交集
                                relevant_kw_set = set(relevant_keywords)
                                ml_match = relevant_kw_set.intersection(ml_keywords)
                                dl_match = relevant_kw_set.intersection(dl_keywords)
                                llm_match = relevant_kw_set.intersection(llm_keywords)
                                
                                # 确定方法分类
                                match_method = "未匹配"
                                if llm_match:
                                    match_method = "LLMs"
                                elif dl_match:
                                    match_method = "deep learning"
                                elif ml_match:
                                    match_method = "machine learning"
                                
                                st.markdown(f"**匹配后方法**: {match_method}")
                            else:
                                st.markdown("**匹配后方法**: 无匹配关键词")
                            
                            # 显示领域分类信息
                            if "application_domains" in result:
                                domains = result.get("application_domains", [])
                                if domains and domains != ["None"]:
                                    st.markdown(f"**应用领域**: {', '.join(domains)}")
                                else:
                                    st.markdown("**应用领域**: 未应用于特定金融领域")
                            
                            # 显示关键词匹配信息
                            if relevant_keywords:
                                st.markdown(f"**相关关键词数量**: {len(relevant_keywords)}")
                                max_kw_display = 5  # 在列表视图中最多显示5个关键词
                                display_kws = relevant_keywords[:max_kw_display]
                                display_text = ', '.join(display_kws)
                                if len(relevant_keywords) > max_kw_display:
                                    display_text += f"... (共{len(relevant_keywords)}个)"
                                st.markdown(f"**关键词**: {display_text}")
                            else:
                                st.markdown("**无匹配关键词**")
                            
                            # 删除功能
                            if st.button("删除", key=f"delete_{i}"):
                                try:
                                    # 根据阶段选择对应的缓存管理器
                                    if metadata.get('stage') == 2:
                                        delete_success = stage2_cache_manager.delete_result(cache_key)
                                    else:
                                        delete_success = cache_manager.delete_result(cache_key)
                                        
                                    if delete_success:
                                        st.success("已删除")
                                        if cache_key in st.session_state.to_delete_results:
                                            st.session_state.to_delete_results.remove(cache_key)
                                        # 刷新缓存
                                        if metadata.get('stage') == 2:
                                            st.session_state.stage2_results_cache = None
                                        else:
                                            st.session_state.results_cache = None
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("删除失败")
                                except Exception as e:
                                    st.error(f"删除结果时出错: {str(e)}")
                if not select_all:
                    current_page_keys = [item['metadata'].get('cache_key', '') for item in current_page_items]
                    for cache_key in list(st.session_state.to_delete_results):
                        if cache_key in current_page_keys:
                            checkbox_key = f"select_{cache_key}"
                            if checkbox_key in st.session_state and not st.session_state[checkbox_key]:
                                st.session_state.to_delete_results.remove(cache_key)
                if len(filtered_results) > 10:
                    st.info(f"当前显示第 {start_idx + 1} - {end_idx} 条，共 {len(filtered_results)} 条结果")
            else:
                st.info("没有找到符合条件的结果")
        # ====== 新增：底部统计表格 ======
        if 'loaded_data' in st.session_state and st.session_state.loaded_data is not None:
            df = st.session_state.loaded_data
            ml_titles = set(df[df['method'] == 'machine learning']['title'])
            dl_titles = set(df[df['method'] == 'deep learning']['title'])
            llm_titles = set(df[df['method'] == 'LLMs']['title'])
            all_titles = set(df['title'])
            ml_dl = ml_titles & dl_titles
            dl_llm = dl_titles & llm_titles
            ml_llm = ml_titles & llm_titles
            ml_dl_llm = ml_titles & dl_titles & llm_titles
            unique_titles = ml_titles | dl_titles | llm_titles
            total_loaded = len(df)
            unique_count = len(unique_titles)
            cross_table = pd.DataFrame({
                '(ML∩DL)': [len(ml_dl)],
                '(DL∩LLM)': [len(dl_llm)],
                '(ML∩LLM)': [len(ml_llm)],
                '(ML∩DL∩LLM)': [len(ml_dl_llm)]
            }, index=['方法交叉文献数'])
            # 处理完成/未完成数量
            cache_manager = CacheManager()
            processed = cache_manager.get_all_processed_items()
            processed_titles_list = [
                item['metadata'].get('title') if 'metadata' in item else item.get('title')
                for item in processed if ('metadata' in item and item['metadata'].get('title')) or ('title' in item and item.get('title'))
            ]
            processed_titles = set(processed_titles_list)
            processed_count = len(processed_titles & unique_titles)
            unprocessed_count = unique_count - processed_count
            # 新增：处理结果独特title数、交集title数、重复title数
            processed_unique_count = len(processed_titles)
            processed_intersection_count = len(processed_titles & unique_titles)
            processed_duplicate_count = len(processed_titles_list) - processed_unique_count
            stat_table = pd.DataFrame({
                '总加载数据量': [total_loaded],
                '独特文献数': [unique_count],
                'ML∩DL': [len(ml_dl)],
                'DL∩LLM': [len(dl_llm)],
                'ML∩LLM': [len(ml_llm)],
                'ML∩DL∩LLM': [len(ml_dl_llm)],
                '已处理文献数': [processed_count],
                '未处理文献数': [unprocessed_count],
                '处理结果独特title数': [processed_unique_count],
                '处理结果与加载数据交集title数': [processed_intersection_count],
                '处理结果重复title数': [processed_duplicate_count]
            })
            st.markdown("---")
            st.markdown("**数据统计总览**")
            st.dataframe(stat_table)
            st.info("\n\n**说明：**\n- 独特文献数是指加载数据后所有ML、DL、LLM方法下title的并集。\n- 处理结果数量统计的是所有处理结果的条数，若同一文献被不同方法多次处理或缓存中有历史遗留数据，可能导致数量大于独特文献数。\n- 处理结果独特title数统计处理结果中title去重后的数量。\n- 处理结果与加载数据交集title数为两者共有的文献数。\n- 处理结果重复title数为处理结果中title出现多次的数量。\n- 若发现数量不一致，建议检查处理流程是否有重复处理、缓存未清理或数据源不一致等问题。\n")
            # ====== 新增：展示重复title的成对数据及deep seek结果 ======
            st.markdown("---")
            st.markdown("**重复title处理结果对比**")
            # 构建title到处理结果的映射
            title_to_items = defaultdict(list)
            for item in processed:
                title = item['metadata'].get('title') if 'metadata' in item else item.get('title')
                if title:
                    title_to_items[title].append(item)
            # 找出重复title
            repeated_titles = [t for t, items in title_to_items.items() if len(items) > 1]
            if not repeated_titles:
                st.success("没有发现重复title的处理结果。")
            else:
                # 添加展开/收起控制
                show_all = st.checkbox("展开显示所有重复文献的详细信息")
                st.write(f"发现{len(repeated_titles)}个重复title的文献，共涉及{sum(len(items) for t, items in title_to_items.items() if len(items) > 1)}条重复记录")
                
                for t in repeated_titles:
                    items = title_to_items[t]
                    with st.expander(f"标题: {t}", expanded=show_all):
                        # 直接展示所有信息
                        st.markdown("---")
                        
                        for idx, item in enumerate(items):
                            # 从元数据中获取cache_key
                            meta = item['metadata'] if 'metadata' in item else item
                            cache_key = meta.get('cache_key', '')
                            
                            # 使用cache_manager.get_cached_result方法获取完整缓存结果
                            full_result = cache_manager.get_cached_result(cache_key) if cache_key else None
                            
                            # 基本元数据信息
                            st.markdown(f"### 第{idx+1}条数据")
                            st.markdown(f"**领域**: {meta.get('area','')}, **方法**: {meta.get('method','')}, **数据源**: {meta.get('source','')}")
                            st.markdown(f"**Cache Key**: `{cache_key}`")
                            
                            # 从处理结果和缓存中获取关键词匹配信息
                            result = item.get('result', {})
                            relevant_keywords = result.get('relevant_keywords', [])
                            explanations = result.get('explanations', {})
                            
                            # 若有full_result，优先使用它的信息
                            if full_result:
                                relevant_keywords = full_result.get('relevant_keywords', relevant_keywords)
                                explanations = full_result.get('explanations', explanations)
                            
                            # 显示匹配关键词及解释
                            st.markdown("**匹配关键词结果**:")
                            if relevant_keywords:
                                for kw in relevant_keywords:
                                    explanation = explanations.get(kw, '无解释')
                                    st.markdown(f"- **{kw}**: {explanation}")
                            else:
                                reason = explanations.get('reason', '无原因')
                                st.markdown(f"- **无匹配关键词**: {reason}")
                            
                            # 显示完整缓存内容(改用按钮+代码块方式，避免嵌套expander)
                            show_cache = st.checkbox(f"查看完整缓存数据 #{idx+1}", key=f"show_cache_{t}_{idx}")
                            if show_cache and full_result:
                                st.code(json.dumps(full_result, ensure_ascii=False, indent=2), language="json")
                            
                            if idx < len(items) - 1:  # 不是最后一项时添加分隔线
                                st.markdown("---")

# ========== 统计分析页面 ==========
def render_statistics_page():
    st.header("📈 统计分析")
    from utils.cache_manager import CacheManager
    import pandas as pd
    import plotly.express as px
    from utils.llm_processor import get_keywords
    import os
    
    # 初始化第一阶段和第二阶段缓存管理器
    cache_manager = CacheManager()
    stage2_cache_manager = Stage2CacheManager()
    
    # 统计分析数据缓存文件路径
    stats_cache_dir = os.path.join("cache", "stats_cache")
    os.makedirs(stats_cache_dir, exist_ok=True)
    stats_cache_file = os.path.join(stats_cache_dir, "stats_dataframe.pkl")
    stats_hash_file = os.path.join(stats_cache_dir, "data_hash.txt")
    
    # 获取当前数据的哈希值
    def get_current_data_hash():
        # 获取第一阶段缓存文件夹的修改时间
        stage1_mtime = 0
        stage1_cache_dir = os.path.join("cache", "results")
        if os.path.exists(stage1_cache_dir):
            stage1_mtime = os.path.getmtime(stage1_cache_dir)
        
        # 获取第二阶段缓存文件夹的修改时间
        stage2_mtime = 0
        stage2_cache_dir = os.path.join("cache", "stage2_results")
        if os.path.exists(stage2_cache_dir):
            stage2_mtime = os.path.getmtime(stage2_cache_dir)
        
        # 组合时间戳创建哈希值
        data_hash = f"{stage1_mtime}_{stage2_mtime}"
        return hashlib.md5(data_hash.encode()).hexdigest()
    
    # 检查缓存是否有效
    def is_cache_valid():
        if not os.path.exists(stats_cache_file) or not os.path.exists(stats_hash_file):
            return False
        
        # 读取保存的哈希值
        with open(stats_hash_file, 'r') as f:
            saved_hash = f.read().strip()
        
        # 比较当前哈希值与保存的哈希值
        current_hash = get_current_data_hash()
        return saved_hash == current_hash
    
    # 尝试从缓存加载数据
    @st.cache_data
    def load_cached_data_frame():
        if is_cache_valid():
            try:
                with st.spinner("正在从缓存加载数据..."):
                    with open(stats_cache_file, 'rb') as f:
                        df = pickle.load(f)
                    st.success("成功从缓存加载数据")
                    return df
            except Exception as e:
                st.warning(f"从缓存加载数据失败: {str(e)}")
        
        # 如果缓存无效或加载失败，返回None
        return None
    
    # 尝试从缓存加载数据
    df = load_cached_data_frame()
    
    # 如果缓存无效或加载失败，重新处理数据
    if df is None:
        with st.spinner("正在从原始数据构建分析数据..."):
            start_time = time.time()
            
            # 显示数据处理进度
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 设置进度条步骤
            total_steps = 5
            current_step = 0
            
            def update_progress(step, message):
                nonlocal current_step
                current_step = step
                progress_bar.progress(step / total_steps)
                status_text.text(f"步骤 {step}/{total_steps}: {message}")
            
            # 步骤1: 加载数据
            update_progress(1, "正在加载第一阶段结果...")
            
            # 加载第一阶段结果
            if "results_cache" not in st.session_state or st.session_state.results_cache is None:
                st.session_state.results_cache = cache_manager.get_results_by_filter({})
            
            # 步骤2: 加载第二阶段结果
            update_progress(2, "正在加载第二阶段结果...")
            
            # 加载第二阶段结果
            if "stage2_results_cache" not in st.session_state or st.session_state.stage2_results_cache is None:
                try:
                    # 获取第二阶段索引数据
                    stage2_index = stage2_cache_manager.get_index()
                    
                    # 将索引数据转换为与第一阶段缓存格式兼容的格式
                    stage2_results = []
                    
                    for item in stage2_index:
                        try:
                            cache_key = item.get('cache_key', '')
                            if not cache_key:
                                continue
                                
                            # 获取详细信息
                            detail = stage2_cache_manager.get_detail(cache_key)
                            if detail:
                                # 转换为第一阶段兼容格式
                                metadata = {
                                    'title': item.get('title', ''),
                                    'abstract': detail.get('paper', {}).get('abstract', ''),
                                    'year': item.get('year', ''),
                                    'area': item.get('area', ''),
                                    'method': item.get('method', ''),
                                    'source': item.get('source', ''),
                                    'cache_key': cache_key,
                                    'timestamp': item.get('timestamp', ''),
                                    'stage': 2,  # 标记为第二阶段数据
                                    'stage1_cache_key': item.get('stage1_cache_key', '') or detail.get('paper', {}).get('cache_key', '')  # 添加关联到第一阶段的缓存键
                                }
                                
                                # 获取第一阶段的处理结果，以便获取关键词解释
                                stage1_explanations = {}
                                if 'paper' in detail and 'cache_key' in detail['paper']:
                                    stage1_cache_key = detail['paper'].get('cache_key', '')
                                    if stage1_cache_key:
                                        try:
                                            # 从第一阶段缓存中获取解释内容
                                            stage1_result = cache_manager.get_cached_result(stage1_cache_key)
                                            if stage1_result and 'explanations' in stage1_result:
                                                stage1_explanations = stage1_result.get('explanations', {})
                                        except Exception:
                                            pass  # 忽略获取第一阶段解释的错误
                                
                                result = {
                                    'relevant_keywords': item.get('stage1_keywords', []),
                                    'application_domains': item.get('application_domains', []),
                                    'justification': detail.get('domain_result', {}).get('justification', ''),
                                    'explanations': stage1_explanations,  # 添加从第一阶段获取的关键词解释
                                    'success': True
                                }
                                
                                stage2_results.append({'metadata': metadata, 'result': result})
                        except Exception:
                            continue  # 忽略单个项目的处理错误
                    
                    st.session_state.stage2_results_cache = stage2_results
                    
                except Exception:
                    st.session_state.stage2_results_cache = []
            
            # 步骤3: 构建DataFrame
            update_progress(3, "正在构建数据框架...")
            
            # 合并第一阶段和第二阶段结果
            all_results = []
            
            # 添加第一阶段结果
            if st.session_state.results_cache:
                # 标记第一阶段数据
                for item in st.session_state.results_cache:
                    if 'metadata' in item:
                        item['metadata']['stage'] = 1
                all_results.extend(st.session_state.results_cache)
            
            # 添加第二阶段结果
            if st.session_state.stage2_results_cache:
                all_results.extend(st.session_state.stage2_results_cache)
            
            if not all_results:
                st.info("暂无处理结果。请先在LLM处理页面处理数据。")
                progress_bar.empty()
                status_text.empty()
                return
            
            # 构建DataFrame
            df = pd.DataFrame([{
                **item['metadata'],
                'relevant_keywords': item['result'].get('relevant_keywords', []),
                'application_domains': item['result'].get('application_domains', []),
                'method': item['metadata'].get('method', ''),
                'source': item['metadata'].get('source', ''),
                'area': item['metadata'].get('area', ''),
                'cache_key': item['metadata'].get('cache_key', ''),
                'stage': item['metadata'].get('stage', 1),  # 默认为第一阶段
            } for item in all_results])
            
            # 只保留有年份的数据
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year'])
            df['year'] = df['year'].astype(int)
            
            # 步骤4: 进行关键词匹配和分类
            update_progress(4, "正在进行关键词匹配和分类...")
            
            # ====== 基于关键词匹配结果对文献进行分类 ======
            # 获取三个方法的关键词
            keywords_dict = get_keywords()
            ml_keywords = keywords_dict.get("machine learning", [])
            dl_keywords = keywords_dict.get("deep learning", [])
            llm_keywords = keywords_dict.get("LLMs", [])
            
            # 确保relevant_keywords是列表
            def ensure_list(value):
                if isinstance(value, str):
                    try:
                        import json
                        return json.loads(value)
                    except:
                        return []
                elif isinstance(value, list):
                    return value
                return []
            
            # 预处理relevant_keywords列，确保是列表格式
            df['relevant_keywords'] = df['relevant_keywords'].apply(ensure_list)
            
            # 将关键词列表转换为集合以加速匹配
            ml_keywords_set = set(ml_keywords)
            dl_keywords_set = set(dl_keywords)
            llm_keywords_set = set(llm_keywords)
            
            # 优化关键词匹配逻辑，使用向量化操作
            def check_match(keywords_list, target_keywords_set):
                if not keywords_list:
                    return False
                return bool(set(keywords_list).intersection(target_keywords_set))
            
            # 使用向量化操作计算匹配标记
            df['has_match'] = df['relevant_keywords'].apply(lambda x: len(x) > 0)
            df['ml_match'] = df['relevant_keywords'].apply(lambda x: check_match(x, ml_keywords_set) if x else False)
            df['dl_match'] = df['relevant_keywords'].apply(lambda x: check_match(x, dl_keywords_set) if x else False)
            df['llm_match'] = df['relevant_keywords'].apply(lambda x: check_match(x, llm_keywords_set) if x else False)
            
            # 使用条件表达式创建匹配分类
            df['match_class'] = 'no_match'
            df.loc[df['ml_match'] & ~df['dl_match'] & ~df['llm_match'], 'match_class'] = 'ML'
            df.loc[df['dl_match'] & ~df['llm_match'], 'match_class'] = 'DL'
            df.loc[df['llm_match'], 'match_class'] = 'LLM'
            
            # 处理application_domains列，确保是列表格式
            df['application_domains'] = df['application_domains'].apply(ensure_list)
            
            # 步骤5: 缓存处理结果
            update_progress(5, "正在缓存处理结果...")
            
            # 缓存DataFrame到硬盘
            try:
                with open(stats_cache_file, 'wb') as f:
                    pickle.dump(df, f)
                
                # 保存当前数据哈希值
                current_hash = get_current_data_hash()
                with open(stats_hash_file, 'w') as f:
                    f.write(current_hash)
                
                end_time = time.time()
                
                # 清除进度条和状态文本
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"数据处理完成并已缓存到硬盘（耗时: {end_time - start_time:.2f}秒）")
            except Exception as e:
                # 清除进度条和状态文本
                progress_bar.empty()
                status_text.empty()
                
                st.warning(f"缓存数据到硬盘失败: {str(e)}")
    
    # 存储全部结果以供后续使用
    all_results = []
    for _, row in df.iterrows():
        metadata = {col: row[col] for col in df.columns if col not in ['relevant_keywords', 'application_domains']}
        result = {
            'relevant_keywords': row['relevant_keywords'],
            'application_domains': row['application_domains'],
        }
        all_results.append({'metadata': metadata, 'result': result})
    
    st.subheader("统计分析")
    
    # ====== 添加处理阶段统计 ======
    st.markdown("### 处理阶段分布")
    stage_counts = df['stage'].value_counts().reset_index()
    stage_counts.columns = ['阶段', '数量']
    stage_counts['阶段'] = stage_counts['阶段'].map({1: '第一阶段', 2: '第二阶段'})
    st.dataframe(stage_counts)
    
    # ====== 添加领域分类统计 ======
    st.markdown("### 第二阶段领域分类统计")
    # 只筛选第二阶段的数据
    stage2_df = df[df['stage'] == 2]
    
    if not stage2_df.empty:
        # 展开应用领域列表，以便统计
        domain_counts = {}
        no_domain_count = 0  # 统计无应用领域的数量
        
        for domains in stage2_df['application_domains']:
            # 确保domains是列表，处理可能的JSON字符串情况
            if isinstance(domains, str):
                try:
                    import json
                    domains = json.loads(domains)
                except:
                    domains = []
                    
            if isinstance(domains, list):
                if not domains or domains == ["None"]:
                    no_domain_count += 1  # 增加无应用领域计数
                else:
                    for domain in domains:
                        if domain != "None":
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        if domain_counts or no_domain_count > 0:
            # 添加"无应用领域"到统计中
            domain_data = list(domain_counts.items())
            if no_domain_count > 0:
                domain_data.append(("无应用领域", no_domain_count))
                
            domain_df = pd.DataFrame({
                '应用领域': [item[0] for item in domain_data],
                '数量': [item[1] for item in domain_data]
            }).sort_values('数量', ascending=False)
            
            # 显示领域分类统计
            st.dataframe(domain_df)
            
            # 绘制饼图
            fig = px.pie(domain_df, values='数量', names='应用领域', title='第二阶段领域分类分布')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("第二阶段数据中没有领域分类信息")
    else:
        st.info("没有第二阶段的处理数据")
        
    # ====== 修改年度关键词匹配统计，使用匹配后的数据 ======
    def yearly_match_stats(df, source):
        # 只保留第一阶段的数据 并筛选源
        df_filtered = df[(df['stage'] == 1) & (df['source'] == source)]
        
        # 去除同一论文多次处理的情况（基于title去重）
        df_filtered = df_filtered.drop_duplicates(subset=['title'], keep='first')
        
        if df_filtered.empty:
            return pd.DataFrame()
            
        # 按年份分组统计
        years = sorted(df_filtered['year'].unique())
        stats = []
        
        for year in years:
            year_df = df_filtered[df_filtered['year'] == year]
            
            # 使用向量化操作统计各类别数量
            ml_count = len(year_df[year_df['match_class'] == 'ML'])
            dl_count = len(year_df[year_df['match_class'] == 'DL'])
            llm_count = len(year_df[year_df['match_class'] == 'LLM'])
            no_match_count = len(year_df[year_df['match_class'] == 'no_match'])
            total = ml_count + dl_count + llm_count + no_match_count
            
            stats.append({
                'year': year,
                'ML': ml_count,
                'DL': dl_count,
                'LLM': llm_count,
                '无匹配': no_match_count,
                '总计': total
            })
        
        # 添加总计行
        if stats:
            total_row = {'year': '总计'}
            for col in ['ML', 'DL', 'LLM', '无匹配', '总计']:
                total_row[col] = sum(row[col] for row in stats)
            stats.append(total_row)
            
        return pd.DataFrame(stats)
    
    # 年度领域匹配统计（第二阶段数据）
    def yearly_domain_stats(df):
        # 只保留第二阶段的数据
        stage2_df = df[df['stage'] == 2]
        
        # 去除同一论文多次处理的情况（基于title去重）
        stage2_df = stage2_df.drop_duplicates(subset=['title'], keep='first')
        
        if stage2_df.empty:
            return pd.DataFrame()
        
        # 收集所有可能的应用领域值
        all_domains = set()
        for domains in stage2_df['application_domains']:
            if domains and domains != ["None"]:
                all_domains.update(domains)
        
        # 去重并排序
        unique_domains = sorted(all_domains)
        
        if not unique_domains:
            return pd.DataFrame()
        
        # 创建年度领域统计
        years = sorted(stage2_df['year'].unique())
        stats = []
        
        for year in years:
            year_df = stage2_df[stage2_df['year'] == year]
            row = {'year': year}
            
            # 使用DataFrame操作统计领域在该年份的分布
            for domain in unique_domains:
                # 统计每年中有特定领域的文章数
                domain_count = sum(1 for domains in year_df['application_domains'] 
                                 if domains and domains != ["None"] and domain in domains)
                row[domain] = domain_count
            
            # 计算总计
            row['总计'] = sum(row[domain] for domain in unique_domains)
            stats.append(row)
        
        # 添加总计行
        if stats:
            total_row = {'year': '总计'}
            for domain in unique_domains:
                total_row[domain] = sum(row[domain] for row in stats)
            total_row['总计'] = sum(total_row[domain] for domain in unique_domains)
            stats.append(total_row)
        
        return pd.DataFrame(stats)
    
    st.markdown("---")
    
    # 第一阶段年度关键词匹配统计
    st.markdown("### 第一阶段年度关键词匹配统计（去重）")
    
    for src in ['CNKI', 'WOS']:
        st.markdown(f"#### {src} 年度关键词匹配统计")
        src_df = df[df['source'] == src]
        if not src_df.empty:
            stats_df = yearly_match_stats(src_df, src)
            st.dataframe(stats_df)
        else:
            st.info(f"没有 {src} 来源的数据。")
    
    # 第二阶段年度领域匹配统计
    st.markdown("---")
    st.markdown("### 第二阶段年度领域匹配统计（去重）")
    
    for src in ['CNKI', 'WOS']:
        st.markdown(f"#### {src} 年度领域匹配统计")
        src_df = df[df['source'] == src]
        if not src_df.empty:
            stats_df = yearly_domain_stats(src_df)
            if not stats_df.empty:
                st.dataframe(stats_df)
            else:
                st.info(f"没有 {src} 的第二阶段领域匹配数据。")
        else:
            st.info(f"没有 {src} 来源的数据。")
    
    # ====== 添加多维度趋势分析 ======
    st.markdown("---")
    st.markdown("### 多维度趋势分析")
    
    # 只筛选第二阶段数据
    trend_df = df[df['stage'] == 2].copy()
    
    # 只保留有应用领域的数据（排除领域为空或为None的数据）
    trend_df = trend_df[trend_df['application_domains'].apply(lambda x: bool(x) and x != ["None"])]
    
    if trend_df.empty:
        st.warning("没有符合条件的第二阶段数据用于趋势分析")
    else:
        # 定义技术方法的颜色和样式
        method_props = {
            "machine learning": {
                "color": '#E74C3C',   # 机器学习 红色
                "marker": "circle",
                "line": "solid",
                "name": "机器学习"
            },
            "deep learning": {
                "color": '#2980B9',   # 神经网络 蓝色
                "marker": "square",
                "line": "dash",
                "name": "深度学习"
            },
            "LLMs": {
                "color": '#27AE60',   # 大语言模型 绿色
                "marker": "diamond",
                "line": "dot",
                "name": "大语言模型"
            }
        }
        
        # 确定所有可能的年份范围
        min_year = int(trend_df['year'].min())
        max_year = int(trend_df['year'].max())
        years_range = list(range(min_year, max_year + 1))
        
        # 添加数据筛选控件
        st.markdown("#### 数据筛选")
        
        # 按来源、应用领域和技术方法进行筛选
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # 获取所有可能的数据来源
            all_sources = sorted(trend_df['source'].unique().tolist())
            selected_sources = st.multiselect(
                "数据来源",
                all_sources,
                default=all_sources,
                key="trend_sources"
            )
        
        with filter_col2:
            # 获取所有可能的技术方法
            all_methods = sorted(trend_df['method'].unique().tolist())
            selected_methods = st.multiselect(
                "技术方法",
                all_methods,
                default=all_methods,
                key="trend_methods"
            )
        
        with filter_col3:
            # 获取所有可能的应用领域
            all_domains = set()
            for domains in trend_df['application_domains']:
                if domains and domains != ["None"]:
                    all_domains.update(domains)
            all_domains = sorted(all_domains)
            
            # 默认选择前3个应用领域（如果有的话）
            default_domains = all_domains[:3] if len(all_domains) >= 3 else all_domains
            
            selected_domains = st.multiselect(
                "应用领域",
                all_domains,
                default=default_domains,
                key="trend_domains"
            )
        
        # 添加图表显示设置（移到筛选之后，图表生成之前）
        st.markdown("#### 图表显示设置")
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            font_size = st.slider(
                "字体大小",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                key="trend_chart_font_size",
                help="调整图表中所有文字的大小"
            )
        
        with chart_col2:
            # 添加调试信息显示当前字体大小
            st.info(f"当前字体大小: {font_size}px")
            # 添加字体效果预览
            st.markdown(f'<p style="font-size: {font_size}px; margin: 0;">字体效果预览</p>', unsafe_allow_html=True)
            # 添加刷新按钮
            if st.button("🔄 刷新图表", 
                        help="如果字体大小没有更新，点击此按钮刷新图表",
                        key=f"refresh_chart_{font_size}"):
                # 清除相关缓存
                if hasattr(st, '_component_cache'):
                    st._component_cache.clear()
                st.rerun()
        
        # 应用筛选条件
        if selected_sources and selected_methods and selected_domains:
            # 筛选数据源和技术方法
            filtered_df = trend_df[
                trend_df['source'].isin(selected_sources) & 
                trend_df['method'].isin(selected_methods)
            ]
            
            # 筛选应用领域（一篇论文可能对应多个领域）
            filtered_df = filtered_df[filtered_df['application_domains'].apply(
                lambda domains: any(domain in selected_domains for domain in domains)
            )]
            
            # 显示筛选后的数据数量
            st.info(f"筛选后共有 {len(filtered_df)} 条数据")
        else:
            filtered_df = pd.DataFrame()  # 创建空DataFrame
            st.warning("请至少选择一个数据来源、技术方法和应用领域")
        
        # 创建年份范围滑块控件
        st.markdown("#### 设置技术方法的年份范围")
        year_controls = {}
        
        # 为每种方法添加年份范围滑块
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**机器学习**")
            ml_min_year = st.slider(
                "起始年份",
                min_value=min_year,
                max_value=max_year,
                value=min_year,
                key="ml_min_year"
            )
            ml_max_year = st.slider(
                "结束年份",
                min_value=ml_min_year,
                max_value=max_year,
                value=max_year,
                key="ml_max_year"
            )
            year_controls["machine learning"] = (ml_min_year, ml_max_year)
        
        with col2:
            st.markdown("**深度学习**")
            dl_min_year = st.slider(
                "起始年份",
                min_value=min_year,
                max_value=max_year,
                value=min_year,
                key="dl_min_year"
            )
            dl_max_year = st.slider(
                "结束年份",
                min_value=dl_min_year,
                max_value=max_year,
                value=max_year,
                key="dl_max_year"
            )
            year_controls["deep learning"] = (dl_min_year, dl_max_year)
        
        with col3:
            st.markdown("**大语言模型**")
            llm_min_year = st.slider(
                "起始年份",
                min_value=min_year,
                max_value=max_year,
                value=min_year,
                key="llm_min_year"
            )
            llm_max_year = st.slider(
                "结束年份",
                min_value=llm_min_year,
                max_value=max_year,
                value=max_year,
                key="llm_max_year"
            )
            year_controls["LLMs"] = (llm_min_year, llm_max_year)
        
        if not filtered_df.empty:    
            # 按技术方法统计各年度论文数量
            method_data = {}
            for method in method_props.keys():
                if method not in selected_methods:
                    continue  # 跳过未被选择的方法
                    
                min_year, max_year = year_controls[method]
                # 筛选方法和年份范围内的数据
                method_df = filtered_df[(filtered_df['method'] == method) & 
                                       (filtered_df['year'] >= min_year) & 
                                       (filtered_df['year'] <= max_year)]
                
                # 按年份分组统计数量
                if not method_df.empty:
                    year_counts = method_df.groupby('year').size().reset_index(name='count')
                    # 确保所有年份都有数据（填充缺失年份为0）
                    full_years = pd.DataFrame({'year': range(min_year, max_year + 1)})
                    year_counts = pd.merge(full_years, year_counts, on='year', how='left').fillna(0)
                    # 按年份排序
                    year_counts = year_counts.sort_values('year')
                    
                    # 计算累计数量
                    year_counts['cumulative_count'] = year_counts['count'].cumsum()
                    
                    # 转换为字典形式，方便后续处理
                    method_data[method] = {
                        'years': year_counts['year'].tolist(),
                        'counts': year_counts['cumulative_count'].tolist()  # 使用累计数量
                    }
            
            # 裁剪数据，去除前面全是0的年份
            for method, data in method_data.items():
                if data['counts']:  # 确保有数据
                    # 找到第一个非零值的索引
                    first_non_zero = next((i for i, count in enumerate(data['counts']) if count > 0), len(data['counts']))
                    # 裁剪数据
                    if first_non_zero > 0:
                        data['years'] = data['years'][first_non_zero:]
                        data['counts'] = data['counts'][first_non_zero:]
                    
                    # 对数刻度处理：确保所有值都大于0（替换0为一个很小的正数）
                    data['counts'] = [max(count, 0.1) for count in data['counts']]
            
            # 创建趋势图
            if any(data['counts'] for method, data in method_data.items() if data['counts']):
                import plotly.graph_objects as go
                from scipy import interpolate
                import numpy as np
                
                # 创建图表
                fig = go.Figure()
                
                # 对每种方法添加折线
                for method, data in method_data.items():
                    if not data['counts']:
                        continue  # 跳过没有数据的方法
                    
                    years = data['years']
                    counts = data['counts']
                    props = method_props[method]
                    
                    # 如果数据点数量足够，使用样条插值生成平滑曲线
                    if len(years) > 2:
                        try:
                            # 创建插值函数
                            x_array = np.array(years)
                            y_array = np.array(counts)
                            
                            # 对年份范围创建更加密集的点，以实现平滑效果
                            x_dense = np.linspace(min(years), max(years), 100)
                            
                            # 使用样条插值 - 修改参数以避免 "m > k must hold" 错误
                            # 根据数据点数量确定样条的阶数(k)
                            # 样条的阶数k必须满足 数据点数量 m > k
                            k = min(3, len(years) - 1)  # 默认使用三次样条，但如果数据点太少则降低阶数
                            
                            if len(x_array) != len(set(x_array)):
                                # 如果有重复的x值，使用更简单的线性插值
                                y_smooth = np.interp(x_dense, x_array, y_array)
                            else:
                                # 使用样条插值，s是平滑参数，在数据点较少时增加平滑度
                                s = 0 if len(years) > 4 else 0.1  # 数据点较少时增加平滑度
                                tck = interpolate.splrep(x_array, y_array, k=k, s=s)
                                y_smooth = interpolate.splev(x_dense, tck, der=0)
                            
                            # 确保插值后的y值不为负
                            y_smooth = np.maximum(y_smooth, 0)
                            
                            # 添加平滑曲线
                            fig.add_trace(go.Scatter(
                                x=x_dense, 
                                y=y_smooth,
                                mode='lines',
                                name=props['name'],
                                line=dict(
                                    color=props['color'],
                                    dash=props['line'],
                                    width=3,
                                    shape='spline'
                                ),
                                hovertemplate='%{x}年: %{y:.0f}篇累计论文<extra></extra>'
                            ))
                        except Exception as e:
                            # 如果插值失败，直接连接原始数据点
                            st.warning(f"{props['name']}的样条插值失败: {str(e)}，将使用简单连线。")
                            fig.add_trace(go.Scatter(
                                x=years, 
                                y=counts,
                                mode='lines',
                                name=props['name'],
                                line=dict(
                                    color=props['color'],
                                    dash=props['line'],
                                    width=3
                                ),
                                hovertemplate='%{x}年: %{y:.0f}篇累计论文<extra></extra>'
                            ))
                    else:
                        # 数据点太少，无法进行样条插值，直接连接原始点
                        fig.add_trace(go.Scatter(
                            x=years, 
                            y=counts,
                            mode='lines',
                            name=props['name'],
                            line=dict(
                                color=props['color'],
                                dash=props['line'],
                                width=3
                            ),
                            hovertemplate='%{x}年: %{y:.0f}篇累计论文<extra></extra>'
                        ))
                    
                    # 添加原始数据点
                    fig.add_trace(go.Scatter(
                        x=years, 
                        y=counts,
                        mode='markers',
                        name=f"{props['name']} (原始数据)",
                        marker=dict(
                            color=props['color'],
                            size=10,
                            symbol=props['marker'],
                            line=dict(width=2, color='white')
                        ),
                        showlegend=False,
                        hovertemplate='%{x}年: %{y}篇累计论文<extra></extra>'
                    ))
                
                # 设置图表布局
                title_text = "技术方法发展趋势"
                if len(selected_sources) < len(all_sources):
                    sources_str = ", ".join(selected_sources)
                    title_text += f"（数据来源: {sources_str}）"
                
                fig.update_layout(
                    # 移除标题
                    # title=title_text,
                    xaxis_title="发表年份",
                    xaxis=dict(
                        title=dict(font=dict(size=font_size)),  # 设置X轴标题字体
                        tickfont=dict(size=font_size)           # 设置X轴刻度字体
                    ),
                    yaxis=dict(
                        type="log",  # 使用对数刻度
                        title="累计论文数量 (对数刻度)",
                        title_font=dict(size=font_size),        # 设置Y轴标题字体
                        tickfont=dict(size=font_size),          # 设置Y轴刻度字体
                        showgrid=True,  # 显示网格线
                        gridwidth=1,    # 网格线宽度
                        gridcolor='rgba(200, 200, 200, 0.3)',  # 网格线颜色
                        exponentformat="none",  # 不使用科学计数法
                        tickmode="auto",
                        nticks=10,      # 刻度数量
                        tickformat=",d"  # 数字格式化，添加千位分隔符
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=1.1,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=font_size),              # 设置图例字体
                        title=dict(
                            font=dict(
                                size=font_size  # 使用滑块控制的字体大小
                            )
                        )
                    ),
                    legend_title="",
                    height=600,
                    # 提高图表分辨率，以导出高清图片
                    width=600,  # 设置较大的宽度
                    hovermode="x unified",
                    hoverlabel=dict(
                        font_size=font_size,                    # 设置悬浮提示字体大小
                        font_family="Arial"
                    ),
                    # 增加图像质量
                    template="plotly_white",  # 使用高质量白色模板
                    font=dict(
                        family="Arial, sans-serif",
                        size=font_size  # 使用滑块控制的字体大小
                    ),
                    margin=dict(l=80, r=80, t=50, b=80)  # 增加边距
                )
                
                # 生成高分辨率图像的配置
                high_res_config = {
                    "toImageButtonOptions": {
                        "format": "png",  # 图像格式
                        "filename": "技术方法趋势分析",
                        "height": 800,  # 高分辨率高度
                        "width": 800,   # 高分辨率宽度
                        "scale": 2       # 缩放因子 (更高 = 更清晰)
                    },
                    "displaylogo": False,  # 移除Plotly logo
                    "modeBarButtonsToAdd": ["downloadImage"]  # 突出显示下载按钮
                }
                
                # 显示高分辨率图表（同时提供下载功能）
                st.markdown(f"### 技术方法趋势分析图 (当前字体大小: {font_size}px)")
                st.caption("💡 提示：可以通过上方的字体大小滑块调整图表文字大小，点击右上角的相机图标下载高清PNG图片")
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    config=high_res_config,
                    key=f"trend_chart_font_{font_size}"  # 使用字体大小作为key强制重新渲染
                )
                
                # 添加对数刻度说明
                st.caption("注：Y轴使用对数刻度，可以更好地展示不同技术方法的增长趋势，特别是当数量差异较大时。")
                
                # 显示原始数据表格
                with st.expander("查看原始数据"):
                    # 合并所有方法的数据到一个DataFrame
                    all_data = []
                    for method, data in method_data.items():
                        if data['years'] and data['counts']:
                            for year, count in zip(data['years'], data['counts']):
                                all_data.append({
                                    'year': year,
                                    'method': method_props[method]['name'],
                                    'count': count
                                })
                    
                    if all_data:
                        data_df = pd.DataFrame(all_data)
                        # 创建透视表
                        pivot_df = data_df.pivot(index='year', columns='method', values='count').fillna(0)
                        
                        # 显示表格
                        st.markdown("**各技术方法按年度累计发表论文数量**")
                        st.dataframe(pivot_df)
                        
                        # 提供下载链接
                        csv_data = pivot_df.to_csv()
                        st.download_button(
                            label="下载CSV格式数据",
                            data=csv_data,
                            file_name="method_cumulative_trends.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("没有数据可显示")
            else:
                st.warning("所选条件下没有趋势数据")
        else:
            st.warning("请选择筛选条件以显示趋势图")
    
    # 添加缓存管理按钮
    st.markdown("---")
    with st.expander("缓存管理"):
        st.info("当前统计数据已缓存到硬盘，可以通过以下按钮手动管理缓存")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("查看缓存状态"):
                if os.path.exists(stats_cache_file):
                    cache_size = os.path.getsize(stats_cache_file) / (1024 * 1024)  # 转换为MB
                    cache_time = datetime.fromtimestamp(os.path.getmtime(stats_cache_file))
                    current_hash = get_current_data_hash()
                    with open(stats_hash_file, 'r') as f:
                        saved_hash = f.read().strip()
                    is_valid = saved_hash == current_hash
                    
                    status_msg = f"缓存文件存在\n大小: {cache_size:.2f}MB\n创建时间: {cache_time}\n缓存状态: {'有效' if is_valid else '需要更新'}"
                    if is_valid:
                        st.success(status_msg)
                    else:
                        st.warning(status_msg)
                else:
                    st.warning("缓存文件不存在")
        with col2:
            if st.button("强制更新缓存"):
                # 主动清除缓存文件
                try:
                    if os.path.exists(stats_cache_file):
                        os.remove(stats_cache_file)
                    if os.path.exists(stats_hash_file):
                        os.remove(stats_hash_file)
                    # 清除Streamlit缓存
                    st.cache_data.clear()
                    st.success("缓存已清除，正在重新生成...")
                    st.rerun()  # 重新加载页面以触发缓存生成
                except Exception as e:
                    st.error(f"清除缓存失败: {str(e)}")
        with col3:
            if st.button("强制清除缓存"):
                try:
                    if os.path.exists(stats_cache_file):
                        os.remove(stats_cache_file)
                    if os.path.exists(stats_hash_file):
                        os.remove(stats_hash_file)
                    st.success("缓存已清除，下次访问将重新生成")
                    # 清除Streamlit缓存
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"清除缓存失败: {str(e)}")
        
        # 添加缓存说明
        st.markdown("""
        **缓存说明：**
        - 统计分析数据会自动缓存到硬盘，提高页面加载速度
        - 只有在原始数据有更新时才会重新生成缓存
        - 如需手动更新缓存，请点击"强制更新缓存"按钮
        - 如需临时禁用缓存，请点击"强制清除缓存"按钮
        """)
        
        # 显示最后一次数据处理时间（如果有缓存）
        if os.path.exists(stats_cache_file):
            last_process_time = datetime.fromtimestamp(os.path.getmtime(stats_cache_file))
            st.info(f"最后数据处理时间: {last_process_time}")
            # 显示当前数据统计规模
            if 'df' in locals() and isinstance(df, pd.DataFrame):
                st.info(f"当前统计数据包含 {len(df)} 条记录，{len(df.columns)} 个特征")
                # 显示数据分布统计
                st.write("数据分布统计:")
                st.write(f"- 第一阶段数据: {len(df[df['stage'] == 1])} 条")
                st.write(f"- 第二阶段数据: {len(df[df['stage'] == 2])} 条")
                st.write(f"- 数据源分布: {df['source'].value_counts().to_dict()}")
                st.write(f"- 方法分布: {df['method'].value_counts().to_dict()}")
                st.write(f"- 年份范围: {df['year'].min()} - {df['year'].max()}")
    
    # ====== 添加数据导出功能 ======
    st.markdown("---")
    st.subheader("数据导出")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        # 导出索引数据
        if st.button("导出索引数据"):
            # 创建包含两个阶段数据的DataFrame
            export_df = pd.DataFrame([{
                **item['metadata'],
                'stage': '第二阶段' if item['metadata'].get('stage') == 2 else '第一阶段',
                'relevant_keywords': ', '.join(ensure_list(item['result'].get('relevant_keywords', []))),
                'application_domains': ', '.join(ensure_list(item['result'].get('application_domains', [])))
            } for item in all_results])
            
            if not export_df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"paper_analysis_index_{timestamp}.csv"
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
        # 导出完整结果
        include_raw = st.checkbox("包含原始响应", value=False)
        stage_to_export = st.radio("选择导出阶段:", ["全部", "仅第一阶段", "仅第二阶段"])
        
        if st.button("导出完整结果"):
            # 根据选择的阶段筛选数据
            if stage_to_export == "仅第一阶段":
                export_results = [r for r in all_results if r['metadata'].get('stage') == 1]
            elif stage_to_export == "仅第二阶段":
                export_results = [r for r in all_results if r['metadata'].get('stage') == 2]
            else:
                export_results = all_results
                
            if export_results:
                export_data = []
                for item in export_results:
                    export_item = {
                        **item['metadata'],
                        'stage': '第二阶段' if item['metadata'].get('stage') == 2 else '第一阶段',
                        'relevant_keywords': ', '.join(ensure_list(item['result'].get('relevant_keywords', []))),
                        'application_domains': ', '.join(ensure_list(item['result'].get('application_domains', []))),
                        'justification': item['result'].get('justification', '')
                    }
                    
                    # 如果包含原始响应
                    if include_raw and 'raw_response' in item['result']:
                        export_item['raw_response'] = item['result']['raw_response']
                    
                    export_data.append(export_item)
                
                export_df = pd.DataFrame(export_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"paper_analysis_full_{timestamp}.csv"
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="下载完整结果CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv"
                )
            else:
                st.warning("没有数据可导出")

# ========== 主入口 ==========
def main():
    # 初始化所有Session状态，防止未初始化报错
    if "to_select_keywords" not in st.session_state:
        st.session_state.to_select_keywords = []
    if "to_delete_keywords" not in st.session_state:
        st.session_state.to_delete_keywords = []
    if "keyword_lists" not in st.session_state:
        st.session_state.keyword_lists = {}
    if "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = []
    if "loaded_data" not in st.session_state:
        st.session_state.loaded_data = None
    if "show_detail_view" not in st.session_state:
        st.session_state.show_detail_view = False
    if "selected_result" not in st.session_state:
        st.session_state.selected_result = None
    if "last_loaded_source" not in st.session_state:
        st.session_state.last_loaded_source = ""
    if "last_loaded_files" not in st.session_state:
        st.session_state.last_loaded_files = []
    if "to_delete_results" not in st.session_state:
        st.session_state.to_delete_results = []
    if "confirm_clear_cache" not in st.session_state:
        st.session_state.confirm_clear_cache = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []
    if "processed_items" not in st.session_state:
        st.session_state.processed_items = []
    if "display_page" not in st.session_state:
        st.session_state.display_page = {
            "unprocessed": 0,
            "processed": 0,
            "processing": 0,
            "results_list": 0,
            "cached": 0,
            "page_size": 10
        }
    if "annotation_results" not in st.session_state:
        st.session_state.annotation_results = {}
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "user_prompt_template" not in st.session_state:
        st.session_state.user_prompt_template = ""
    if "prompt_examples" not in st.session_state:
        st.session_state.prompt_examples = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "last_session_time" not in st.session_state:
        st.session_state.last_session_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "filter_no_match" not in st.session_state:
        st.session_state.filter_no_match = False
    if "deduplication" not in st.session_state:
        st.session_state.deduplication = True
    st.title("📚 结果查看与数据分析")
    page = st.sidebar.radio(
        "导航",
        ["📊 数据加载", "🔑 关键词管理", "📋 结果查看", "📈 统计分析"]
    )
    if page == "📊 数据加载":
        render_data_loading_page()
    elif page == "🔑 关键词管理":
        render_keywords_management_page()
    elif page == "📋 结果查看":
        render_results_view_page()
    elif page == "📈 统计分析":
        render_statistics_page()

if __name__ == "__main__":
    main()
