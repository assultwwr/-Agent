import os
from langchain_core.documents import Document
from utils.config_handler import milvus_config, rag_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, docx_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from pymilvus import MilvusClient
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from pydantic import Field



class VectorStoreService:
    def __init__(self, drop_existing: bool = False, auto_load: bool = True):
        """初始化向量存储服务
        
        Args:
            drop_existing: 是否删除已存在的collection并重建
            auto_load: 是否自动加载data文件夹中的新文档
        """
        # 使用 MilvusClient
        self.client = MilvusClient(uri=milvus_config["uri"])
        self.collection_name = milvus_config["collection_name"]

        # 如果需要删除旧的collection
        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.warning(f"已删除旧的Collection: {self.collection_name}")

        # 检查并创建collection
        if not self.client.has_collection(self.collection_name):
            # 获取embedding维度
            test_embedding = embedding_model.embed_query("test")
            dimension = len(test_embedding)

            # 创建collection使用简化的schema，只定义必要字段
            # 其他字段将作为动态字段自动处理
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=dimension,
                metric_type="IP",  # 内积相似度
                auto_id=True,
                enable_dynamic_field=True  # 关键：启用动态字段，允许任意metadata
            )
            logger.info(f"创建新的Collection: {self.collection_name}, 维度: {dimension}")
        else:
            logger.info(f"使用已存在的Collection: {self.collection_name}")

            # 确保collection已加载
            try:
                # 获取collection统计信息会触发加载
                self.client.get_collection_stats(self.collection_name)
                logger.info(f"Collection {self.collection_name} 已加载")
            except Exception as e:
                logger.warning(f"Collection加载状态检查: {e}")

        # 文本分割器
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_config["chunk_size"],
            chunk_overlap=milvus_config["chunk_overlap"],
            separators=milvus_config["separators"],
            length_function=len,
        )
        
        # 自动加载文档
        if auto_load:
            # 先同步文件系统（删除已不存在的文件）
            self.sync_documents()
            # 再加载新文档或更新已修改的文档
            self.load_document()

    def add_documents(self, documents: List[Document], file_path: str = None, md5_hex: str = None):
        """添加文档到向量库
        
        Args:
            documents: 文档列表
            file_path: 源文件路径（用于同步删除）
            md5_hex: 文件 MD5（用于同步删除）
        """
        if not documents:
            return

        # 准备数据
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # 生成向量
        embeddings = embedding_model.embed_documents(texts)

        # 准备插入数据只包含基本字段，避免schema冲突
        data = []
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            # 构建基础数据
            item = {
                "text": text,
                "vector": embedding,
                "producer": "auto_generated"
            }

            # 添加文件溯源信息（用于同步删除）
            if file_path:
                item["source_file"] = file_path
            if md5_hex:
                item["file_md5"] = md5_hex

            # 只添加简单的metadata字段（过滤掉可能导致schema冲突的字段）
            for key, value in metadata.items():
                # 跳过None值
                if value is None:
                    continue
                # 只添加简单类型（字符串、数字、布尔值）
                if isinstance(value, (str, int, float, bool)):
                    item[key] = value

            data.append(item)

        # 批量插入
        try:
            result = self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            logger.info(f"成功插入 {len(data)} 条数据")
            return result
        except Exception as e:
            logger.error(f"插入数据失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """相似度搜索"""
        if k is None:
            k = milvus_config.get("k", 3)

        # 确保collection已加载
        try:
            # 尝试获取collection统计信息来触发加载
            self.client.get_collection_stats(self.collection_name)
        except Exception as e:
            logger.warning(f"Collection可能未加载: {e}")

        # 将查询转为向量
        query_embedding = embedding_model.embed_query(query)

        # 搜索
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=k,
                output_fields=["text", "*"]  # 输出text字段和所有动态字段
            )
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

        documents = []
        for result in results[0]:
            entity = result['entity']

            # 提取metadata（排除text和vector字段）
            metadata = {}
            for key, value in entity.items():
                if key not in ['text', 'vector']:
                    metadata[key] = value

            # 添加相似度分数到metadata
            metadata['score'] = result['distance']

            doc = Document(
                page_content=entity.get('text', ''),
                metadata=metadata
            )
            documents.append(doc)

        return documents

    def get_retriever(self, k: int = None):
        """返回混合检索器（BM25 + 向量检索 + Rerank）"""
        if k is None:
            k = milvus_config.get("k", 3)
    
        # 1. 向量检索器
        class MilvusRetriever(BaseRetriever):
            """Milvus检索器"""
            vector_store: 'VectorStoreService' = Field(description="Milvus 向量存储服务")
            search_k: int = Field(default=k, description="向量检索返回的候选文档数量")
    
            def _get_relevant_documents(
                    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return self.vector_store.similarity_search(query, k=self.search_k)
    
        vector_retriever = MilvusRetriever(vector_store=self, search_k=k * 3)  # 召回更多候选文档
    
        # 2. BM25检索器（需要从已有文档构建）
        # 注意：BM25需要原始文档列表，这里从向量库检索所有文档来构建
        # 优化：如果文档数量过大，可以考虑缓存或从文件系统加载
        try:
            all_docs = self.similarity_search("", k=1000)  # 获取尽可能多的文档构建BM25索引
            if not all_docs:
                logger.warning("向量库为空，BM25检索器将无法使用")
                bm25_retriever = BM25Retriever.from_texts([""], k=k)
            else:
                bm25_retriever = BM25Retriever.from_documents(all_docs)
                bm25_retriever.k = k * 3  # BM25也召回更多候选
        except Exception as e:
            logger.error(f"构建BM25检索器失败: {e}")
            # 降级为仅使用向量检索
            return vector_retriever
    
        # 3. 混合检索（Ensemble）
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
    
        # 4. Rerank模型
        try:
            reranker_model = HuggingFaceCrossEncoder(
                model_name=get_abs_path(rag_config["rerank_model_path"]),
                model_kwargs={"device": rag_config.get("rerank_device", "cpu")}  # 从配置读取设备
            )
            compressor = CrossEncoderReranker(model=reranker_model, top_n=k)
                
            rerank_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
            logger.info("混合检索器初始化成功（BM25 + 向量 + Rerank）")
            return rerank_retriever
        except Exception as e:
            logger.warning(f"Rerank模型加载失败，降级为混合检索: {e}")
            # 降级：不使用Rerank
            return vector_retriever


    def sync_documents(self):
        """同步文件系统与向量库：删除已不存在的文件对应的向量数据"""
        logger.info("[文件同步] 开始检查文件变更...")
        
        # 0. 兼容迁移：将旧格式（纯MD5）迁移为新格式（MD5|file_path）
        self._migrate_md5_format()
        
        # 1. 获取当前 data 目录下的所有文件路径
        current_files = set(listdir_with_allowed_type(
            get_abs_path(milvus_config["data_path"]),
            tuple(milvus_config["allow_knowledge_file_type"]),
        ))
        logger.info(f"[文件同步] data 目录当前文件数: {len(current_files)}")
        
        # 2. 从 md5.text 读取已记录的文件路径
        md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
        if not os.path.exists(md5_file_path):
            logger.info("[文件同步] MD5 记录文件不存在，跳过同步检查")
            return
        
        # 解析 md5|file_path 格式
        recorded_files = {}
        with open(md5_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) == 2:
                    md5_hex, file_path = parts
                    recorded_files[file_path] = md5_hex
        
        logger.info(f"[文件同步] MD5 记录中的文件数: {len(recorded_files)}")
        
        # 3. 找出已删除的文件（在 md5 记录中但不在当前目录）
        deleted_files = set(recorded_files.keys()) - current_files
        
        if not deleted_files:
            logger.info("[文件同步] 无已删除文件，无需同步")
            return
        
        # 4. 删除已删除文件对应的向量数据
        total_deleted = 0
        for file_path in deleted_files:
            logger.info(f"[文件同步] 检测到文件已删除: {file_path}")
            count = self.delete_documents_by_file(file_path)
            total_deleted += count
            
            # 从 md5 记录中移除
            self._remove_md5_record(recorded_files[file_path], file_path)
        
        logger.info(f"[文件同步] 同步完成，共删除 {total_deleted} 条向量数据")

    def _migrate_md5_format(self):
        """迁移旧格式 MD5 记录（纯MD5 -> MD5|file_path）"""
        md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
        if not os.path.exists(md5_file_path):
            return
        
        # 读取所有记录
        with open(md5_file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # 检查是否已经是新格式
        if all("|" in line for line in lines):
            logger.debug("[MD5迁移] 已是新格式，无需迁移")
            return
        
        logger.info("[MD5迁移] 检测到旧格式，开始迁移...")
        
        # 获取当前 data 目录下的文件
        current_files = listdir_with_allowed_type(
            get_abs_path(milvus_config["data_path"]),
            tuple(milvus_config["allow_knowledge_file_type"]),
        )
        
        # 构建 MD5 -> file_path 映射
        md5_to_file = {}
        for file_path in current_files:
            md5_hex = get_file_md5_hex(file_path)
            if md5_hex:
                md5_to_file[md5_hex] = file_path
        
        # 迁移记录
        new_lines = []
        migrated_count = 0
        skipped_count = 0
        
        for line in lines:
            if "|" in line:
                # 已经是新格式
                new_lines.append(line)
            else:
                # 旧格式，尝试匹配文件
                md5_hex = line
                if md5_hex in md5_to_file:
                    file_path = md5_to_file[md5_hex]
                    new_lines.append(f"{md5_hex}|{file_path}")
                    migrated_count += 1
                else:
                    # 找不到对应文件，跳过（可能已删除）
                    skipped_count += 1
        
        # 写回文件
        with open(md5_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        
        logger.info(f"[MD5迁移] 完成：迁移 {migrated_count} 条，跳过 {skipped_count} 条（文件已删除）")

    def delete_documents_by_file(self, file_path: str) -> int:
        """根据文件路径删除向量库中的相关文档
        
        Args:
            file_path: 文件路径
        Returns:
            删除的文档数量
        """
        try:
            # Milvus 删除语法：delete(collection, filter="source_file == 'xxx'")
            result = self.client.delete(
                collection_name=self.collection_name,
                filter=f"source_file == '{file_path}'"
            )
            deleted_count = result.get('delete_count', 0)
            if deleted_count > 0:
                logger.info(f"[同步删除] 已删除 {deleted_count} 条向量: {file_path}")
            return deleted_count
        except Exception as e:
            logger.error(f"[同步删除] 删除失败 {file_path}: {e}")
            return 0

    def _remove_md5_record(self, md5_hex: str, file_path: str):
        """从 MD5 记录文件中移除指定记录"""
        md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
        if not os.path.exists(md5_file_path):
            return
        
        target_line = f"{md5_hex}|{file_path}\n"
        
        # 读取所有行，过滤掉目标行
        with open(md5_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        new_lines = [line for line in lines if line.strip() != target_line.strip()]
        
        # 写回文件
        with open(md5_file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        
        logger.debug(f"[MD5清理] 已移除记录: {md5_hex}|{file_path}")

    def delete_collection(self):
        """删除整个collection（慎用）"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.warning(f"已删除 Collection: {self.collection_name}")

    def count_documents(self) -> int:
        """统计文档数量"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return stats.get('row_count', 0)
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def load_collection(self):
        """手动加载collection到内存"""
        try:
            # 通过获取统计信息来触发加载
            self.client.get_collection_stats(self.collection_name)
            logger.info(f"Collection {self.collection_name} 已加载到内存")
        except Exception as e:
            logger.error(f"加载collection失败: {e}")

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        要计算文件的MD5做去重，支持文件变更后的自动更新
        """

        def check_md5_hex(md5_for_check: str, file_path: str):
            """检查文件MD5是否已存在（格式：md5|file_path）"""
            md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
            if not os.path.exists(md5_file_path):
                # 创建文件
                with open(md5_file_path, "w", encoding="utf-8") as f:
                    pass
                return False

            record = f"{md5_for_check}|{file_path}"
            with open(md5_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == record:
                        return True
                return False

        def save_md5_hex(md5_for_check: str, file_path: str):
            """保存文件MD5记录（格式：md5|file_path）"""
            with open(get_abs_path(milvus_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(f"{md5_for_check}|{file_path}\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            if read_path.endswith("docx"):
                return docx_loader(read_path)
            return []

        def delete_file_vectors(file_path: str):
            """删除指定文件的向量数据"""
            try:
                # 使用Milvus的delete功能，通过metadata中的source_file字段删除
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'source_file == "{file_path}"'
                )
                logger.info(f"[删除旧向量] 已删除文件 {file_path} 的向量数据")
            except Exception as e:
                logger.warning(f"[删除旧向量] 删除失败（可能无数据）: {e}")

        def clear_md5_records_for_file(file_path: str):
            """清除MD5记录中指定文件的所有记录"""
            md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
            if not os.path.exists(md5_file_path):
                return
            
            with open(md5_file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # 过滤掉该文件的记录
            new_lines = [line for line in lines if not line.strip().endswith(f"|{file_path}")]
            
            with open(md5_file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(milvus_config["data_path"]),
            tuple(milvus_config["allow_knowledge_file_type"]),
        )

        logger.info(f"开始加载知识库，共发现 {len(allowed_files_path)} 个文件")

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if not md5_hex:
                logger.warning(f"[加载知识库] {path} MD5计算失败，跳过")
                continue

            # 检查MD5是否已存在（相同文件且MD5相同则跳过）
            if check_md5_hex(md5_hex, path):
                logger.info(f"[加载知识库] {path} 内容已经存在知识库内，跳过")
                continue

            try:
                documents = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库] {path} 内没有有效文本内容，跳过")
                    continue

                split_document = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库] {path} 分片后没有有效文本内容，跳过")
                    continue

                # 如果文件已存在但MD5不同（文件被修改），先删除旧数据
                md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
                if os.path.exists(md5_file_path):
                    with open(md5_file_path, "r", encoding="utf-8") as f:
                        if any(path in line for line in f.readlines()):
                            logger.info(f"[加载知识库] {path} 文件已修改，更新向量数据")
                            delete_file_vectors(path)
                            clear_md5_records_for_file(path)

                # 添加metadata标记来源文件
                for doc in split_document:
                    doc.metadata["source_file"] = path
                    doc.metadata["file_md5"] = md5_hex

                # 将内容存入向量库
                self.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex, path)

                logger.info(f"[加载知识库] {path} 内容加载成功，分片数: {len(split_document)}")
            except Exception as e:
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                continue

        logger.info(f"知识库加载完成，当前总文档数: {self.count_documents()}")


if __name__ == '__main__':
    # 重要：第一次运行时，删除旧collection并重建
    # 设置drop_existing=True来删除旧的collection
    vs = VectorStoreService(drop_existing=True)

    # 加载文档
    vs.load_document()

    # 手动确保collection已加载（可选）
    vs.load_collection()

    # 测试检索
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")

    print(f"\n检索结果（共 {len(res)} 条）：")
    for i, r in enumerate(res, 1):
        score = r.metadata.get('score', 'N/A')
        print(f"\n--- 结果 {i} (相似度: {score}) ---")
        content = r.page_content[:200] if len(r.page_content) > 200 else r.page_content
        print(content)
        print("-" * 50)