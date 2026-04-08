import os
from langchain_core.documents import Document
from utils.config_handler import milvus_config
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from pymilvus import MilvusClient
from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field



class VectorStoreService:
    def __init__(self, drop_existing: bool = False):
        """初始化向量存储服务"""
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

    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
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

    def get_retriever(self):
        """返回兼容 LangChain 的 retriever"""


        class MilvusRetriever(BaseRetriever):
            """Milvus检索器"""
            vector_store: 'VectorStoreService' = Field(description="Milvus向量存储服务")
            k: int = Field(default=3, description="返回的文档数量")

            def _get_relevant_documents(
                    self, query: str, *, run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return self.vector_store.similarity_search(query, k=self.k)

        return MilvusRetriever(vector_store=self, k=milvus_config.get("k", 3))

    def delete_collection(self):
        """删除整个 collection（慎用）"""
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
        要计算文件的MD5做去重
        """

        def check_md5_hex(md5_for_check: str):
            md5_file_path = get_abs_path(milvus_config["md5_hex_store"])
            if not os.path.exists(md5_file_path):
                # 创建文件
                with open(md5_file_path, "w", encoding="utf-8") as f:
                    pass
                return False

            with open(md5_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(milvus_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

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

            if check_md5_hex(md5_hex):
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

                # 将内容存入向量库
                self.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库] {path} 内容加载成功，分片数: {len(split_document)}")
            except Exception as e:
                logger.error(f"[加载知识库] {path} 加载失败：{str(e)}", exc_info=True)
                continue

        logger.info(f"知识库加载完成，当前总文档数: {self.count_documents()}")


if __name__ == '__main__':
    # 重要：第一次运行时，删除旧collection并重建
    # 设置drop_existing=True来删除旧的collection
    vs = VectorStoreService(drop_existing=False)

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