"""
RAG 检索测试脚本
用于调试向量库检索效果
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from rag.vector_store import VectorStoreService
from utils.logger_handler import logger


def test_rag_search():
    """测试 RAG 检索功能"""
    print("=" * 50)
    print("RAG 检索测试")
    print("=" * 50)
    
    # 初始化向量库
    print("\n[1] 初始化向量库...")
    vs = VectorStoreService(drop_existing=False, auto_load=False)
    print("✓ 初始化完成")
    
    # 测试检索词
    test_queries = [
        "父母家 地址",
        "父母 城市",
        "家庭地址",
        "爸妈家在哪里",
        "父母家 三亚",
    ]
    
    print("\n[2] 开始测试检索...\n")
    
    for query in test_queries:
        print(f"-" * 50)
        print(f"检索词: {query}")
        print("-" * 50)
        
        # 执行检索
        try:
            results = vs.similarity_search(query, k=3)
            
            if not results:
                print("❌ 未返回任何结果")
            else:
                print(f"✓ 返回 {len(results)} 条结果:")
                for i, doc in enumerate(results, 1):
                    score = doc.metadata.get("score", 0)
                    content = doc.page_content[:200]  # 截取前200字符
                    source = doc.metadata.get("source", "未知")
                    
                    print(f"\n  结果 {i}:")
                    print(f"  相关度: {score:.4f}")
                    print(f"  来源: {source}")
                    print(f"  内容: {content}")
                    
        except Exception as e:
            print(f"❌ 检索失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 列出所有已索引的文件
    print("=" * 50)
    print("已索引的文件列表:")
    print("=" * 50)
    try:
        all_docs = vs.similarity_search("", k=100)  # 空搜索获取所有文档
        
        seen_sources = set()
        for doc in all_docs:
            source = doc.metadata.get("source", "")
            if source and source not in seen_sources:
                seen_sources.add(source)
                print(f"  - {os.path.basename(source)}")
    except Exception as e:
        print(f"无法获取文件列表: {e}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == "__main__":
    test_rag_search()
