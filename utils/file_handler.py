import os, hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from utils.logger_handler import logger


def get_file_md5_hex(file_path): # 获取文件的md5的十六进制字符串
    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件不存在：{file_path}")
        return None

    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return None

    md5_obj = hashlib.md5()

    chunk_size = 4096 # 4KB切片，避免内存溢出
    try:
        with open(file_path, 'rb') as f: # 必须rb二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            '''
            :=海象运算符，等同于
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            '''
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}计算失败：{e}")
        return None

def listdir_with_allowed_type(path: str, allowed_type: tuple[str]): # 返回文件夹内的文件列表（允许的文件后缀）
    files = []

    if not os.path.isdir(path):
        logger.error(f"[文件列表类型]路径{path}不是文件夹")
        return []

    for f in os.listdir(path):
        if f.endswith(allowed_type):  # endswith用于检查文件后缀
            files.append(os.path.join(path, f))

    return files

def pdf_loader(file_path, passwd=None) -> list[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.debug(f"[PDF加载]成功加载文件: {file_path}")
        return documents
    except Exception as e:
        logger.error(f"[PDF加载]加载文件{file_path}时出错: {e}")
        return []


def txt_loader(file_path: str) -> list[Document]:
    """加载TXT文件，自动检测编码"""
    # 尝试多种编码顺序
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
    for encoding in encodings:
        try:
            loader = TextLoader(file_path, encoding=encoding)
            documents = loader.load()
            logger.debug(f"[TXT加载]成功使用{encoding}编码加载文件: {file_path}")
            return documents
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"[TXT加载]加载文件{file_path}时出错: {e}")
            continue

        # 如果所有编码都失败
    logger.error(f"[TXT加载]无法加载文件{file_path}，尝试了编码: {encodings}")
    return []

def docx_loader(file_path: str) -> list[Document]:
    """加载DOCX文件"""
    try:
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        logger.debug(f"[DOCX加载]成功加载文件: {file_path}")
        return documents
    except Exception as e:
        logger.error(f"[DOCX加载]加载文件{file_path}时出错: {e}")
        return []