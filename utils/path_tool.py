'''
为整个工程提供统一的路径
'''

import os

def get_project_root() -> str:
    """获取工程根目录"""
    current_file = os.path.abspath(__file__) # 获取当前文件绝对路径
    current_dir = os.path.dirname(current_file) # dirname作用为去掉文件名，即上一级路径
    project_root = os.path.dirname(current_dir) # 再去上一级获取到工程根目录

    return project_root

def get_abs_path(relative_path):
    """提供相对路径获取绝对路径"""
    project_root = get_project_root()
    abs_path = os.path.join(project_root, relative_path)
    return abs_path


