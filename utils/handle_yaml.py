# -*- coding: utf-8 -*-
"""
# @File    : handle_yaml.py
# @Date    : 2024-08-02
# @Author  : jeremyzhai
# @Description : 项目内 YAML 读写统一入口（read_yaml_data / write_yaml_data），
#               其他模块禁止直接 import yaml 操作文件
"""

import yaml
import os
from utils.handle_log import log


# yaml文件读取
def read_yaml_data(file_dir: str) -> dict:
    """
    读取指定yaml文件内容
    :param file_dir: 文件路径
    :return: 解析后的 dict；文件不存在/不可读时返回 None
    """
    # 转换获取文件的绝对路径
    # BASEDIR = os.path.dirname(os.path.dirname(__file__))
    # CONF_DIR = os.path.join(BASEDIR, "case_data")
    # fileDir = os.path.join(CONF_DIR, fileDir)
    if os.path.exists(file_dir) and os.access(file_dir, os.R_OK):
        # 1- 把文件open到内存里
        with open(file_dir, encoding='utf-8') as fo:
            return yaml.safe_load(fo.read())  # 2- 使用yaml读取
    else:
        log.info(f"File {file_dir} does not exist or is not readable.")


# yaml文件写入
def write_yaml_data(file_dir: str, data: dict) -> bool:
    """
    将 dict 覆盖写入指定 yaml 文件（中文不转义、键序保持）
    :param file_dir: 文件路径（目录不存在时自动创建）
    :param data: 待写入的 dict
    :return: 写入是否成功
    """
    try:
        dir_name = os.path.dirname(file_dir)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_dir, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        log.info(f"YAML 文件写入成功: {file_dir}")
        return True
    except Exception as e:
        log.info(f"写入 YAML 文件失败 {file_dir}: {e}")
        return False


if __name__ == '__main__':
    fileDir = '../config/process_para_format.yaml'
    res = read_yaml_data(file_dir=fileDir)
    log.info(f"{type(res)}, {res}")
