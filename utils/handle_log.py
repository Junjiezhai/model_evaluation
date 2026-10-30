# -*- coding: utf-8 -*-
"""
# @File    : handle_log.py
# @Date    : 2024-08-02
# @Author  : jeremyzhai
# @Description : 项目日志统一入口（loguru）
#
# 输出策略：
#   - 控制台：默认 stderr  sink（保持原有可见性）
#   - 文件  ：logs/ 目录下按日创建 txt 执行日志（同名同日追加），
#             每次导入本模块时自动清理超过 LOG_RETENTION_DAYS 天的旧日志
"""

import logging
import os
import re
from datetime import datetime, timedelta

from loguru import logger

# 项目根目录（本文件位于 utils/ 下）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(_BASE_DIR, 'logs')
LOG_RETENTION_DAYS = 30          # 执行日志保留天数（到期自动清理）
LOG_SUFFIX = '.txt'
_DATE_PATTERN = re.compile(r'(\d{4})-(\d{2})-(\d{2})')  # 从日志文件名提取日期


class PropogateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


def _cleanup_expired_logs(log_dir: str, retention_days: int) -> int:
    """清理超过保留期的执行日志文件（文件名含 YYYY-MM-DD 的 txt），返回清理数量"""
    cutoff = datetime.now() - timedelta(days=retention_days)
    removed = 0
    for name in os.listdir(log_dir):
        if not name.endswith(LOG_SUFFIX):
            continue
        m = _DATE_PATTERN.search(name)
        if not m:
            continue
        try:
            file_date = datetime(*map(int, m.groups()))
        except ValueError:
            continue
        if file_date < cutoff:
            os.remove(os.path.join(log_dir, name))
            removed += 1
    return removed


def _setup_file_sink() -> None:
    """初始化文件 sink：按日创建 txt 执行日志，并自动清理过期日志"""
    os.makedirs(LOG_DIR, exist_ok=True)
    removed = _cleanup_expired_logs(LOG_DIR, LOG_RETENTION_DAYS)
    today = datetime.now().strftime('%Y-%m-%d')
    logger.add(
        os.path.join(LOG_DIR, f'{today}{LOG_SUFFIX}'),
        encoding='utf-8',
        enqueue=True,  # 多线程（asyncio.to_thread / 批量回退）下安全写入
        format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}',
    )
    if removed:
        logger.info(f'[日志] 已自动清理 {removed} 个 {LOG_RETENTION_DAYS} 天前的执行日志')


logger.add(PropogateHandler(), format="{time:YYYY-MM-DD at HH:mm:ss} | {message}")  # , encoding="utf-8"
_setup_file_sink()

log = logger

if __name__ == '__main__':
    log.debug("这是一条debug日志")
    log.info("第二条")
