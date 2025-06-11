import logging
import os

def create_logger(file_name='log_files'):
    """
    创建日志记录器，同时输出到文件和控制台。
    自动避免重复添加 handler。
    """
    log_path = file_name + '.log'

    # 如果 logger 已存在 handler，先清空，避免重复日志
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建目录（若路径中含目录）
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    # 添加文件 handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # # 添加控制台输出 handler
    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)
    return logger
