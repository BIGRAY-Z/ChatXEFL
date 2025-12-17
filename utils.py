#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pymongo
from datetime import datetime

#Configs
MILVUS_HOST = "10.19.48.181"
MILVUS_PORT = "19530"

# 修改用户名和密码 (如果是第5组，请改为 group5 / Group5)
MILVUS_USER = "cs286_2025_group8"
MILVUS_PASSWORD = "Group8"
MILVUS_DB_NAME = "cs286_2025_group8" 

# ===========================================
# MongoDB Configs (根据图片内容修改)
# ===========================================
MONGO_USER = "cs286_ro"
MONGO_PASS = "cs286"
MONGO_HOST = "10.19.48.181"
MONGO_PORT = "27017"
MONGO_DB_NAME = "chatxfel"

# 按照图片逻辑构建带认证的 URI: mongodb://user:pass@ip:port/dbname
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB_NAME}"
# ===========================================

def get_milvus_connection():
    connection_args = {
        "host": MILVUS_HOST,
        "port": MILVUS_PORT,
    }
    
    if MILVUS_USER and MILVUS_PASSWORD:
        connection_args["user"] = MILVUS_USER
        connection_args["password"] = MILVUS_PASSWORD
        connection_args["db_name"] = MILVUS_DB_NAME 
    return connection_args

def get_mongodb_client(db_name=MONGO_DB_NAME):
    """
    获取 MongoDB 客户端。
    根据图片，连接 URI 中已经包含了认证信息和默认数据库。
    """
    try:
        # 使用配置好的包含用户名/密码的 MONGO_URI
        client = pymongo.MongoClient(MONGO_URI)
        
        # 尝试简单的连接检查
        # client.admin.command('ping')
        
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def format_docs(docs):
    """
    将检索回来的 Document 对象列表拼接成字符串，用于 Prompt Context。
    被 rag.py 调用。
    """
    return "\n\n".join(doc.page_content for doc in docs)

def log_rag(data: dict, use_mongo=False):
    """
    记录问答日志或用户反馈。
    被 chatxfel_app.py 调用。
    """
    # 1. 打印到终端方便实时查看
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] RAG LOG: {data}")

    # 2. 如果开启，则写入 MongoDB
    if use_mongo:
        try:
            # 注意：这里默认使用上面配置的 client (chatxfel 库权限)
            # 如果 logs 存在于不同的数据库，可能需要检查 cs286_ro 用户是否有权限写入 chatxfel_logs
            # 假设日志也存在于 chatxfel 库或者该用户有全局权限：
            
            client = get_mongodb_client()
            if client:
                # 如果你想存入专门的 log 库，可以尝试切换，但前提是用户有权限
                # 这里默认存入配置的 chatxfel 库，或者根据原代码意图存入 chatxfel_logs
                target_db_name = "chatxfel_logs" 
                db = client.get_database(target_db_name)
                col = db.get_collection("access_logs")
                
                # 确保这是一个新字典，以免修改原引用
                log_entry = data.copy()
                if 'timestamp' not in log_entry:
                    log_entry['timestamp'] = datetime.now()
                
                col.insert_one(log_entry)
        except Exception as e:
            print(f"Warning: Failed to log to MongoDB: {e}")