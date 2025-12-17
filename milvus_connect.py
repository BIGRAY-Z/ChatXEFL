from pymilvus import connections, utility, Collection

# ================= 配置信息 =================
# 注意：代码连接通常使用 19530 端口，而不是 UI 的 30411 端口
HOST = '10.19.48.181'   # 更新 IP
PORT = '19530'          # 更新端口
USER = 'cs286_2025_group8'  # 更新用户名
PASSWORD = 'Group8'         # 更新密码
DB_NAME = 'cs286_2025_group8'          # 新增数据库名变量

def inspect_milvus_data():
    try:
        # 1. 连接到 Milvus 数据库
        print(f"正在连接到 {HOST}...")
        connections.connect(
            alias="default", 
            host=HOST, 
            port=PORT, 
            user=USER, 
            password=PASSWORD,
            db_name=DB_NAME  # 注意：在 connect 中添加 db_name 参数
        )
        print("✅ 连接成功！")

        # 2. 获取所有集合（Collection）名称
        collections = utility.list_collections()
        if not collections:
            print("⚠️ 该数据库中没有发现任何集合（Collection）。")
            return
        
        print(f"📚 发现集合: {collections}")

        # 3. 遍历集合并查看结构与数据（以第一个集合为例）
        target_collection_name = collections[0] 
        print(f"\n--- 正在检查集合: [{target_collection_name}] ---")
        
        # 加载集合对象
        collection = Collection(target_collection_name)
        
        # 打印 Schema (字段结构)
        print(f"结构 (Schema): {collection.schema}")
        print(f"数据总行数 (Approx): {collection.num_entities}")

        # 4. 加载集合到内存以便查询 (Query 需要 load，但如果是 huge dataset 请谨慎)
        # 注意：只读账号可能有权限限制，如果无法 load，可能只能做 search
        try:
            collection.load()
            print("集合已加载到内存。")
        except Exception as e:
            print(f"⚠️ 加载集合失败 (可能是权限或内存问题): {e}")

        # 5. 查询前 3 条数据 (Query)
        # output_fields=["*"] 表示返回所有字段（包括向量和元数据）
        # limit=3 限制返回条数
        # expr="" 为空表示无过滤条件，但这在 Milvus 旧版可能不支持，通常建议带个简单条件
        # 这里使用 limit 配合 expr (id > 0 或类似，视主键类型而定)
        # 为了通用性，我们先尝试获取主键字段名
        pk_field = collection.primary_field.name
        
        print(f"正在读取 {pk_field} ...")
        
        results = collection.query(
            expr=f"{pk_field} != -1", # 假设主键不等于 -1 的所有数据
            output_fields=["*"],     # 获取所有字段内容
            limit=3                  # 只看3条
        )

        print("\n🔎 数据预览 (前 3 条):")
        for res in results:
            print("-" * 30)
            print(res)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        print("提示: 如果连接超时，请确认 19530 端口是否对你的机器开放。")

if __name__ == "__main__":
    inspect_milvus_data()