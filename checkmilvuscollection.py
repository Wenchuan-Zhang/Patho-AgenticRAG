from pymilvus import connections, utility

try:
    # 1. 连接 Milvus
    # 默认地址是 localhost, 端口是 19530 (脚本中映射的)
    connections.connect(alias="default", host='localhost', port='19530')
    print("✅ 成功连接到 Milvus 服务。")

    # 2. 列出所有的 Collection
    collection_names = utility.list_collections()
    
    if not collection_names:
        print("📭 Milvus 中目前没有任何 Collection。")
    else:
        print("🔍 找到了以下 Collections:")
        for name in collection_names:
            print(f"  - {name}")

except Exception as e:
    print(f"❌ 连接 Milvus 失败: {e}")
    print("请确保 Milvus Docker 容器正在运行且处于 'healthy' 状态。")

finally:
    # 断开连接
    try:
        connections.disconnect("default")
        print("\n👋 已断开连接。")
    except:
        pass