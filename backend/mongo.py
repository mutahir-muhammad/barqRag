import pymongo
import numpy as np

uri = "mongodb+srv://myAtlasDBUser:mongo123mutahir456@myatlasclusteredu.fjdg2os.mongodb.net/?retryWrites=true&w=majority&appName=myAtlasClusterEDU&connectTimeoutMS=120000"
client = pymongo.MongoClient(uri)
db = client["my_database"]
collection = db["processed_data"]

stored_data = list(collection.find({}, {"_id": 0}))
print(f"🔍 Found {len(stored_data)} records.")

if stored_data:
    print(f"Sample record: {stored_data[0]}")
    print("✅ Embedding shape:", np.array(stored_data[0]['embedding']).shape)
