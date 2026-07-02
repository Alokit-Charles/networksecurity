from pymongo import MongoClient
uri = "mongodb+srv://charlestoppo2002_db_user:<@password>@cluster0.vkteycb.mongodb.net/?appName=Cluster0"
client = MongoClient(uri)
try:
    client.admin.command("ping")
    print("Connected successfully")
    client.close()

except Exception as e:
    raise Exception(
        "The following error occurred: ", e)