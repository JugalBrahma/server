from astrapy import DataAPIClient
import os
from dotenv import load_dotenv

load_dotenv()

client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))

db = client.get_database_by_api_endpoint(
    os.getenv("ASTRA_DB_API_ENDPOINT"),
    keyspace=os.getenv("ASTRA_DB_KEYSPACE")
)

# Collections (tables)
users_col = db.get_collection("users")
folders_col = db.get_collection("folders")

print("Connected to AstraDB successfully")