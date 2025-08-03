import os
import voyageai
import pymongo
from pymongo import UpdateOne

# Connect to your Atlas cluster
mongo_client = pymongo.MongoClient("MONGO_URI")
db = mongo_client["Training"]

# Define a filter to exclude documents with null or empty 'summary' fields
filter = { 'summary': { '$exists': True, "$nin": [ None, "" ] } }

# Find the collection name: should have a chunks name or something simila
collection = db["listingsAndReviews"]

# Get a subset of documents in the collection 
# Need to change - chunk has metadata as well (e.g. _id, country, title, etc) 
documents = collection.find(filter, {'_id': 1, 'summary': 1})

# Specify your Voyage API key and embedding model
os.environ["VOYAGE_API_KEY"] = "<api-key>"
model = "voyage-3-large"
vo = voyageai.Client()


# Define a function to generate embeddings
def get_embedding(data, input_type = "document"):
  embeddings = vo.embed(
      data, model = model, input_type = input_type
  ).embeddings
  return embeddings[0]

# Generate an embedding
embedding = get_embedding("foo")
print(embedding)



# Generate the list of bulk write operations
operations = []
for doc in documents:
   summary = doc["summary"]
   # Generate embeddings for this document
   embedding = get_embedding(summary)

   # Uncomment the following line to convert to BSON vectors
   # embedding = generate_bson_vector(embedding, BinaryVectorDtype.FLOAT32)

   # Add the update operation to the list
   operations.append(UpdateOne(
      {"_id": doc["_id"]},
      {"$set": {
         "embedding": embedding
      }}
   ))

# Execute the bulk write operation
if operations:
   result = collection.bulk_write(operations)
   updated_doc_count = result.modified_count

print(f"Updated {updated_doc_count} documents.")

# Queries [Embedding]
from pymongo.operations import SearchIndexModel

# Create your index model, then create the search index
search_index_model = SearchIndexModel(
  definition = {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "similarity": "dotProduct",
        "numDimensions": <dimensions>
      }
    ]
  },
  name="vector_index",
  type="vectorSearch"
)
collection.create_search_index(model=search_index_model)



