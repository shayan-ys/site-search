import os, time
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from datasets import load_dataset

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY")

# configure client
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(
    cloud="aws", region="us-west-2"
)
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
dataset = load_dataset(
    "json",
    data_files="./site_data.json",
    split="train"
)

index_name = "knife-2-rag"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn"t if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,
        metric="dotproduct",
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# index.delete(delete_all=True)
# time.sleep(1)
# view index stats
# print(index.describe_index_stats())
data = dataset.to_pandas()  # this makes it easier to iterate over the dataset

batch_size = 100
text_field = "content"

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    # get batch of data
    batch = data.iloc[i:i_end]
    # generate unique ids for each URL
    ids = [f'{x["url"]}-{x["chunk-id"]}' for _, x in batch.iterrows()]
    # get text to embed
    texts = [x[text_field] for _, x in batch.iterrows()]
    # embed text
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {
            "text": x[text_field],
            "source": x["url"]
        } for _, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

# print(index.describe_index_stats())
# exit(0)
# initialize the vector store object
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embed_model,
    text_key="text",
)

query = "Sharpening Pricing Western Style Knife"

print(vectorstore.similarity_search(query, k=3))
