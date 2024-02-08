from langchain_community.document_loaders import JSONLoader

FILE_PATH='./site_data.json'

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["source"] = record.get("url")
    metadata["hyperlinks"] = record.get("hyperlinks")

    return metadata

def load_docs():
    loader = JSONLoader(
        file_path=FILE_PATH,
        jq_schema='.[]',
        content_key="content",
        text_content=False,
        metadata_func=metadata_func
    )

    return loader.load()
