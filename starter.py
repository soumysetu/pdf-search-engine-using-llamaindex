from llama_index import download_loader

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import OpenAI
from llama_index.query_engine import CitationQueryEngine
from pathlib import Path
from llama_index import Document

service_context = ServiceContext.from_defaults(chunk_size=1000)
PDFReader = download_loader("PDFReader")

loader = PDFReader()
documents = loader.load_data(file=Path('./Amazon 10K.pdf'))
doc_text = "\n\n".join([d.get_content() for d in documents])
docs = [Document(text=doc_text)]
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
llm = OpenAI(model="gpt-3")

query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    citation_chunk_size=512,
)

response = query_engine.query(
    "what is the net sells for aws and online stores in 2022"
)

print(response)
print(len(response.source_nodes))