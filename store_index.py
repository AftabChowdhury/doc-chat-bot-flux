from dotenv import load_dotenv
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

extracted_data = load_pdf_file(data='pdf-data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Get the API key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)  # Get the API key from environment variables

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
