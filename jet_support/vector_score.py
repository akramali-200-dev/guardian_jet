from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from typing import List
import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

_pinecone_client = None
_embeddings = None


def init_connections():
    global _pinecone_client, _embeddings

    if _pinecone_client is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Missing required environment variables")

        _pinecone_client = Pinecone(api_key=pinecone_api_key)
        _embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        logger.info("Connections initialized")


def search_available_namespace(query: str, filters, k:int = 50) -> List[Document]:
    vector_store = PineconeVectorStore(
        index=_pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME")),
        embedding=_embeddings,
        namespace="__default__",
        text_key="embedding_summary"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    documents = retriever.get_relevant_documents(query)
    return documents


def search_sold_namespace(query: str, filters, k: int = 50) -> List[Document]:
    vector_store = PineconeVectorStore(
        index=_pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME")),
        embedding=_embeddings,
        namespace="aircraft-sold",
        text_key="embedding_summary"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    documents = retriever.get_relevant_documents(query)
    return documents


def search_both_namespaces(query: str, filters, k: int = 25) -> List[Document]:
    try:
        final_documents = []
        documents = search_sold_namespace(query, filters, k)
        documents2 = search_available_namespace(query, filters, k)
        final_documents.extend(documents)
        final_documents.extend(documents2)

        logger.info(f"Found {len(documents)} total results")
        return final_documents

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []


def get_content_from_metadata(metadata: dict, status: str) -> str:
    manufacturer = metadata.get('manufacturer', 'Unknown')
    model = metadata.get('model', 'Unknown')
    year = metadata.get('model_year', 'Unknown')
    serial = metadata.get('airframe_serial_number', 'N/A')
    location = metadata.get('physical_location', 'N/A')

    asking_price = metadata.get('asking_price', 'N/A')
    sold_price = metadata.get('sold_price', 'N/A')
    estimated_value = metadata.get('guardianjets_estimated_value', 'N/A')

    total_time = metadata.get('airframe_total_time', 'N/A')
    left_engine_hours = metadata.get('left_engine_hours_since_new', 'N/A')
    right_engine_hours = metadata.get('right_engine_hours_since_new', 'N/A')

    days_on_market = metadata.get('days_on_market', 'N/A')
    date_listed = metadata.get('date_listed', 'N/A')
    date_sold = metadata.get('date_sold', 'N/A')

    listing_broker = metadata.get('listing_broker', 'N/A')
    seller = metadata.get('seller', 'N/A')
    options = metadata.get('options_pairs', 'N/A')

    content_parts = [
        f"{status}: {manufacturer} {model} ({year})",
        f"Serial: {serial}",
        f"Location: {location}"
        f"Features/Options: {options if options != 'N/A' else 'None'}"
    ]

    if status == "Available":
        if asking_price and asking_price != 'N/A':
            content_parts.append(f"Asking Price: ${format_price(asking_price)}")
        if estimated_value and estimated_value != 'N/A':
            content_parts.append(f"Estimated Value: ${format_price(estimated_value)}")
    else:
        if asking_price and asking_price != 'N/A':
            content_parts.append(f"Original Price: ${format_price(asking_price)}")
        if sold_price and sold_price != 'N/A':
            content_parts.append(f"Sold Price: ${format_price(sold_price)}")
        if date_sold and date_sold != 'N/A':
            content_parts.append(f"Date Sold: {date_sold[:10]}")

    if total_time and total_time != 'N/A':
        content_parts.append(f"Total Time: {total_time} hours")

    if left_engine_hours and left_engine_hours != 'N/A':
        content_parts.append(f"Engine Hours: L:{left_engine_hours} R:{right_engine_hours}")

    if days_on_market and days_on_market != 'N/A':
        content_parts.append(f"Days on Market: {days_on_market}")

    if date_listed and date_listed != 'N/A':
        content_parts.append(f"Listed: {date_listed[:10]}")

    if listing_broker and listing_broker != 'N/A':
        content_parts.append(f"Broker: {listing_broker}")

    if seller and seller != 'N/A':
        content_parts.append(f"Seller: {seller}")

    return " | ".join(content_parts)


def format_price(price_str: str) -> str:
    try:
        if price_str and price_str != 'N/A' and price_str != '0':
            return f"{float(price_str):,.0f}"
        return "N/A"
    except:
        return str(price_str) if price_str else "N/A"
