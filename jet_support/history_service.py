import logging
from app.jet_support.user_history import (
    add_user_message,
    add_assistant_response,
    get_conversation_context,
)
import app.jet_support.sql_utils as sql_utils

logger = logging.getLogger(__name__)

class HistoryService:
    @staticmethod
    def add_user_query(user_id: str, query: str):
        add_user_message(user_id, query)
    
    @staticmethod
    def add_response(user_id: str, response: str, sql_query: str = None):
        if sql_query:
            add_assistant_response(user_id, response, sql_query)
        else:
            add_assistant_response(user_id, response)
    
    @staticmethod
    def add_search_response(user_id: str, updated_query: str, answer: str):
        concat_resp = f"""
        User Query: {updated_query},
        Assistant Response: {answer}
    """
        add_assistant_response(user_id, concat_resp)
    
    @staticmethod
    async def enhance_query(user_id: str, original_query: str) -> str:
        history = get_conversation_context(user_id)
        return await sql_utils.update_query(original_query, history)
    
    @staticmethod
    async def classify_intent(query: str):
        return await sql_utils.route_query(query)