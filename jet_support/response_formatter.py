import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ResponseFormatter:

    @staticmethod
    def format_greeting_response(query: str) -> Dict[str, Any]:
        message = (
            "I am here to provide you with relevant information about available aircraft or past sales. "
            "If you have specific questions or need details about certain aircraft, feel free to ask!"
        )
        return {
            "success": True,
            "query": query,
            "answer": message,
            "records": "",
            "total_results": 0
        }
    
    @staticmethod
    def format_data_response(query: str, answer: str, records: str, total_results: int) -> Dict[str, Any]:
        return {
            "success": True,
            "query": query,
            "answer": answer,
            "records": records,
            "total_results": total_results
        }
    
    @staticmethod
    def format_search_response(result) -> Dict[str, Any]:
        return {
            "success": result.success,
            "query": result.query,
            "answer": result.answer,
            "records": result.records,
            "total_results": result.total_results
        }
    
    @staticmethod
    def format_error_response(query: str, error_message: str) -> Dict[str, Any]:
        return {
            "success": False,
            "query": query,
            "answer": error_message,
            "records": "",
            "total_results": 0
        }