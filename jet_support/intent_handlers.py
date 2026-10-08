import logging
from datetime import date
from app.database import DBSession
from app.jet_support.schema import SearchRequest
from app.jet_support.models import AircraftListing, SoldAircraftListing
import app.jet_support.repository as repository
import app.jet_support.ai_service as ai_service
import app.jet_support.sql_utils as sql_utils
import app.jet_support.query_processor as query_processor
from app.jet_support.history_service import HistoryService
from app.jet_support.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class IntentHandlers:
    @staticmethod
    def handle_greeting(user_id: str, query: str):
        response = ResponseFormatter.format_greeting_response(query)
        HistoryService.add_response(user_id, response["answer"])
        return response
    
    @staticmethod
    async def handle_data_query(
            db_session: DBSession,
            request: SearchRequest,
            updated_query: str,
            intent,
            fall_back: bool = False,
    ):
        try:
            table = SoldAircraftListing if intent.stock_type == "SOLD" else AircraftListing
            
            sql_query = await sql_utils.nl_to_sql_with_schema_and_configs(
                table, updated_query, date.today()
            )
            
            HistoryService.add_response(request.user_id, updated_query, sql_query)
            
            db_response = await repository.process_query(db_session, sql_query)
            
            if not db_response.get("count") and (not fall_back):
                return await IntentHandlers._fallback_to_vector_search(
                    db_session, request, updated_query, intent
                )
            
            ai_answer, records = ai_service.process_prompt(request.query, updated_query, db_response)
            return ResponseFormatter.format_data_response(
                request.query, ai_answer, records, db_response.get("count", 0)
            )
            
        except Exception as e:
            logger.error(f"Data query handler error: {str(e)}")
            raise
    
    @staticmethod
    async def handle_vector_search(db_session: DBSession, request: SearchRequest, updated_query: str, intent):
        try:
            result = await query_processor.search_aircraft(
                db_session, updated_query, request.user_id, intent.stock_type
            )
            
            HistoryService.add_search_response(request.user_id, updated_query, result.answer)
            
            if not result.success:
                raise Exception(result.answer)

            if result.answer == "None":
                result = await IntentHandlers.handle_data_query(
                    db_session, request, updated_query, intent, fall_back=True
                )
            
            return ResponseFormatter.format_search_response(result)
            
        except Exception as e:
            logger.error(f"Vector search handler error: {str(e)}")
            raise
    
    @staticmethod
    async def _fallback_to_vector_search(db_session: DBSession, request: SearchRequest, updated_query: str, intent):
        result = await query_processor.search_aircraft(
            db_session, updated_query, request.user_id, intent.stock_type
        )
        
        HistoryService.add_search_response(request.user_id, updated_query, result.answer)
        return ResponseFormatter.format_search_response(result)