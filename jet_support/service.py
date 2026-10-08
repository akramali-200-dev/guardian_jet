import logging
from app.database import DBSession
from app.jet_support.schema import SearchRequest
import app.jet_support.ai_service as ai_service
import app.jet_support.vector_score as vector_score
from app.jet_support.history_service import HistoryService
from app.jet_support.intent_handlers import IntentHandlers

logger = logging.getLogger(__name__)

class AircraftService:
    @staticmethod
    def init_services():
        try:
            vector_score.init_connections()
            ai_service.init_ai()
            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise
    
    @staticmethod
    async def search_aircraft(db_session: DBSession, request: SearchRequest):
        try:
            enhanced_query = await HistoryService.enhance_query(request.user_id, request.query)
            
            intent = await HistoryService.classify_intent(enhanced_query)
            
            HistoryService.add_user_query(request.user_id, request.query)
            
            if intent.type == "GREETING":
                return IntentHandlers.handle_greeting(request.user_id, request.query)
            elif intent.type == "DATA":
                return await IntentHandlers.handle_data_query(db_session, request, enhanced_query, intent)
            else:
                return await IntentHandlers.handle_vector_search(db_session, request, enhanced_query, intent)
                
        except Exception as e:
            logger.error(f"Search service error: {str(e)}")
            raise

def init_services():
    return AircraftService.init_services()

async def search_aircraft_service(db_session: DBSession, request: SearchRequest):
    return await AircraftService.search_aircraft(db_session, request)