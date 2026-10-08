from fastapi import APIRouter, HTTPException, status
from app.jet_support.schema import SearchRequest
import app.jet_support.service as service
import logging
from app.database import DBSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/aircraft", tags=["Aircraft Search"])

@router.post("/search")
async def search_aircraft(db_session: DBSession, request: SearchRequest):
    try:
        return await service.search_aircraft_service(db_session, request)
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.on_event("startup")
async def startup_event():
    try:
        service.init_services()
        logger.info("Aircraft search service started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise