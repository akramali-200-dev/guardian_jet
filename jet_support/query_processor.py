import re
import json
import logging
from app.jet_support.schema import SearchResponse
import app.jet_support.vector_score as vector_score
import app.jet_support.repository as repository
import app.jet_support.ai_service as ai_service
from app.database import DBSession

logger = logging.getLogger(__name__)

FILTER_FIELDS = {
    "listing_id": "text",
    "airframe_serial_number": "text",
    "manufacturer": "text",
    "model": "text",
    "model_year": "number",
    "airframe_total_time": "number",
    "left_engine_hours_since_new": "number",
    "right_engine_hours_since_new": "number",
    "left_engine_hours_since_overhauled": "number",
    "right_engine_hours_since_overhauled": "number",
    "date_hours_last_updated": "date",
    "date_listed": "date",
    "days_on_market": "number",
    "listing_broker": "text",
    "seller": "text",
    "physical_location": "text",
    "asking_price": "text",
    "airframe_total_time_value_adjustment": "number",
    "engine_maintenance_program_value_adjustment": "number",
    "auxiliary_power_unit_value_adjustment": "number",
    "damage_devaluation": "number",
    "interior_value_adjustment": "number",
    "new_interior_value_adjustment": "number",
    "paint_value_adjustment": "number",
    "new_paint_value_adjustment": "number",
    "aiframe_total_time_engine_adjustment": "number",
    "total_options_and_adjustments_to_value": "number",
    "base_value_for_model_year": "number",
    "adjusted_value": "number",
    "guardianjet_estimated_value": "number",
    "features": "list",
    "options_pairs": "list",
    "aircraft_configuration_and_options": "text",
    "embedding_summary": "text",
    "has_internet_by_category": "boolean",
    "namespace": "text",
    "parent_id": "text",
}

def build_filter_from_query(query: str) -> dict | None:
    from openai import OpenAI
    
    client = OpenAI()
    schema = json.dumps(FILTER_FIELDS, indent=2)

    prompt = f"""
        You are a query-to-filter translator for Pinecone vector search.
        Given a natural language query and a schema, return a JSON object with filters.

        Schema:
        {schema}

        Rules:
        - For text fields: use {{"field_name": {{"$eq": "value"}}}} or {{"field_name": {{"$in": ["v1","v2"]}}}}
        - For number fields: {{"field_name": {{"$lt": 100}}}}, {{"field_name": {{"$gte": 50}}}}
        - Combine multiple with {{"$and": [filter1, filter2]}} or {{"$or": [filter1, filter2]}}
        - Field name is the KEY, operator object is the VALUE
        - Only output valid JSON (no text, no markdown).

        Examples:
        - Single filter: {{"status": {{"$eq": "active"}}}}
        - Multiple filters: {{"$and": [{{"status": {{"$eq": "active"}}}}, {{"price": {{"$lt": 100}}}}]}}

        Query: "{query}"
        """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return None

async def search_aircraft(db_session: DBSession, query: str, user_id: str, stock_type: str) -> SearchResponse:
    try:
        filters = None
        if stock_type == "SOLD":
            documents = vector_score.search_sold_namespace(query, filters)
        elif stock_type == "AVAILABLE":
            documents = vector_score.search_available_namespace(query, filters)
        else:
            documents = vector_score.search_both_namespaces(query, filters)

        answer = ai_service.generate_answer_json(query, documents)

        ans_match = re.search(r"<ANSWER>\s*(.*?)\s*</ANSWER>", answer, flags=re.DOTALL)
        rec_match = re.search(r"<RECORDS>\s*(.*?)\s*</RECORDS>", answer, flags=re.DOTALL)

        answer = ans_match.group(1).strip()
        records = rec_match.group(1).strip()

        repository.save_user_history(db_session, query, answer, user_id)

        return SearchResponse(
            success=True,
            query=query,
            answer=answer,
            records=records,
            total_results=len(documents)
        )

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return SearchResponse(
            success=False,
            query=query,
            answer=f"Error while searching for '{query}'.",
            records="",
            total_results=0
        )

async def process_query(db_session: DBSession, sql_query: str):
    return await repository.process_query(db_session, sql_query)