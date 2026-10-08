from app.database import DBSession
from app.jet_support.models import SearchHistory
from sqlalchemy import text

def save_user_history(
        db: DBSession,
        query: str,
        ai_response: str,
        user_id: str,
) -> SearchHistory:

    history_entry = SearchHistory(
        query=query,
        ai_response=ai_response,
        user_id=user_id,
    )

    db.add(history_entry)
    db.commit()
    db.refresh(history_entry)
    return history_entry

async def process_query(
        db: DBSession,
        sql_query: str
):
    # is_count = sql_query.strip().upper().startswith("SELECT COUNT")
    # is_percentage = "as percentage" in sql_query.lower()
    result = db.execute(text(sql_query))

    # if is_count and not is_percentage:
    #     return {"count": result.scalar()}
    # elif is_percentage and not is_count:
    #     return {"percentage": result.scalar()}
    # elif is_count and is_percentage:
    #     rows = result.mappings().all()
    #     return dict(rows[0]) if rows else {"count": 0, "percentage": 0}
    # else:
    rows = result.mappings().all()
    return {"rows": [dict(row) for row in rows], "count": len(rows)}
