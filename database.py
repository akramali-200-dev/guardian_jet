
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends
from typing import Annotated
from app.config import settings

SQLALCHEMY_DATABASE_URL = settings.SYNC_DATABASE_URL
engine = create_engine(SQLALCHEMY_DATABASE_URL, future=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DBSession = Annotated[Session, Depends(get_db_session)]
