from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from app.jet_support.router import router


app = FastAPI(title="Guardian Jet AI")
origins = ['*']


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
)

# serve the chat UI at /chat/
BASE_DIR = Path(__file__).resolve().parent
CHAT_DIR = BASE_DIR / "static" / "chat"   # <- absolute path
app.mount("/chat", StaticFiles(directory=str(CHAT_DIR), html=True), name="chat")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(router)
