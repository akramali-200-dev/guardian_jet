from typing import List, Dict, Any, Optional
from collections import deque
import threading
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class UserHistoryManager:
    def __init__(self, max_messages_per_user: int = 3, default_context_messages: int = 3):
        self.max_messages = max_messages_per_user
        self.default_context_messages = default_context_messages
        self._user_histories: Dict[str, deque] = {}
        self._user_sql_cache: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def add_message(self, user_id: str, role: str, content: str, sql_query: Optional[str] = None):
        with self._lock:
            if user_id not in self._user_histories:
                self._user_histories[user_id] = deque(maxlen=self.max_messages)
                self._user_sql_cache[user_id] = {}

            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

            if sql_query:
                message["sql_query"] = sql_query
                cache_key = content[:100] if len(content) > 100 else content
                self._user_sql_cache[user_id][cache_key] = sql_query

            self._user_histories[user_id].append(message)

    def get_history(self, user_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            if user_id not in self._user_histories:
                return []
            return list(self._user_histories[user_id])

    def get_recent_sql_queries(self, user_id: str) -> Dict[str, str]:
        with self._lock:
            return self._user_sql_cache.get(user_id, {})

    def clear_history(self, user_id: str):
        with self._lock:
            if user_id in self._user_histories:
                self._user_histories[user_id].clear()
            if user_id in self._user_sql_cache:
                self._user_sql_cache[user_id].clear()

    def get_conversation_context(
        self,
        user_id: str,
        *,
        include_sql: bool = True,
        include_sql_cache_tail: bool = False,
        context_messages: Optional[int] = None,
        truncate_at: Optional[int] = None
    ) -> str:
        history = self.get_history(user_id)
        if not history:
            return ""

        k = context_messages or self.default_context_messages
        recent = history[-k:]

        lines: List[str] = []
        for msg in recent:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            if truncate_at is not None and truncate_at > 0:
                content = content[:truncate_at]

            lines.append(f"{role_label}: {content}")
            if include_sql and "sql_query" in msg and msg["sql_query"]:
                lines.append(f"{role_label} SQL: {msg['sql_query']}")

        if include_sql_cache_tail:
            cache = self.get_recent_sql_queries(user_id)
            if cache:
                lines.append("\nRecent SQL cache (snippet → SQL):")
                for snippet, sql in cache.items():
                    snip = snippet if truncate_at is None else snippet[:truncate_at]
                    lines.append(f"- [{snip}] -> {sql}")

        return "\n".join(lines)


history_manager = UserHistoryManager()


def add_user_message(user_id: str, message: str):
    history_manager.add_message(user_id, "user", message)


def add_assistant_response(user_id: str, response: str, sql_query: Optional[str] = None):
    history_manager.add_message(user_id, "assistant", response, sql_query)


def get_user_chat_history(user_id: str) -> List[Dict[str, Any]]:
    return history_manager.get_history(user_id)


def get_conversation_context(
    user_id: str,
    *,
    include_sql: bool = True,
    include_sql_cache_tail: bool = False,
    context_messages: Optional[int] = None,
    truncate_at: Optional[int] = None
) -> str:
    return history_manager.get_conversation_context(
        user_id,
        include_sql=include_sql,
        include_sql_cache_tail=include_sql_cache_tail,
        context_messages=context_messages,
        truncate_at=truncate_at,
    )


def get_user_sql_cache(user_id: str) -> Dict[str, str]:
    return history_manager.get_recent_sql_queries(user_id)
