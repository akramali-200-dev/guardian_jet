import json
import logging
from decimal import Decimal
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
import tiktoken

logger = logging.getLogger(__name__)

_ai_client = None
client = OpenAI()

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

def init_ai():
    global _ai_client

    if not _ai_client:
        _ai_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            seed=42
        )

        logger.info("AI client initialized")

def custom_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def generate_answer(query: str, documents: list) -> str:
    try:
        if not documents:
            return f"I couldn't find any aircraft matching '{query}' in our database. Try searching for manufacturer names like 'Gulfstream', 'Cessna', or 'Boeing' for better results."

        POP_COLS = {"options_pairs", "embedding_summary", "features", "parent_id", "namespace", "parent_id"}

        context = ""
        for i, doc in enumerate(documents[:25], 1):
            doc_dict = dict(getattr(doc, "metadata", {})) if hasattr(doc, "metadata") else dict(doc)

            for key in POP_COLS:
                doc_dict.pop(key, None)
            _value = doc_dict.pop("aircraft_configuration_and_options")
            if type(_value) is not str:
                _value = json.dumps(_value)

            options = json.loads(_value) or []
            for item in options:
                doc_dict[item.get("Feature")] = item.get("Price Adjustment")

            context += f"{i}. {json.dumps(doc_dict, default=custom_serializer, indent=2)}\n"

        prompt = ChatPromptTemplate.from_template(
            """You are an aircraft market analyst. Answer strictly from the data below; do not invent or infer details not present.

            Client Question: "{query}"

            Search Results:
            {context}

            Core rules:
            - Answer the client's question directly and conversationally & should **not include bullets** points, **it should be conversational**.
            - If no exact matches, suggest alternatives from the results.
            - If the question is about currently available / in-stock aircraft, use ONLY items marked as "Available Aircraft". Do NOT mention sold aircraft in that case.
            - If the question is about past sales, you may use only items marked "Sold Aircraft".
            - Never suggest contacting brokerages, setting alerts, or searching elsewhere.
            - Keep responses concise, data-focused and relevant to the query.
            - If no relevant aircraft are found, say so directly.
            
            Intent handling (VERY IMPORTANT):
            - First, determine intent from the Client Question:
              • COMPARISON: the question asks to compare models (e.g., contains "compare", "vs", "versus", or clearly names two distinct models).
              • LISTINGS: the question asks for items/details of listings.
            - If intent is COMPARISON:
              • Provide ONLY a concise narrative comparison of the two models using the Search Results.
              • Do NOT include any tables or listing summaries.
              • Do NOT include phrases like "Here's a summary of the available aircraft listings…" or any similar lead-ins.
            - If intent is LISTINGS:
              • Follow the table rule below. LIST ALL THE COLUMNS FROM THE CONTEXT PROVIDED TO YOU, I WANT EACH VALUE IN THE TABLE, NORMALIZE THE COLUMNS NAMES
              e.g adjusted_value will be Adjusted Value
              - DO NOT UPDATE THE COLUMNS NAMES IT SHOULD BE EXACT AS PASSED IN THE CONTEXT.
            
            Table rule (LISTINGS intent only):
              - If the data in {context} contains **multiple comparable aircraft listings**, Output a clean Markdown table.
              - Use every key present in {context} as a table column header.
              - No extra spaces for alignment and no trailing pipes.
              - Price should be "$#,###,###" or "Make Offer".
              - Total Hours should be an integer (no commas or decimals).
              - Otherwise, respond in conversational paragraph form.
            """
        )

        chain = prompt | _ai_client
        print(f"Token for context {count_tokens(context)}")
        response = chain.invoke({
            "query": query,
            "context": context,
        })

        return response.content.strip()

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error fetching response for '{query}'."

def generate_answer_json(query: str, documents: list) -> str:
    try:
        if not documents:
            return (
                f"I couldn't find any aircraft matching '{query}' in our database. Try searching for "
                f"manufacturer names like 'Gulfstream', 'Cessna', or 'Boeing' for better results."
            )

        POP_COLS = {"options_pairs", "embedding_summary", "features", "parent_id", "namespace", "parent_id"}

        context = ""
        for i, doc in enumerate(documents[:25], 1):
            doc_dict = dict(getattr(doc, "metadata", {})) if hasattr(doc, "metadata") else dict(doc)

            for key in POP_COLS:
                doc_dict.pop(key, None)
            _value = doc_dict.pop("aircraft_configuration_and_options")
            if type(_value) is not str:
                _value = json.dumps(_value)

            options = json.loads(_value) or []
            for item in options:
                doc_dict[item.get("Feature")] = item.get("Price Adjustment")

            context += f"{i}. {json.dumps(doc_dict, default=custom_serializer, indent=2)}\n"

        prompt = ChatPromptTemplate.from_template(
            """You are an aircraft market analyst. Answer strictly from the data below; do not invent or infer details not present.

            Client Question: "{query}"

            Search Results:
            {context}

            Relevance Gate (apply BEFORE answering):
            - Treat an item in {{context}} as RELEVANT only if ALL are true:
              1) It matches the user intent (COMPARISON or LISTINGS) and the question scope.
              2) It matches any explicit model/variant/year/status constraints stated in the question.
              3) It contains the specific entities or attributes referenced in the question as plain text in the item.
            - Discard all items that fail any check.

            Core rules:
            - Answer the client's question directly and conversationally (no bullet points).
            - If the question is about currently available / in-stock aircraft, use ONLY items marked "Available Aircraft". Do NOT mention sold aircraft then.
            - If the question is about past sales, use ONLY items marked "Sold Aircraft".
            - Never suggest contacting brokerages, setting alerts, or searching elsewhere.
            - Keep responses concise, data-focused, and strictly grounded in the RELEVANT items.
            - If no relevant aircraft are found, output exactly: None

            Intent handling:
            - Determine intent from the Client Question:
              • COMPARISON: user compares models (e.g., “compare”, “vs/versus”, or clearly two distinct models).
              • LISTINGS: user asks for items/details of listings.
            - If COMPARISON:
              • Provide only a concise narrative comparison using RELEVANT items from {{context}}.
              • Do NOT include disclaimers or lead-ins about listings.
            - If LISTINGS:
              • Provide a concise narrative describing what the records show, based on RELEVANT items only.

            JSON Records (always output after the narrative if not None):
            - Immediately after the narrative, output a machine-readable JSON payload of the RELEVANT items, preserving keys EXACTLY as they appear in {{context}}. Do not rename or normalize keys.
            - The payload must only include fields present in {{context}}; do not add new fields.
            - Price values must be either "$#,###,###" or "Make Offer" (if a price field exists in {{context}}).
            - If a "Total Hours" field exists, render it as an integer (no commas/decimals).
            - If there is exactly one relevant item, JSON may be a single object; otherwise output a JSON array of objects.

            Strict output format (no extra commentary, headers, or code fences):
            <ANSWER>
            {{Write the conversational answer here (one or two short paragraphs).}}
            </ANSWER>
            <RECORDS>
            {{Write only the JSON here. If no relevant items exist, you must have output exactly "None" above and leave nothing else.}}
            </RECORDS>
            """
        )

        chain = prompt | _ai_client
        print(f"Token for context {count_tokens(context)}")
        response = chain.invoke({
            "query": query,
            "context": context,
        })

        return response.content.strip()

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error fetching response for '{query}'."

def generate_answer_v1(query: str, documents: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template(
            """You are an aircraft market analyst. Answer strictly from the data below; do not invent or infer details not present.

            Client Question: "{query}"

            Search Results:
            {context}

            Core rules:
            - Answer the client's question directly and conversationally & should **not include bullets** points, **it should be conversational**.
            - Never suggest contacting brokerages, setting alerts, or searching elsewhere.
            - Keep responses concise, data-focused and relevant to the query.
            - Be helpful and provide actionable information
            - If data is provided to you, always use it to answer the question as its the fact full information which need to be used ALWAYS.
            - The context that will be provided to you is from SQL database queries, so it will always be as per user response
            - IMPORTANT: If no relevant information is found, but data is available in the context, USE IT!
            
            Additional formatting rule:
              - If the data in {context} contains **multiple comparable aircraft listings** (based on multiple columns), output them as a clean Markdown table with headers: as per the data available.
              - No extra spaces for alignment and no trailing pipes.
              - Price should be "$#,###,###" or "Make Offer".
              - Total Hours should be an integer (no commas or decimals).
              - Otherwise, respond in conversational paragraph form.
              - When displaying multiple aircraft, create a markdown table that includes ALL available data columns from the search results
              - Use EVERY column present in the data - do not omit any fields
              - Column headers should match the actual field names from the data & the column header name should be normalize to look readable with spaces.
              - Format values appropriately:
                  * Prices: "$#,###,###" format or "Make Offer"
                  * Hours: integers (no decimals)
                  * Dates: YYYY-MM-DD format
                  * Decimals: round to 2 decimal places where appropriate
              - No extra spaces for alignment and no trailing pipes
              - If only one aircraft, respond conversationally in paragraph form and include all relevant details
              
            IMPORTANT: Include every single data field available in the results. Do not selectively choose columns - show the complete dataset in table format.
            Make sure it should show all the results, no row should be neglected.
            """
        )

        chain = prompt | _ai_client
        response = chain.invoke({
            "query": query,
            "context": documents
        })

        return response.content.strip()

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"Error fetching response for '{query}'."

def summarize_table_response(query: str, documents: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template(
            """You are an aircraft market analyst.  
            Your job is to provide a short acknowledgement or overview based strictly on the data below.  
            Do not generate tables or bullet points. Do not invent or infer details not present.

            Client Question: "{query}"

            Search Results (from SQL query):
            {context}

            Core rules:
            - Always respond in 1–3 sentences maximum, conversational and data-focused.
            - Responses must be concise, neutral, and professional, phrased as an analyst briefing a client.
            - State figures plainly but not bluntly; use formal verbs such as “stood at,” “were recorded at,” “accounted for,” “represented.”
            - Do not use casual fillers (e.g., “shows,” “reports,” “indicates,” “meaning,” “so”).
            - Do not calculate or describe differences unless explicitly stated in the data.
            - Always frame outputs as clear analytic statements, not raw numbers or CSV-style listings.
            - Do not include the exact count of items when presenting lists, but if the query is analytic or statistical and the data explicitly provides numbers, include those figures in full.
            - Never output a table, bullets, or lists.
            - Never suggest contacting brokerages, setting alerts, or searching elsewhere.
            - Be concise, helpful, and actionable.
            - If multiple rows are present, acknowledge the number of matches and optionally highlight key features.
            - If exactly one row is present, refer to it naturally and include the most relevant details.
            - If the data appears empty, still acknowledge the provided context (don’t claim no data unless truly empty).
            - Absolutely never fabricate information not in the context.
            """
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": prompt.format(query=query, context=documents)
            }]
        )
        return response.choices[0].message.content.strip()

        # chain = prompt | _ai_client
        # response = chain.invoke({"query": query, "context": documents})
        # return response.content.strip()

    except Exception as e:
        logger.error(f"Error summarizing table: {str(e)}")
        return f"Error summarizing response for '{query}'."

def process_prompt(query, updated_query, response):
    answer = ""
    records = ""
    if (
            ("count" in response or "percentage" in response)
            and "rows" not in response
    ):
        parts = []
        if "count" in response:
            parts.append(f"Count: {response['count']}")
        if "percentage" in response:
            parts.append(f"Percentage: {response['percentage']}")

        context = f"Total results found against query: {query} " + ", ".join(parts)
        answer = generate_answer_v1(query, context)

    elif "rows" in response:
        rows = [
            {k: (v if v not in (None, "", []) else None) for k, v in row.items()}
            for row in response["rows"][:50]
        ]
        exclude = ["id"]
        filtered_rows = []

        for row in rows:
            clean_row = {col: val for col, val in row.items() if col not in set(exclude)}

            options = clean_row.pop("options", {}) or {}
            for feature, value in options.items():
                clean_row[feature] = value
            filtered_rows.append(clean_row)
        if all(x == {} for x in filtered_rows):
            filtered_rows = rows

        records = json.dumps(filtered_rows, default=custom_serializer, indent=2)
        # answer = generate_answer_v1(query, records)
        answer = summarize_table_response(updated_query, records)
        records = "" if (len(filtered_rows) < 2 and filtered_rows[0].__len__() < 4) else records

    return answer, records