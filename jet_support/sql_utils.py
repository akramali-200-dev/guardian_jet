import json
from openai import OpenAI
import logging
from app.jet_support.schema import RouteDecision
from typing import Type, Optional, Tuple, List
from sqlalchemy import inspect
from datetime import date
from app.config import settings

logger = logging.getLogger(__name__)

client = OpenAI()

async def route_query(query: str) -> RouteDecision:
    ROUTE_TOOL = {
        "type": "function",
        "function": {
            "name": "classify_route",
            "description": "Classify the aircraft query safely for SQL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["DATA", "DOCS", "GREETING"],
                        "description": (
                            "DATA = query that can be answered with safe SQL SELECT, filters, count, group by, aggregations "
                            "using the allowed fields. "
                            "DOCS = conceptual/explanatory/spec/maintenance/history, or if destructive ops. "
                            "GREETING = salutations/smalltalk."
                        )
                    },
                    "stock_type": {
                        "type": "string",
                        "enum": ["AVAILABLE", "SOLD", "ANY"],
                        "description": "AVAILABLE = for-sale listings, SOLD = sold aircraft, ANY = not specified."
                    }
                },
                "required": ["type", "stock_type"]
            }
        }
    }

    ROUTER_PROMPT = """
        You are an aircraft assistant query router. Your task is to en-route the user's aircraft query to select best possible options for them.
        Try to understand the query, there is chance that there is TYPO but maximize to try classify it as DATA if possible.
        Always classify the query into one of three categories: DATA, DOCS, GREETING. 
        No other output is allowed.
        
        Allowed SQL fields (with type):
        - listing_id (text)
        - airframe_serial_number (text)
        - manufacturer (text)
        - model (text)
        - model_year (number)
        - airframe_total_time (number)
        - days_on_market (number)
        - listing_broker (text)
        - seller (text)
        - physical_location (text)
        - asking_price (number)
        - sold_price (number)
        - date_listed (datetime)
        - date_sold (datetime)
        - category (text)
        - feature (text)
        - left_engine_hours_since_overhauled
        - right_engine_hours_since_overhauled
        
        Rules:
        
        1. DATA
           - If the query can be expressed using only these allowed fields, classify as DATA. 
           - Valid operations: SELECT, filtering, listing, counting, ordering, grouping, averages, sums, min/max, comparisons.
           - Always prefer DATA if it can be generated from the allowed fields.
           - SAFE only: never classify as DATA if the query attempts DELETE, UPDATE, ALTER, DROP, INSERT, CREATE, TRUNCATE, etc.
        
        2. DOCS
           - If the query mentions concepts outside the allowed fields (maintenance, performance, specs, history, definitions, etc.).
           - Or if the query tries destructive SQL operations.
        
        3. GREETING
           - Greetings, thanks, polite expressions, or small talk.
        
        Stock Type (for DATA only):
        - SOLD if mentions 'sold', 'sold price', 'recently sold', 'comps'.
        - AVAILABLE if mentions 'for sale', 'available', 'asking price', 'active listing'.
        - ANY if unclear.
        
        Examples:
        Q: "show me serial number 5057"
        A: {"type":"DATA","stock_type":"ANY"}
        
        Q: "average asking_price of G550 in 2020"
        A: {"type":"DATA","stock_type":"AVAILABLE"}
        
        Q: "recently sold G550 comps"
        A: {"type":"DATA","stock_type":"SOLD"}
        
        Q: "delete aircraft with id 5057"
        A: {"type":"DOCS","stock_type":"ANY"}
        
        Q: "what is the typical maintenance cost of a G550?"
        A: {"type":"DOCS","stock_type":"ANY"}
        
        Q: "hi there"
        A: {"type":"GREETING","stock_type":"ANY"}
        """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            seed=42,
            tools=[ROUTE_TOOL],
            tool_choice={"type": "function", "function": {"name": "classify_route"}},
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        tcalls = resp.choices[0].message.tool_calls or []
        if not tcalls:
            return RouteDecision(type="DOCS", stock_type="ANY")

        args = json.loads(tcalls[0].function.arguments)
        return RouteDecision(**args)

    except Exception:
        return RouteDecision(type="DOCS", stock_type="ANY")

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_sql",
        "description": (
            "Generate a PostgreSQL SELECT * or SELECT COUNT(*) query for the given parent table. "
            "Use WHERE filters with ILIKE for text. For configuration filters, use EXISTS subqueries "
            "against the related configuration table keyed by the FK."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "The final SQL query text."}
            },
            "required": ["sql"],
            "additionalProperties": False
        }
    }
}

def _schema_of(model_cls: Type) -> str:
    insp = inspect(model_cls)
    cols = [f"{c.name} ({str(c.type)})" for c in insp.columns]
    return ", ".join(sorted(cols))

def _config_relation(model_cls: Type) -> Optional[Tuple[str, str, str, List[str]]]:
    insp = inspect(model_cls)
    parent_pk = None
    for c in insp.columns:
        if c.primary_key:
            parent_pk = c.name
            break

    rel = next((r for r in insp.relationships if r.key == "configurations"), None)
    if not rel:
        return None

    target = rel.mapper.class_
    target_insp = inspect(target)
    config_table = target_insp.local_table.name

    fk_col = None
    for c in target_insp.columns:
        for fk in c.foreign_keys:
            if fk.column.table.name == insp.local_table.name:
                fk_col = c.name
                break
        if fk_col:
            break

    config_cols = [c.name for c in target_insp.columns]

    if not (parent_pk and config_table and fk_col):
        return None

    return (parent_pk, config_table, fk_col, sorted(config_cols))

def _build_prompt(model_cls: Type, user_query: str, current_date: date = None) -> str:
    parent_table = model_cls.__tablename__
    parent_schema = _schema_of(model_cls)

    cfg_info = _config_relation(model_cls)
    if cfg_info:
        parent_pk, cfg_table, cfg_fk, cfg_cols = cfg_info
        cfg_schema = ", ".join([f"{c}" for c in cfg_cols])
        cfg_block = f"""
Related configuration table:
- Name: {cfg_table}
- Foreign key on config: {cfg_fk} → {parent_table}.{parent_pk}
- Columns: {cfg_schema}

Configuration filtering rules:
- When the user filter targets configuration fields (e.g., category/feature/price_adjustment),
  use EXISTS subqueries.
- When the user asked for the models like CJ3, CJ$ ... etc, the should not include the manufacture name in it.

- Feature matching shorthand:
  If the user supplies a feature search term, match it against BOTH columns with **MUST** OR + ILIKE:
  (COALESCE(cfg.feature,'') ILIKE '%term%' OR COALESCE(cfg.category,'') ILIKE '%term%').

- If multiple configuration constraints are present, combine them in the SAME EXISTS when they
  must apply to the SAME config row. If constraints must match DIFFERENT config rows, use
  multiple EXISTS clauses.

- Avoid plain JOINs that duplicate parent rows. Prefer EXISTS. For COUNT(*), EXISTS also avoids
  over-counting.
  
- If the user asks about aircraft features, you MUST normalize query values strictly to the allowed catalog:
{settings.FEATURE_VALUES}

- Treat each JSON object's key as the category/option and its value as the allowed features list.
- Matching is case-insensitive. Synonyms and spelling variants must be mapped to the closest allowed value.
- IMPORTANT: Never invent features, never alter names. Always use values verbatim from the allowed list.
- IMPORTANT: Always use **ILIKE for filter matching**.

- When generating SQL:
  * Do NOT use free-text filters like ILIKE '%aft galley%'.
  * Always constrain filters to the **exact canonical feature string(s)** from the allowed list (e.g., "Galley - AFT" under Interior).
  * If multiple synonyms are provided, they must all be mapped before query generation; only the canonical form should appear in SQL.
  * Example: User says "aft galley" → map to `"Galley - AFT"` in category "Interior", and generate:
    `WHERE cfg.feature = 'Galley - AFT'`
  * If canonical values are not available in your schema, generate a CTE or join to a synonym→canonical mapping table and filter by the canonical value only.

Examples:
-- If the user query contains features regarding interior 
-- e.g: "Which aircraft had both new paint and interior and sold in the last 12 months?"
AND EXISTS (
  SELECT 1
  FROM {cfg_table} AS cfg
  WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (
      (COALESCE(cfg.feature,'') ILIKE '%paint%' OR COALESCE(cfg.category,'') ILIKE '%paint%')
      OR
      (COALESCE(cfg.feature,'') ILIKE '%interior%' OR COALESCE(cfg.category,'') ILIKE '%interior%')
    )
);
*** Try to not use new key word in the feature during the above query***


-- Single feature keyword (matches feature OR category)
AND EXISTS (
  SELECT 1
  FROM {cfg_table} AS cfg
  WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (
      COALESCE(cfg.feature,'')  ILIKE :feature_q
      OR
      COALESCE(cfg.category,'') ILIKE :feature_q
    )
);

-- Multiple feature keywords where ALL must match the SAME cfg row
AND EXISTS (
  SELECT 1
  FROM {cfg_table} AS cfg
  WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (
      (COALESCE(cfg.feature,'')  ILIKE :t0 OR COALESCE(cfg.category,'') ILIKE :t0)
      AND
      (COALESCE(cfg.feature,'')  ILIKE :t1 OR COALESCE(cfg.category,'') ILIKE :t1)
      -- add more terms as needed
    )
);

-- Multiple feature keywords where ANY may match (still a single cfg row)
AND EXISTS (
  SELECT 1
  FROM {cfg_table} AS cfg
  WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (
      (COALESCE(cfg.feature,'')  ILIKE :t0 OR COALESCE(cfg.category,'') ILIKE :t0)
      OR
      (COALESCE(cfg.feature,'')  ILIKE :t1 OR COALESCE(cfg.category,'') ILIKE :t1)
    )
);

-- If terms must match DIFFERENT cfg rows, use multiple EXISTS:
AND EXISTS (
  SELECT 1 FROM {cfg_table} cfg WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (COALESCE(cfg.feature,'') ILIKE :t0 OR COALESCE(cfg.category,'') ILIKE :t0)
)
AND EXISTS (
  SELECT 1 FROM {cfg_table} cfg WHERE cfg.{cfg_fk} = {parent_table}.{parent_pk}
    AND (COALESCE(cfg.feature,'') ILIKE :t1 OR COALESCE(cfg.category,'') ILIKE :t1)
);
"""
    else:
        cfg_block = """
(There is no accessible configuration relation; ignore configuration filters.)
"""

    prompt = f"""You are a PostgreSQL query generation assistant specialized in aircraft listing queries.

DATABASE SCHEMA:
Parent table: {parent_table}
Columns: {parent_schema}

{cfg_block}

CORE QUERY GENERATION RULES:

1. SELECT CLAUSE DETERMINATION:
   a) COUNT queries: When user asks "how many", "count", "number of" → use SELECT COUNT(*)
   b) PERCENTAGE queries: When user asks "percentage", "%", "percent", "share", "ratio", "proportion" (without count)
      → Return single column: ROUND(100.0 * <numerator> / NULLIF(<denominator>, 0), 2) AS percentage
   c) SPECIFIC FIELDS: When user requests particular columns → SELECT [requested_fields]
   d) COMPREHENSIVE: When full analysis needed → SELECT * FROM {parent_table}
   e) DEFAULT FALLBACK: SELECT *
   f) DO NOT INCLUDE ANY ADDITIONAL FILTER WHILE CREATING SQL QUERY, e.g column NOT NULL etc.
   g) If user's query is regarding comparison, use SELECT COUNT(*)

2. FILTERING RULES (MANDATORY):
   - **ALWAYS use ILIKE** for ALL text matching (manufacturer, model, physical_location, etc.)
   - **NEVER use exact = matching** for text fields - only use ILIKE '%value%'
   - **ALL string values**: Always wrap in single quotes and use ILIKE '%value%'
   - **Case-insensitive only**: Never use LIKE - always ILIKE
   - Multiple conditions: Combine with AND
   - Use exact column names from schema only
   - Configuration filters: Use EXISTS subqueries (never JOINs to avoid duplicates)
   {settings.FEATURE_VALUES}

3. DATE AND TIME HANDLING:
   - Current date: {current_date}
   - Date format: 'YYYY-MM-DD'
   - Year filters: Use BETWEEN '2025-01-01' AND '2025-12-31' for "in 2025"
   - Always wrap OR date conditions in parentheses

4. LOCATION FILTERING:
   When filtering by physical_location:
   - COUNTRIES: Include full name + ISO codes + abbreviations
     Example: "United States" → "United States", "US", "USA", "U.S."
   - STATES/REGIONS: Include full name + postal abbreviation
     Example: "Florida" → "Florida", "FL"
   - State codes: Use regex with word boundaries: column ~* '\\mFL\\M' (single backslash)
   - CITIES: Include name + common variants only
   - Never mix location levels in same query
   - Wrap location alternatives in OR within parentheses

5. TIME WINDOW COMPARISONS:
   For "last N vs previous N" queries:
   - Current window: [current_date - interval 'N months', current_date)
   - Previous window: [current_date - interval '2N months', current_date - interval 'N months')
   
   COUNT requests:
   SELECT
     COUNT(*) FILTER (WHERE <filters> AND date_col >= current_date - interval 'N months') AS current_period,
     COUNT(*) FILTER (WHERE <filters> AND date_col >= current_date - interval '2N months' 
                              AND date_col < current_date - interval 'N months') AS previous_period
   FROM {parent_table};
   
   LIST requests: Include period column with CASE statement

6. AVERAGE CALCULATIONS:
   For "average per month/year" metrics:
   - Calculate total period: EXTRACT(YEAR FROM AGE()) * 12 + EXTRACT(MONTH FROM AGE())
   - Use GREATEST(1, period) to prevent division by zero
   - Round to 2 decimal places

7. NUMERIC RANGES:
   Use appropriate operators: >=, <=, BETWEEN for price/numeric ranges

CONSISTENCY REQUIREMENTS:
- Always use the exact same logic for identical inputs
- Maintain deterministic output for same user_query
- Follow schema column names precisely
- Apply filters in consistent order

MANDATORY ILIKE EXAMPLES:
- Model filtering: manufacturer ILIKE '%gulfstream%' AND model ILIKE '%G550%'
- Location filtering: physical_location ILIKE '%United States%' OR physical_location ILIKE '%US%'
- Serial number: airframe_serial_number ILIKE '%5057%'
- Broker: listing_broker ILIKE '%jetcraft%'
- NEVER use exact = for text: manufacturer = 'Gulfstream' (WRONG)
- ALWAYS use ILIKE: manufacturer ILIKE '%Gulfstream%' (CORRECT)

STATIC RULES (NO EXCEPTIONS):
1. ALL text columns MUST use ILIKE '%value%' - never exact matching
2. ALL string comparisons MUST be case-insensitive with ILIKE
3. Location queries MUST include variations: (physical_location ILIKE '%United States%' OR physical_location ILIKE '%US%' OR physical_location ILIKE '%USA%')
4. Model queries MUST be fuzzy: model ILIKE '%G550%' not model = 'G550'
5. ALWAYS generate identical SQL for identical inputs - be deterministic
6. Use consistent filter order: manufacturer, model, physical_location, features, dates, prices

DETERMINISTIC QUERY PATTERNS:
- "GV aircraft": SELECT * FROM {parent_table} WHERE model ILIKE '%GV%';
- "Falcon 2000" : SELECT * FROM {parent_table} WHERE model ILIKE '%Falcon 2000%';
- "G550 in US": SELECT * FROM {parent_table} WHERE model ILIKE '%G550%' AND (physical_location ILIKE '%United States%' OR physical_location ILIKE '%US%' OR physical_location ILIKE '%USA%');
- "average days on market": SELECT AVG(days_on_market) FROM {parent_table} WHERE [filters];
- "compare X vs Y": Use CASE statements or separate queries with UNION
- "refurbished / overhauled engines": SELECT * FROM aircraft_listings WHERE (COALESCE(left_engine_hours_since_overhauled, 0) > 0 OR COALESCE(right_engine_hours_since_overhauled, 0) > 0);
MANDATORY QUERY STRUCTURE:
1. SELECT clause (*, specific fields, aggregations)
2. FROM {parent_table} 
3. WHERE clause with ILIKE filters in consistent order
4. EXISTS subqueries for configurations (if needed)
5. GROUP BY, ORDER BY, LIMIT (if requested)

USER REQUEST: "{user_query}"

Generate the exact same PostgreSQL query every time for this input. Use ILIKE for ALL text matching. Be completely deterministic."""
    return prompt

async def nl_to_sql_with_schema_and_configs(model_cls: Type, user_query: str, current_date: date = None) -> str:
    prompt = _build_prompt(model_cls, user_query, current_date)
    parent_table = model_cls.__tablename__

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            seed=42,
            tools=[SQL_TOOL],
            tool_choice={"type": "function", "function": {"name": "generate_sql"}},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query},
            ],
        )
        tool_calls = resp.choices[0].message.tool_calls or []
        if not tool_calls:
            return f"SELECT * FROM {parent_table};"
        args = json.loads(tool_calls[0].function.arguments)
        sql = args.get("sql") or f"SELECT * FROM {parent_table};"
        return sql
    except Exception as e:
        logger.error(f"SQL generation failed: {e}")
        return f"SELECT * FROM {parent_table};"

async def update_query(query, history_context):
    prompt = f"""
    You are an assistant for an aircraft marketplace platform.

The platform contains two types of aircraft listings:
- Available listings: aircraft currently for sale.
- Sold listings: aircraft that have already been sold.

You are given:
1) The user's current query.
2) A recent conversation history, which may include prior enhanced queries and/or "Assistant SQL" lines.

Your task is to ENHANCE the user's query intelligently - preserving their core intent while making strategic improvements based on context and established patterns.
Never ask clarifying questions. Always output one concrete enhanced query in plain natural language.

DATE AND TIME HANDLING:
   - Current date: {date.today()}
   - Date format: 'YYYY-MM-DD'
   - Year filters: Use BETWEEN '2025-01-01' AND '2025-12-31' for "in 2025"
   - Always wrap OR date conditions in parentheses

----------------------------------------------------------------
### Core Behaviors (STRICTLY ENFORCE)
- **Preserve user intent.** Keep the user's core question and comparison structure intact.
- **Smart context usage.** Apply history context and established patterns to improve query completeness.
- **Contextual enhancement.** Add relevant details from history when they improve query precision.
- **Intelligent completion.** Fill gaps and improve clarity while maintaining the user's voice.
- **Natural language only.** Never generate SQL - describe intent in plain language.

----------------------------------------------------------------
### Enhancement Rules (Apply When Beneficial)
1. **Preserve aircraft model names** exactly as user writes them (GV, G5, G550 Falcon 2000, etc. - do NOT auto-correct)
2. **Add scope clarification** when context suggests specific focus (available/sold based on history)
3. **Inherit relevant filters** from recent history that apply to current query
4. **Expand time references** consistently (recently → specific date ranges if established)
5. **Standardize location references** for better matching (US → United States, FL → Florida)
6. **Preserve comparison structure** while ensuring both sides are complete
7. **Add result preferences** from history (sort order, limits) when continuing conversation
8. **Recent Items** if query is regarding recent/recently, include 6 Month of period

----------------------------------------------------------------
### Context Inheritance Patterns
- **Model continuity**: If discussing G550s, continue with G550 unless user changes model
- **Scope continuity**: If focus was on sold aircraft, maintain unless explicitly changed
- **Filter continuity**: Carry forward location, feature, time constraints unless contradicted  
- **Preference continuity**: Maintain sort order, result limits from recent queries

----------------------------------------------------------------
### Location Expansion Rules (Apply Only When Completing Missing Context)
- **Countries**: Full name + ISO codes + abbreviations
  "United States" → "United States", "US", "USA", "U.S."
- **States/Regions**: Full name + postal abbreviation  
  "Florida" → "Florida", "FL"
  "California" → "California", "CA"
- **Cities**: Main name + common variants
  "Dubai" → "Dubai", "Dubayy"

----------------------------------------------------------------
### Sticky Filters (Carry Forward Until Changed)
**Always maintain these across related queries:**
- **Result limit** (top 5, limit 10, first 20, etc.)
- **Sort preference** (cheapest, most expensive, newest, oldest)
- **Feature requirements** (internet connectivity, crew rest, galley)
- **Location constraints** (country, state, city with full expansions)
- **Time windows** (recently = maintain same date interpretation)
- **Aircraft category** (jets, turboprops, helicopters)
- **Scope preference** (available vs sold vs both)

----------------------------------------------------------------
### History Context Usage (Smart)
**Use history to:**
- Complete obvious gaps in aircraft model names
- Inherit compatible filters from recent queries
- Maintain established scope preferences (available/sold)
- Apply consistent time window definitions
- Carry forward sort orders and limits when continuing conversations

----------------------------------------------------------------
### Output Format
Return ONLY the enhanced query in plain natural language.
Do NOT answer the query or provide explanations.
If reusing context from history, explicitly state what was reused in parentheses at the end.

----------------------------------------------------------------
### Examples

**Model name preservation:**
User: "What's the average number of days on market for a US registered, aft galley GV vs international aft galley?"
Enhanced: "What's the average number of days on market for a US registered, aft galley GV vs international aft galley?"

**Context inheritance:**
History: "Show available Cessna aircraft, sorted by lowest asking price, limit 5"
User: "What about G550?"
Enhanced: "Retrieve available G550 aircraft, sorted by lowest asking price, limit 5 (reuse prior sort and limit)"

**Filter accumulation:**
History: "Available jets with internet connectivity, limit 10"
User: "In Florida"  
Enhanced: "Retrieve available jets in Florida that feature internet connectivity, limit 10 (reuse prior jets category, internet connectivity, and limit 10)"

**Scope switching:**
History: "Available Dassault aircraft, cheapest first, top 5"
User: "Show me sold ones"
Enhanced: "Retrieve sold Dassault aircraft, sorted by lowest asking price, limit 5 (reuse prior Dassault focus, sort, and limit)"

**Recent/Recently:**
User: "List of aircraft recently sold"
Enhanced: "Retrieve sold aircraft from the last 6 months (apply 'recent' = last 6 months)"

----------------------------------------------------------------
<HISTORY>
{history_context}
</HISTORY>
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        seed=42,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
    )
    text = resp.choices[0].message.content.strip()
    return text