import os
import ssl
import certifi
import json
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# --- FIX FOR SSL CERTIFICATE ERROR ---
os.environ['SSL_CERT_FILE'] = certifi.where()


# --- 1. SET UP YOUR API KEY ---
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# --- 2. HELPER FUNCTIONS & MODEL FACTORY ---

def clean_json_string(raw_string: str) -> str:
    """Cleans the raw output from an LLM, removing markdown code blocks."""
    if "```json" in raw_string:
        raw_string = raw_string.split("```json")[1].split("```")[0]
    start = raw_string.find('{')
    end = raw_string.rfind('}')
    if start != -1 and end != -1:
        return raw_string[start:end+1].strip()
    return raw_string.strip()

def get_llm(model_name: str = "gpt-4o"):
    """LLM Factory: Instantiates and returns an LLM client."""
    print(f"Initializing model: {model_name}")
    if "gpt" in model_name:
        return ChatOpenAI(model_name=model_name, temperature=0.7)
    else:
        raise NotImplementedError(f"Model provider for '{model_name}' is not configured.")

# --- 3. LOAD KNOWLEDGE BASE AND STATIC CONTEXT ---
print("Loading knowledge base and static context files...")
try:
    # Load the global schema from its root path
    with open('./knowledge_base/wizard101_quest_schema.json', 'r', encoding='utf-8') as f:
        quest_schema_str = f.read()

    # Load all Wizard City JSON indexes from their new paths
    with open('./knowledge_base/wizard_city/indexes_json/wc_monsters.json', 'r', encoding='utf-8') as f:
        monster_index_str = f.read()
    with open('./knowledge_base/wizard_city/indexes_json/wc_npcs.json', 'r', encoding='utf-8') as f:
        npc_index_str = f.read()
    with open('./knowledge_base/wizard_city/indexes_json/wc_locations.json', 'r', encoding='utf-8') as f:
        location_index_str = f.read()

    # Combine all indexes into a single, well-formatted string for the prompt
    full_index_str = f"--- MONSTER INDEX ---\n{monster_index_str}\n\n--- NPC INDEX ---\n{npc_index_str}\n\n--- LOCATION INDEX ---\n{location_index_str}"

    # Load the main story arc summary from its new path
    with open('./knowledge_base/wizard_city/lore_md/story_arcs/wizard_city_story_arc.md', 'r', encoding='utf-8') as f:
        arc_summary_str = f.read()

    print("Successfully loaded schema, all JSON indexes, and story arc.")

except FileNotFoundError as e:
    print(f"FATAL: A required context file was not found. Error: {e}")
    exit()

# --- 4. INITIALIZE VECTOR STORE (RAG) ---
print("Initializing vector store for RAG...")

# Point to the parent directory containing all your markdown lore
lore_path = "./knowledge_base/wizard_city/lore_md"
documents = []
try:
    for dirpath, _, filenames in os.walk(lore_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_path = os.path.join(dirpath, filename)
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(f"No .md files found in {lore_path} or its subdirectories.")
    else:
        print(f"Successfully loaded {len(documents)} documents for the vector store.")

except FileNotFoundError as e:
    print(f"FATAL: Could not load documents for vector store. Error: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = text_splitter.split_documents(documents)
vector_store = Chroma.from_documents(documents=split_docs, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()
print("Vector store is ready.")

# --- 5. ADVANCED PROMPT TEMPLATE ---
questline_prompt_template = """
You are an expert video game quest designer for Wizard101. Your task is to generate the *next* quest in a sequence, ensuring it is logical, lore-accurate, and creative.
Follow these instructions precisely:
1.  **Reason Step-by-Step:** First, use the `<thinking>` block to outline your design process.
2.  **Use the Game Data Indexes:** You MUST select all NPCs, locations, and enemy targets exclusively from the provided Game Data Indexes.
3.  **Adhere to the Schema:** Your final output MUST be a single, valid JSON object that strictly follows the provided JSON Schema.

---
**STATIC CONTEXT**
**Overarching Story Arc:**
{arc_summary}

**Game Data Indexes (Required for all objectives):**
{all_indexes}

**JSON Schema to Follow:**
{schema}
---
**DYNAMIC CONTEXT**
**Retrieved Game Lore (for reference):**
{context}
**Questline State (Summary of what has happened so far):**
{questline_state}
**User Request:**
{question}
---
**YOUR RESPONSE**
**1. Chain of Thought Reasoning:**
<thinking>
(Your reasoning here: 1. Analyze the previous quest from the Questline State. 2. Select a logical NPC and location for the next step using the Game Data Indexes. 3. Define a lore-appropriate objective using an enemy from the Game Data Indexes. 4. Propose a suitable title and dialogue. 5. Determine appropriate rewards.)
</thinking>
**2. Valid Raw JSON Output (Do NOT include the <thinking> block):**
"""
QUESTLINE_PROMPT = PromptTemplate(
    template=questline_prompt_template,
    input_variables=["context", "question", "questline_state"],
    partial_variables={
        "schema": quest_schema_str,
        "all_indexes": full_index_str,
        "arc_summary": arc_summary_str
    }
)

# --- 6. CORE GENERATION FUNCTIONS ---

def generate_quest_in_sequence(query: str, questline_state: dict, llm) -> (dict, str):
    """Generates the next quest in a sequence using the advanced RAG pipeline."""
    print(f"\n> Generating ADVANCED quest for: '{query}'")
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "questline_state": itemgetter("questline_state"),
        }
        | QUESTLINE_PROMPT
        | llm
        | StrOutputParser()
    )

    try:
        response = rag_chain.invoke({
            "question": query,
            "questline_state": json.dumps(questline_state, indent=2)
        })
        raw_output = response
        thinking, json_str = "Error: Could not parse.", raw_output
        if "</thinking>" in raw_output:
            parts = raw_output.split("</thinking>", 1)
            thinking_content = parts[0]
            if "<thinking>" in thinking_content:
                thinking = thinking_content.split("<thinking>", 1)[1].strip()
            json_str = parts[1]
        quest_json = json.loads(clean_json_string(json_str))
        return quest_json, thinking
    except Exception as e:
        print(f"!! Advanced Generation Error: {e}")
        return {"error": "Failed to generate valid quest JSON.", "details": str(e)}, "Generation failed."

def generate_baseline_quest(query: str, llm) -> (dict, str):
    """Generates a quest using only the LLM and the schema (control group)."""
    print(f"\n> Generating BASELINE quest for: '{query}'")
    baseline_prompt_template = """
    You are a video game quest designer for the game Wizard101.
    Your output MUST be a single, valid JSON object that strictly adheres to the provided schema.
    Do not include any text, explanations, or markdown formatting. Just provide the raw JSON.

    **JSON Schema to Follow:**
    ```json
    {schema}
    ```
    **User Request:**
    {user_query}
    **Valid Raw JSON Output:**
    """
    BASELINE_PROMPT = PromptTemplate(
        template=baseline_prompt_template, input_variables=["user_query"],
        partial_variables={"schema": quest_schema_str}
    )
    try:
        baseline_chain = BASELINE_PROMPT | llm
        response = baseline_chain.invoke({"user_query": query})
        cleaned_output = clean_json_string(response.content)
        quest_json = json.loads(cleaned_output)
        return quest_json, "N/A"
    except Exception as e:
        print(f"!! Baseline Generation Error: {e}")
        return {"error": "Failed to generate valid baseline quest.", "details": str(e)}, "Generation failed."

# --- 7. MAIN EXECUTION BLOCK (for simple testing) ---
if __name__ == '__main__':
    pass
