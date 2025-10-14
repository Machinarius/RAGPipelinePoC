from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
from dotenv import load_dotenv
from chroma_client import get_chroma_client
from typing import Any

# Ensure the required libraries are available and load environment variables
try:
    # NOTE: The host must point to the Docker service name if running FastAPI inside Docker,
    # but here we assume the user runs the ingestion/server from the host machine for simplicity.
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    from langchain_ollama import OllamaLLM
    from langchain.prompts import PromptTemplate
    from langchain.chains import RetrievalQA
    from langchain_core.documents import Document
except ImportError:
    # This should be caught by rag_cli.py before server starts, but as a safeguard:
    raise RuntimeError(
        "Missing required LangChain dependencies. Run 'pip install -r requirements.txt'"
    )

# --- Configuration (loaded from .env via rag_cli.py or environment) ---
load_dotenv()
CHROMA_HOST = os.getenv("CHROMA_HOST")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
GENERATION_MODEL = os.getenv("GENERATION_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
COLLECTION_NAME_CHILDREN = os.getenv("COLLECTION_NAME")
COLLECTION_NAME_PARENTS = os.getenv("COLLECTION_NAME_PARENTS")

# --- Environment Variable Validation ---
_VARS = {
    "CHROMA_HOST": CHROMA_HOST,
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "COLLECTION_NAME": COLLECTION_NAME,
    "GENERATION_MODEL": GENERATION_MODEL,
    "EMBEDDING_MODEL": EMBEDDING_MODEL,
    "COLLECTION_NAME_PARENTS": COLLECTION_NAME_PARENTS,
    "COLLECTION_NAME_CHILDREN": COLLECTION_NAME_CHILDREN,
}
_missing_vars = [k for k, v in _VARS.items() if v is None]
if _missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(_missing_vars)}. "
        "Please ensure they are set in your environment or .env.local file."
    )


app = FastAPI(
    title="RAG Prototype API (Python/Ollama)",
    description="Backend for testing RAG pipeline with local models.",
)

# Global variables for RAG components
chroma_client = get_chroma_client()

embed_model = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
llm = OllamaLLM(
    model=GENERATION_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,  # Low temperature for factual, deterministic answers
)

# Initialize ChromaDB connections (Child collection for searching, Parent for retrieval)
chroma_children = Chroma(
    collection_name=COLLECTION_NAME_CHILDREN,
    embedding_function=embed_model,
    client=chroma_client,
)
chroma_parents = Chroma(
    collection_name=COLLECTION_NAME_PARENTS,
    embedding_function=embed_model,
    client=chroma_client,
)


# --- 2. DATA MODELS ---


class Query(BaseModel):
    """Data model for the incoming user query."""

    query: str
    k: int = 10  # Number of search results (child chunks) to retrieve


class SourceDocument(BaseModel):
    """Data model for a single source citation returned to the user."""

    page_content_start: str
    metadata: dict[str, Any]


class Answer(BaseModel):
    """Data model for the full API response."""

    answer: str
    source_documents: list[SourceDocument]


# --- 3. LIFECYCLE AND DEPENDENCIES ---


@app.on_event("startup")
def startup_event():
    """Initializes LLM and Vector Store connections when the server starts."""
    global embed_model, llm, chroma_children, chroma_parents
    print("Initializing RAG components...")

    # Initialize Embedding and LLM models

    print("RAG components initialized successfully.")


# --- 4. RAG HELPER FUNCTIONS ---

# Regex to find Article/Parágrafos citation in the query for filtering
CITATION_REGEX = re.compile(r"(ARTÍCULO\s+\d+|PARÁGRAFO\s+\d+o?\.)", re.IGNORECASE)


def extract_article_filter(query: str) -> tuple[str, dict[str, Any] | None]:
    """
    Extracts the Article or Parágrafo filter from the user query.
    Returns the cleaned query and a filter dictionary for ChromaDB.
    """
    match = CITATION_REGEX.search(query)

    if match:
        full_match = match.group(0)

        # Clean the query by removing the citation text
        cleaned_query = query.replace(full_match, "").strip()

        # Extract type and value (e.g., "ARTÍCULO", "70")
        parts = full_match.split(maxsplit=1)
        if len(parts) == 2:
            citation_type = parts[0].upper()
            citation_id = (
                full_match.upper()
            )  # Use the full cleaned string (e.g., "ARTÍCULO 70")

            # The filter needs the correct key based on what was matched
            filter_key = "ARTÍCULO" if "ARTÍCULO" in citation_type else "PARÁGRAFO"

            metadata_filter = {filter_key: citation_id}
            return cleaned_query, metadata_filter

    return query, None  # No filter found


# --- NUEVA FUNCIÓN DE REESCRITURA DE CONSULTA ---
def rewrite_query_for_synonyms(query: str, llm_instance: OllamaLLM) -> str:
    """Uses the LLM to rewrite the user query using formal terminology."""

    # Este prompt le pide a Mistral que actúe como corrector de términos legales.
    REWRITE_PROMPT = f"""
    Eres un experto en el Código Nacional de Tránsito de Colombia. Tu única tarea es reescribir
    la PREGUNTA del usuario para reemplazar cualquier término coloquial (como 'moto', 'carro', 'bus')
    por el término legal formal (ej., 'motocicleta', 'automóvil', 'buseta').

    No añadas ninguna otra palabra o explicación. Devuelve SOLO la pregunta reescrita.

    PREGUNTA: "{query}"
    PREGUNTA REESCRITA:
    """

    # Limitamos la generación para asegurar que solo devuelva la frase reescrita
    # Usamos un LLM.invoke directo ya que es una llamada de generación simple
    try:
        rewritten_query = llm_instance.invoke(
            REWRITE_PROMPT,
            stop=["\n"],  # Detener la generación después de una línea
        ).strip()

        # Validación: si el LLM devuelve algo muy diferente o vacío, usamos el original.
        if len(rewritten_query) < 5 or rewritten_query.lower().startswith("no puedo"):
            return query

        # El LLM puede ser propenso a empezar con la etiqueta "PREGUNTA REESCRITA:", limpiamos eso.
        if rewritten_query.upper().startswith("PREGUNTA REESCRITA:"):
            rewritten_query = rewritten_query[len("PREGUNTA REESCRITA:") :].strip()

        print(f"Query Reescrita: {rewritten_query}")
        return rewritten_query

    except Exception as e:
        print(
            f"Error durante la reescritura de la consulta: {e}. Usando consulta original."
        )
        return query


def get_parent_document(parent_id: str) -> Document | None:
    """
    Retrieves a single Parent Document (the full Article chunk) by its ID from the parent collection.
    """
    try:
        # Chroma's get method returns a dictionary containing a list of documents
        result = chroma_parents.get(ids=[parent_id], include=["metadatas", "documents"])

        if result and result.get("documents") and result["documents"][0]:
            # Reconstruct the Document object
            return Document(
                page_content=result["documents"][0], metadata=result["metadatas"][0]
            )
        return None
    except Exception as e:
        print(f"Error retrieving parent document {parent_id}: {e}")
        return None


def format_parent_context(documents: list[Document]) -> str:
    """
    Extracts the full Parent Document (Article) text and formats it for the LLM prompt.
    Retrieves the Parent Document for every unique Parent ID found in the search results.
    """
    parent_ids = {
        doc.metadata.get("parent_id")
        for doc in documents
        if doc.metadata.get("parent_id")
    }

    # 1. Fetch Parent Documents
    parent_context_documents: list[Document] = []

    for parent_id in parent_ids:
        if not parent_id:
            continue
        parent_doc = get_parent_document(parent_id)
        if parent_doc:
            parent_context_documents.append(parent_doc)

    if not parent_context_documents:
        # If no parents were found, fallback to the child chunks' content
        # This fallback is unlikely to happen with the current ingestion strategy
        return "\n\n".join([doc.page_content for doc in documents])

    # 2. Format the Parent Documents for the LLM
    context_parts = []

    # Ensure source information is only pulled from the Parent metadata
    for doc in parent_context_documents:
        metadata = doc.metadata

        citation_parts = []
        if metadata.get("ARTÍCULO"):
            citation_parts.append(metadata["ARTÍCULO"])
        if metadata.get("PARÁGRAFO"):
            citation_parts.append(metadata["PARÁGRAFO"])

        # Format the citation string
        citation_str = (
            f"[{' '.join(citation_parts)}]" if citation_parts else "[SECCIÓN SIN CITA]"
        )

        # Add page info for the LLM's grounding instruction
        source_parts = [metadata.get("source")]
        if metadata.get("page") is not None:
            source_parts.append(f"Page {metadata['page']}")
        source_str = (
            f"Fuente: {', '.join(filter(None, source_parts))}"
            if any(source_parts)
            else "Fuente: Desconocida"
        )

        # Combine citation, source, and full content
        formatted_context = f"### {citation_str}\n{source_str}\nContenido Completo del Artículo:\n{doc.page_content}"
        context_parts.append(formatted_context)

    return "\n\n---\n\n".join(context_parts)


# --- 5. RAG EXECUTION ENDPOINT ---


@app.post("/ask", response_model=Answer)
async def ask_rag_agent(query_data: Query):
    """
    Performs the RAG query: Child Search -> Parent Retrieval -> LLM Generation.
    """

    original_query = query_data.query
    k_value = query_data.k

    # --- STEP A: Query Pre-processing (Filter & Rewriting) ---

    # 1. Extract filter for strict article search (e.g., ARTÍCULO 70)
    cleaned_query, metadata_filter = extract_article_filter(original_query)

    # 2. Rewrite query to handle synonyms (runs on the cleaned query if a filter was used)
    # If no filter was used, rewrite runs on the original query.
    rewritten_query = (
        rewrite_query_for_synonyms(cleaned_query, llm)
        if metadata_filter
        else rewrite_query_for_synonyms(original_query, llm)
    )

    print(f"Query original: {original_query} -> Buscando con: {rewritten_query}")

    search_kwargs = {"k": k_value}

    # --- STEP B: Retrieval Passes ---

    # --- PASS 1: Strict Filtered Search (Child Chunks) ---
    search_results: list[Document] = []
    if metadata_filter:
        print(f"PASS 1: Strict search using filter: {metadata_filter}")
        try:
            # We search the Children collection using the rewritten query
            search_results = chroma_children.similarity_search(
                rewritten_query, k=k_value, filter=metadata_filter
            )
        except Exception as e:
            print(f"Error in filtered search: {e}")

    # --- PASS 2: Semantic Fallback Search (If Pass 1 failed or no filter) ---
    if not search_results:
        print("PASS 2: Executing general semantic search (Fallback).")
        try:
            # We search the Children collection without the strict filter
            search_results = chroma_children.similarity_search(
                rewritten_query,  # Use the rewritten query here!
                k=k_value,
            )
        except Exception as e:
            return Answer(
                answer=f"Error fatal en la búsqueda de documentos: {e}",
                source_documents=[],
            )

    if not search_results:
        return Answer(
            answer="No se encontraron documentos relevantes para su consulta.",
            source_documents=[],
        )

    # --- STEP C: Parent Document Retrieval (Context Expansion) ---
    # We retrieve the full article (Parent Chunk) for every unique Child found
    context = format_parent_context(search_results)

    if not context:
        return Answer(
            answer="Se encontraron documentos, pero no se pudo recuperar el contexto completo del artículo.",
            source_documents=[],
        )

    # --- STEP D: LLM Generation ---
    RAG_PROMPT_TEMPLATE = """
    Eres un experto en el Código Nacional de Tránsito de Colombia. Tu tarea es responder la pregunta del usuario
    de manera concisa, precisa y exclusivamente basada en el CONTEXTO que se te proporciona a continuación.

    Instrucciones Clave:
    1. Si la respuesta está fundamentada en un ARTÍCULO, **debes citar el ARTÍCULO o PARÁGRAFO relevante al inicio de tu respuesta** con el formato: [CITA: ARTÍCULO X.].
    2. Si el CONTEXTO no contiene la información necesaria para responder la pregunta, responde: "No puedo responder basándome en la información actual".
    3. No inventes ni alucines números de artículos, datos o información que no esté en el CONTEXTO.
    4. Usa español formal. SIEMPRE responde en español.
    5. Puedes usar formato Markdown para mejorar la legibilidad de tu respuesta.

    CONTEXTO (Artículos completos recuperados):
    ---
    {context}
    ---

    PREGUNTA DEL USUARIO: "{question}"

    RESPUESTA:
    """

    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=original_query)

    try:
        llm_response = llm.invoke(final_prompt)
    except Exception as e:
        return Answer(
            answer=f"Error en la generación de la respuesta por el LLM: {e}",
            source_documents=[],
        )

    # --- STEP E: Final Formatting for API Response ---

    # Extract metadata from the child chunks found during search for debugging/citation source display
    formatted_sources = []
    for doc in search_results:
        metadata = doc.metadata.copy()

        # Ensure page is a string for JSON output
        page_num = metadata.get("page")
        if page_num is not None:
            metadata["page"] = str(page_num)

        # The citation_id helps the front end, let's keep it here for now
        metadata["citation_id"] = (
            f"{metadata.get('ARTÍCULO', '')}_{metadata.get('PARÁGRAFO', '')}"
        )

        formatted_sources.append(
            SourceDocument(
                page_content_start=doc.page_content[:150].strip() + "...",
                metadata=metadata,
            )
        )

    return Answer(answer=llm_response, source_documents=formatted_sources)
