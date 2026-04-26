"""
mcp_server.py — Cooking Knowledge Base MCP Server (Step 9)

Exposes the VP RAG Eval pipeline as an MCP server.
Any Claude instance (Claude Code, Claude.ai, any app) can call these tools
without knowing anything about ChromaDB, BGE-large, or our pipeline internals.

The difference from Step 6e (Agentic RAG):
  Step 6e:  retrieve() tool defined INSIDE agentic_pipeline.py
            → only that script can use it
  Step 9:   retrieve() tool exposed via MCP server
            → ANY Claude instance can call it — universal plug

Tools exposed:
  search_cooking_knowledge(query, top_k)  — retrieve chunks from ChromaDB
  ask_cooking_question(question)          — full RAG: retrieve + generate answer

Resources exposed:
  cooking://topics  — list of all 20 cooking topics in the knowledge base

How to connect to Claude Code:
  Add to ~/.claude/claude_desktop_config.json or .claude/settings.json:
  {
    "mcpServers": {
      "cooking-rag": {
        "command": "python3",
        "args": ["/Users/vipin/Downloads/vp-rag-eval/mcp_server.py"],
        "env": {
          "ANTHROPIC_API_KEY": "<your key>"
        }
      }
    }
  }

Usage (run directly to test):
    python3 mcp_server.py

"""

import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ── Config ────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent
CHROMA_DIR      = ROOT / "chroma_db"
ENV_FILE        = ROOT / ".env"

load_dotenv(ENV_FILE)

MODEL            = "claude-opus-4-5"
MAX_TOKENS       = 512
COLLECTION_NAME  = "cooking_techniques"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K    = 3

GROUNDED_PROMPT = """\
Answer the question using ONLY the information provided below.
Do not use outside knowledge. If the answer is not in the provided \
context, say "I cannot answer this from the provided information."

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:"""

COOKING_TOPICS = [
    "Blanching", "Sautéing", "Emulsification", "Braising", "Caramelization",
    "Maillard Reaction", "Brining", "Tempering Chocolate", "Reduction",
    "Poaching", "Deglazing", "Marinating", "Rendering Fat", "Blooming Spices",
    "Resting Meat", "Folding", "Seasoning Cast Iron", "Making a Roux",
    "Steaming", "Knife Skills",
]


# ── ChromaDB connection ───────────────────────────────────────────────────────
def get_collection():
    client   = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve_chunks(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search ChromaDB for relevant chunks."""
    collection = get_collection()
    results    = collection.query(query_texts=[query], n_results=top_k)
    chunks     = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "doc_id":   meta["doc_id"],
            "topic":    meta["topic"],
            "section":  meta["section"],
            "content":  doc,
            "distance": round(dist, 4),
        })
    return chunks


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate grounded answer from retrieved chunks."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['topic']} — {chunk['section']}]\n{chunk['content']}"
        )
    context = "\n\n".join(context_parts)
    prompt  = GROUNDED_PROMPT.format(context=context, question=question)

    client  = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


# ── MCP Server ────────────────────────────────────────────────────────────────
server = Server("cooking-rag")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare what tools this MCP server exposes."""
    return [
        types.Tool(
            name="search_cooking_knowledge",
            description=(
                "Search the cooking techniques knowledge base for relevant information. "
                "Returns the most relevant chunks from 20 cooking technique documents "
                "(blanching, braising, emulsification, Maillard reaction, etc.). "
                "Use this to find specific cooking technique content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — write as document search terms, not a question. "
                                       "Example: 'fond deglazing pan sauce' not 'How do I make pan sauce?'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of chunks to return (1-5, default 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="ask_cooking_question",
            description=(
                "Ask a question about cooking techniques and get a grounded answer "
                "based ONLY on the knowledge base content. "
                "Handles the full RAG pipeline: retrieve relevant chunks → generate answer. "
                "Use this for direct cooking questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "A cooking technique question to answer from the knowledge base.",
                    },
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from Claude."""

    if name == "search_cooking_knowledge":
        query  = arguments["query"]
        top_k  = min(int(arguments.get("top_k", DEFAULT_TOP_K)), 5)
        chunks = retrieve_chunks(query, top_k)

        # Format results for Claude
        parts = [f"Found {len(chunks)} chunks for query: '{query}'\n"]
        for i, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{i}] {chunk['topic']} — {chunk['section']}\n"
                f"    doc_id  : {chunk['doc_id']}\n"
                f"    distance: {chunk['distance']} (lower = more relevant)\n"
                f"    content : {chunk['content'][:300]}..."
            )
        return [types.TextContent(type="text", text="\n\n".join(parts))]

    elif name == "ask_cooking_question":
        question = arguments["question"]
        chunks   = retrieve_chunks(question, DEFAULT_TOP_K)
        answer   = generate_answer(question, chunks)

        result = (
            f"Question: {question}\n\n"
            f"Sources used ({len(chunks)} chunks):\n"
            + "\n".join(f"  - {c['topic']} — {c['section']}" for c in chunks)
            + f"\n\nAnswer:\n{answer}"
        )
        return [types.TextContent(type="text", text=result)]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Expose knowledge base metadata as a resource."""
    return [
        types.Resource(
            uri="cooking://topics",
            name="Cooking Topics",
            description="List of all 20 cooking technique topics in the knowledge base",
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Return resource content."""
    if str(uri) == "cooking://topics":
        return json.dumps({
            "total_topics": len(COOKING_TOPICS),
            "topics": COOKING_TOPICS,
            "description": "VP RAG Eval — 20 cooking technique documents",
        }, indent=2)
    raise ValueError(f"Unknown resource: {uri}")


# ── Entry point ───────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
