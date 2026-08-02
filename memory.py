"""
FRIDAY Memory System
====================

This is FRIDAY's long-term memory.

Think of it like a very smart notebook:
- Every important thing the user says gets written down
- Each note is converted into numbers (called an embedding)
- These numbers capture the MEANING of what was said
- Later, when the user asks something, we search the notebook
  for notes with similar meaning
- Those notes get included in the AI's context

This is called RAG (Retrieval-Augmented Generation).
Real AI assistants like ChatGPT Plus, Siri, and Google Assistant
all use similar systems internally.

Technical stack:
- ChromaDB: the vector database (stores the number-lists)
- SentenceTransformers: converts text to number-lists (embeddings)
- The embeddings are created locally — no extra API calls needed
"""

import chromadb
import json
import time
import hashlib
import logging
import os
from typing import Optional
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

logger = logging.getLogger("friday-memory")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Where ChromaDB stores its files on disk
# This means memories persist even after the server restarts
# CHANGE (portability): configurable via CHROMA_PATH env var so it can point
# at a Railway Volume mount in production; defaults to a local folder for dev.
# COMPAT: the default path keeps the legacy "aria_" prefix so existing local
# databases keep working — it's a disk path, never shown to users or the AI.
CHROMA_PATH = os.getenv("CHROMA_PATH", "./aria_memory_db")

# The embedding model we use to convert text → numbers
# "all-MiniLM-L6-v2" is small (80MB), fast, and surprisingly good
# It downloads automatically the first time it runs
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Maximum number of memories to retrieve per query
MAX_RETRIEVED = 5

# Maximum number of total memories per user (to control storage)
MAX_MEMORIES_PER_USER = 500

# Minimum similarity score to consider a memory relevant (0.0 to 1.0)
# 0.7 = "must be 70% similar in meaning"
SIMILARITY_THRESHOLD = 0.70


# ─── MEMORY TYPES ─────────────────────────────────────────────────────────────
# Not all memories are equal. We categorize them so we can
# retrieve the right type when needed.

class MemoryType:
    FACT       = "fact"        # "I live in Addis Ababa"
    PREFERENCE = "preference"  # "I prefer short answers"
    EVENT      = "event"       # "I have a meeting tomorrow at 3pm"
    SKILL      = "skill"       # "I know Python programming"
    EMOTION    = "emotion"     # "I was upset about my exam"
    GENERAL    = "general"     # Anything else


# ─── FRIDAY MEMORY MANAGER ──────────────────────────────────────────────────────

class FridayMemoryManager:
    """
    The main memory system for FRIDAY.
    
    This class handles:
    1. Storing new memories (with embeddings)
    2. Retrieving relevant memories (via semantic search)
    3. Organizing memories by user and type
    4. Cleaning up old/irrelevant memories
    """

    def __init__(self):
        logger.info("Initializing FRIDAY Memory Manager...")

        # Initialize ChromaDB — the vector database
        # PersistentClient means data is saved to disk between restarts
        # Think of it like a file-based database, but specialized for vectors
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Get or create the "memories" collection
        # A collection in ChromaDB is like a table in a regular database
        # We use cosine similarity — the standard for text embeddings
        # Cosine similarity measures the angle between two vectors
        # Two identical vectors → similarity = 1.0
        # Two completely different vectors → similarity ≈ 0.0
        self.collection = self.chroma_client.get_or_create_collection(
            # COMPAT: collection name intentionally keeps the legacy "aria_"
            # prefix — renaming it would orphan every memory already stored
            # on disk. It is a storage key, not an identity string; the AI
            # never sees it.
            name="aria_memories",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize the embedding model
        # This runs locally on your server — no API needed
        # First run downloads ~80MB model file, then it's cached
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Memory Manager ready.")

    # ─── GENERATE EMBEDDING ───────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """
        Convert text into a vector (list of numbers).
        
        Example:
          "I live in Addis Ababa" 
          → [0.82, -0.31, 0.54, 0.12, ... 384 numbers total]
        
        Similar sentences produce similar vectors.
        That's what makes semantic search work.
        """
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    # ─── STORE A MEMORY ───────────────────────────────────────────────────────

    def store_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = MemoryType.GENERAL,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Save a piece of information to FRIDAY's long-term memory.
        
        Args:
            user_id: Who this memory belongs to (e.g., "user_123")
            content: What to remember (e.g., "User lives in Addis Ababa")
            memory_type: Category of this memory
            metadata: Extra info to store alongside the memory
        
        Returns:
            The ID of the stored memory
        """
        # Generate a unique ID for this memory
        # We use a hash of content + timestamp so duplicate content
        # doesn't create duplicate memories
        memory_id = hashlib.md5(
            f"{user_id}:{content}:{time.time()}".encode()
        ).hexdigest()

        # Build the metadata to store alongside the vector
        full_metadata = {
            "user_id": user_id,
            "memory_type": memory_type,
            "timestamp": time.time(),
            "date_str": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "content_preview": content[:100],  # For debugging
        }
        if metadata:
            full_metadata.update(metadata)

        # Convert the content to an embedding vector
        embedding = self.embed(content)

        # Store in ChromaDB
        # documents = the actual text
        # embeddings = the vector representation
        # metadatas = extra info about this memory
        # ids = unique identifier
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[full_metadata],
            ids=[memory_id]
        )

        logger.info(f"Stored memory [{memory_type}] for {user_id}: {content[:60]}...")

        # Clean up old memories if we're over the limit
        self._enforce_memory_limit(user_id)

        return memory_id

    # ─── RETRIEVE RELEVANT MEMORIES ───────────────────────────────────────────

    def retrieve_relevant(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[str] = None,
        max_results: int = MAX_RETRIEVED
    ) -> list[dict]:
        """
        Search for memories relevant to a query.
        
        This is the core of RAG (Retrieval-Augmented Generation).
        
        Process:
        1. Convert the query to an embedding vector
        2. Search ChromaDB for stored vectors with similar direction
        3. Return the most similar memories
        
        The key insight: we're not searching for EXACT words.
        We're searching for SIMILAR MEANING.
        
        "What city do I live in?" will find "User lives in Addis Ababa"
        even though none of those words overlap.
        """
        if self.collection.count() == 0:
            return []

        # Convert the query to a vector
        query_embedding = self.embed(query)

        # Build filter — only search this user's memories
        where_filter = {"user_id": user_id}
        if memory_type:
            where_filter["memory_type"] = memory_type

        try:
            # The actual semantic search
            # ChromaDB finds the stored vectors CLOSEST to our query vector
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

        # Process results
        memories = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Convert distance to similarity score
            # ChromaDB's cosine distance: 0 = identical, 2 = opposite
            # We convert to: 1.0 = identical, 0.0 = opposite
            similarity = 1 - (dist / 2)

            # Only include memories above our threshold
            if similarity >= SIMILARITY_THRESHOLD:
                memories.append({
                    "content": doc,
                    "type": meta.get("memory_type", "general"),
                    "date": meta.get("date_str", ""),
                    "similarity": round(similarity, 3),
                    "user_id": meta.get("user_id", "")
                })

        # Sort by similarity (most relevant first)
        memories.sort(key=lambda x: x["similarity"], reverse=True)

        logger.info(
            f"Retrieved {len(memories)} relevant memories for query: "
            f"'{query[:50]}...'"
        )
        return memories

    # ─── GET ALL MEMORIES FOR A USER ──────────────────────────────────────────

    def get_all_memories(self, user_id: str) -> list[dict]:
        """Get all stored memories for a user. Used for the memory dashboard."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            memories = []
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", [])
            ):
                memories.append({
                    "content": doc,
                    "type": meta.get("memory_type", "general"),
                    "date": meta.get("date_str", ""),
                    "timestamp": meta.get("timestamp", 0)
                })
            # Sort by newest first
            memories.sort(key=lambda x: x["timestamp"], reverse=True)
            return memories
        except Exception as e:
            logger.error(f"get_all_memories error: {e}")
            return []

    # ─── DELETE A SPECIFIC MEMORY ─────────────────────────────────────────────

    def delete_memory(self, memory_id: str):
        """Let users delete specific memories (privacy control)."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
        except Exception as e:
            logger.error(f"delete_memory error: {e}")

    # ─── CLEAR ALL MEMORIES FOR A USER ────────────────────────────────────────

    def clear_user_memories(self, user_id: str):
        """Nuclear option — delete all memories for a user."""
        try:
            results = self.collection.get(where={"user_id": user_id})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} memories for {user_id}")
        except Exception as e:
            logger.error(f"clear_user_memories error: {e}")

    # ─── MEMORY LIMIT ENFORCEMENT ─────────────────────────────────────────────

    def _enforce_memory_limit(self, user_id: str):
        """
        Delete the oldest memories when a user exceeds the limit.
        Like a notebook that only has 500 pages — when it's full,
        the oldest pages get torn out.
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            ids = results.get("ids", [])

            if len(ids) > MAX_MEMORIES_PER_USER:
                # Sort by timestamp and delete the oldest
                paired = list(zip(
                    ids,
                    [m.get("timestamp", 0) for m in results.get("metadatas", [])]
                ))
                paired.sort(key=lambda x: x[1])  # oldest first

                to_delete_count = len(ids) - MAX_MEMORIES_PER_USER
                to_delete = [p[0] for p in paired[:to_delete_count]]
                self.collection.delete(ids=to_delete)
                logger.info(f"Pruned {to_delete_count} old memories for {user_id}")

        except Exception as e:
            logger.error(f"Memory limit enforcement error: {e}")

    # ─── FORMAT MEMORIES FOR AI CONTEXT ───────────────────────────────────────

    def format_for_context(self, memories: list[dict]) -> str:
        """
        Convert retrieved memories into text the AI can use.
        
        This text gets prepended to the user's message before
        sending to the AI. The AI sees it as part of the context.
        
        Example output:
        
        FRIDAY's Memory (what I know about you):
        [fact] You live in Addis Ababa. (remembered 3 days ago)
        [preference] You prefer concise responses. (remembered 1 week ago)
        [skill] You know Python programming. (remembered yesterday)
        """
        if not memories:
            return ""

        lines = ["FRIDAY's Memory (what I know about you):"]
        for mem in memories:
            mem_type = mem.get("type", "general")
            content = mem.get("content", "")
            date = mem.get("date", "")
            similarity = mem.get("similarity", 0)

            # Only include high-confidence memories in the context
            if similarity >= 0.75:
                lines.append(f"[{mem_type}] {content} (from: {date})")

        if len(lines) == 1:
            return ""  # No high-confidence memories

        return "\n".join(lines)

    # ─── MEMORY STATS ─────────────────────────────────────────────────────────

    def get_stats(self, user_id: str) -> dict:
        """Get memory statistics for a user."""
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=["metadatas"]
            )
            metadatas = results.get("metadatas", [])
            type_counts = {}
            for meta in metadatas:
                t = meta.get("memory_type", "general")
                type_counts[t] = type_counts.get(t, 0) + 1

            return {
                "total_memories": len(metadatas),
                "by_type": type_counts,
                "limit": MAX_MEMORIES_PER_USER
            }
        except Exception as e:
            return {"error": str(e)}


# ─── MEMORY EXTRACTOR ─────────────────────────────────────────────────────────

class MemoryExtractor:
    """
    Decides what to remember from a conversation.
    
    Not everything the user says is worth remembering.
    "Open YouTube" — not worth storing.
    "I live in Addis Ababa" — definitely worth storing.
    "My name is Cherinet" — absolutely store this.
    "I hate loud music" — store as a preference.
    
    This class uses rules + the AI itself to decide what's memorable.
    """

    # Patterns that suggest something is worth remembering
    FACT_PATTERNS = [
        "my name is", "i am", "i'm", "i live in", "i work at",
        "i study at", "my job is", "my age is", "i was born",
        "my phone number", "my email", "i speak", "i know how to"
    ]

    PREFERENCE_PATTERNS = [
        "i prefer", "i like", "i love", "i hate", "i don't like",
        "i enjoy", "i want", "i need", "my favorite", "i always",
        "i never", "please always", "don't ever", "remind me to always"
    ]

    EVENT_PATTERNS = [
        "tomorrow", "next week", "next month", "on monday", "on friday",
        "i have a meeting", "i have an exam", "my appointment", "don't forget",
        "remind me", "i need to", "deadline", "i'm going to"
    ]

    SKILL_PATTERNS = [
        "i know", "i learned", "i can", "i understand", "i'm good at",
        "i study", "i'm learning", "my hobby", "i play", "i practice"
    ]

    def extract_memories(self, user_message: str, ai_response: str) -> list[dict]:
        """
        Extract memorable information from a conversation turn.
        
        Returns a list of dicts, each with:
          - content: what to remember
          - memory_type: the category
        """
        memories = []
        lower_msg = user_message.lower().strip()

        # Check for facts
        for pattern in self.FACT_PATTERNS:
            if pattern in lower_msg:
                memories.append({
                    "content": self._clean_memory(user_message),
                    "memory_type": MemoryType.FACT
                })
                break

        # Check for preferences
        for pattern in self.PREFERENCE_PATTERNS:
            if pattern in lower_msg:
                memories.append({
                    "content": self._clean_memory(user_message),
                    "memory_type": MemoryType.PREFERENCE
                })
                break

        # Check for events/reminders
        for pattern in self.EVENT_PATTERNS:
            if pattern in lower_msg:
                memories.append({
                    "content": self._clean_memory(user_message),
                    "memory_type": MemoryType.EVENT
                })
                break

        # Check for skills
        for pattern in self.SKILL_PATTERNS:
            if pattern in lower_msg:
                memories.append({
                    "content": self._clean_memory(user_message),
                    "memory_type": MemoryType.SKILL
                })
                break

        return memories

    def _clean_memory(self, text: str) -> str:
        """
        Clean up the text before storing.
        Remove filler words, normalize phrasing.
        """
        # Capitalize first letter, end with period
        text = text.strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text[:500]  # Max 500 chars per memory


# ─── SINGLETON INSTANCE ───────────────────────────────────────────────────────
# We create one instance shared across all requests
# Like having one filing cabinet for the whole office

_memory_manager: Optional[FridayMemoryManager] = None
_memory_extractor = MemoryExtractor()

def get_memory_manager() -> FridayMemoryManager:
    """Get the shared memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = FridayMemoryManager()
    return _memory_manager

def get_memory_extractor() -> MemoryExtractor:
    """Get the shared memory extractor instance."""
    return _memory_extractor