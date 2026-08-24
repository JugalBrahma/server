"""
High-Performance Direct RAG Agent
- Direct Document Retrieval via VectorSearchManager
- Single-pass grounded synthesis with LLM (Groq / OpenAI)
- Clean and fast response without extra LLM hops
"""

import os
from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from .llm_manager import LLMManager
from .vector_manager import VectorSearchManager
from .guardrail import check_greeting_guardrail


class AgentConfig:
    def __init__(self, index_name: Optional[str] = None, namespace: str = "test-2", model_name: Optional[str] = None):
        # Resolve index name from environment or fallback
        index_name = index_name or os.getenv("PINECONE_INDEX_NAME", os.getenv("AGENT_INDEX_NAME", "askadocument"))
        # Initialize LLM (uses default configured model from config/models.py or .env)
        self.llm_manager = LLMManager(model_name=model_name)
        self.llm = self.llm_manager.get_llm()
        # Initialize Pinecone vector store manager for the given namespace
        self.vector_manager = VectorSearchManager(index_name=index_name, namespace=namespace)


class RAGAgent:
    """Streamlined RAG Agent for fast document retrieval and grounded answering"""
    
    def __init__(self, index_name: Optional[str] = None, namespace: str = "test-2", model_name: Optional[str] = None):
        self.config = AgentConfig(index_name=index_name, namespace=namespace, model_name=model_name)
        self.namespace = namespace

    @traceable(run_type="retriever", name="Pinecone Document Retrieval")
    def retrieve_documents(self, query: str) -> str:
        """Search vector database for relevant document chunks"""
        try:
            print(f"🔍 Searching for: '{query}' in namespace '{self.namespace}'")
            # Fetch top 8 matching chunks from Pinecone
            results = self.config.vector_manager.search(query=query, k=8)
            
            if not results:
                return ""
            
            # Group chunks by document ID and S3 source
            results_by_doc = defaultdict(list)
            for doc in results:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                doc_id = metadata.get('doc_id', metadata.get('s3_key', 'unknown_doc'))
                s3_key = metadata.get('s3_key', 'unknown_source')
                namespace_hint = metadata.get('namespace', self.namespace)
                key = (doc_id, s3_key, namespace_hint)
                results_by_doc[key].append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "content_type": metadata.get('content_type', 'unknown')
                })

            # Format grouped chunks with page numbers and headers
            formatted_results = []
            for i, ((doc_id, s3_key, namespace_hint), chunks) in enumerate(results_by_doc.items(), 1):
                # Sort chunks by page number
                chunks.sort(key=lambda c: c["metadata"].get('page_number', 0))
                combined_text = "\n\n".join(chunk["content"] for chunk in chunks)
                
                # Calculate page span
                first_page = chunks[0]["metadata"].get('page_number')
                last_page = chunks[-1]["metadata"].get('page_number')
                if first_page is not None and last_page is not None:
                    page_span = f"pages {first_page}–{last_page}" if first_page != last_page else f"page {first_page}"
                elif first_page is not None:
                    page_span = f"page {first_page}"
                else:
                    page_span = "unknown pages"

                content_type = chunks[0]["content_type"]
                section_header = (
                    f"[Document {i}]\n"
                    f"ID: {doc_id}\n"
                    f"Source: {s3_key}\n"
                    f"Type: {content_type}\n"
                    f"Pages: {page_span}\n"
                )

                formatted_results.append(
                    f"{section_header}Content:\n{combined_text}\n" + "=" * 50 + "\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return ""

    @traceable(run_type="chain", name="RAG Agent Query")
    def query(self, user_input: str, folder_id: Optional[str] = None, user_id: Optional[str] = None) -> dict:
        """Process user query with greeting guardrail and single-pass RAG"""
        assert isinstance(user_input, str)
        
        # Attach folder and user context to LangSmith trace
        try:
            run_tree = get_current_run_tree()
            if run_tree:
                if folder_id:
                    run_tree.extra.setdefault("metadata", {})["folder_id"] = folder_id
                    if f"folder_{folder_id}" not in run_tree.tags:
                        run_tree.tags.append(f"folder_{folder_id}")
                if user_id:
                    run_tree.extra.setdefault("metadata", {})["user_id"] = user_id
                    if f"user_{user_id}" not in run_tree.tags:
                        run_tree.tags.append(f"user_{user_id}")
        except Exception as e:
            print(f"⚠️ Trace metadata error: {e}")
        
        # 1. Ingress Guardrail: Check for simple greetings (instant return without search)
        is_greeting, greeting_reply = check_greeting_guardrail(user_input)
        if is_greeting and greeting_reply:
            print(f"🛡️ Greeting guardrail triggered for '{user_input}'")
            return {
                "answer": greeting_reply
            }

        # 2. Retrieve relevant context chunks from Pinecone
        retrieved_docs = self.retrieve_documents(user_input)
        
        # 3. Build system instructions and user prompt
        if retrieved_docs:
            # Enforce strict grounding with <doc> tags
            system_prompt = (
                "You are an intelligent document assistant. "
                "Answer the question based strictly on the retrieved document context below. "
                "Wrap all facts, quotes, figures, and direct information from the documents in <doc> and </doc> tags. "
                "If the context does not contain the answer, politely state that the information was not found in the documents."
            )
            user_prompt = f"""User Question: "{user_input}"

Retrieved Documents:
{retrieved_docs}

Answer:"""
        else:
            # Fallback prompt when no matching documents are found
            system_prompt = (
                "You are a helpful assistant. If the user greets you, reply politely. "
                "If they ask a question about specific documents, inform them that no matching documents were found in this folder."
            )
            user_prompt = user_input

        # 4. Single LLM invocation (Groq / OpenAI)
        response = self.config.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # 5. Extract text content
        response_content = response.content if hasattr(response, 'content') else str(response)
        if isinstance(response_content, list):
            response_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response_content
            )

        # 6. Return response payload
        return {
            "answer": response_content
        }

    def add_documents(self, documents, metadatas=None, s3_key=None):
        """Add documents to the vector store"""
        # Embed and upsert document chunks to Pinecone
        return self.config.vector_manager.add_documents(documents, metadatas, s3_key)

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store by ID"""
        # Remove vector embeddings from Pinecone matching doc_ids
        return self.config.vector_manager.delete_documents(doc_ids)
