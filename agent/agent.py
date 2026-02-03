"""
RAG Agent with LangGraph
- Smart agent that decides between greetings and document queries
- Uses RAG pipeline functions for document retrieval
- Complete agent orchestration
"""

import json
from typing import TypedDict, Annotated, List, Any
from collections import defaultdict
import os
import tiktoken
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, AnyMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from .llm_manager import LLMManager
from .vector_manager import VectorSearchManager
from .conversation_manager import FirestoreConversationManager
from pydantic import BaseModel, Field, ValidationError

# Initialize tiktoken encoder for token counting
try:
    _tokenizer = tiktoken.encoding_for_model("gpt-4")
except Exception:
    _tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens using local tiktoken (fast, no API call)."""
    if not text:
        return 0
    try:
        return len(_tokenizer.encode(text))
    except Exception:
        return len(text) // 4

class MCQ(BaseModel):
    question: str = Field(..., description="The multiple choice question")
    options: List[str] = Field(..., min_items=4, max_items=4,
                               description="List of 4 options")
    answer: str = Field(..., description="The correct answer from the options")


class MCQList(BaseModel):
    mcqs: List[MCQ] = Field(default_factory=list, description="List of MCQs")


# ==================== STATE DEFINITION ====================
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# ==================== AGENT CONFIGURATION ====================
def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage_from_message(message, input_text: str = ""):
    """Return normalized token usage info from an AIMessage (if available).
    Falls back to manual counting if API doesn't provide token usage."""
    if message is None:
        return None
        
    metadata = getattr(message, "response_metadata", None) or {}
    
    # Extract content from message - handle various formats
    content = ""
    if hasattr(message, "content"):
        raw_content = message.content
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            # Handle list format (common in some LangChain versions)
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
    
    print(f"DEBUG extract_usage: input_text='{input_text}', content='{content[:100]}...'")
    print(f"DEBUG: Full metadata: {json.dumps(metadata, indent=2, default=str)[:500]}")
    
    # Try to get usage from API response metadata - check multiple locations
    usage_meta = None
    
    # Check 1: Direct usage_metadata
    if "usage_metadata" in metadata:
        usage_meta = metadata["usage_metadata"]
        print(f"DEBUG: Found usage_metadata at root level")
    
    # Check 2: Direct usage
    elif "usage" in metadata:
        usage_meta = metadata["usage"]
        print(f"DEBUG: Found usage at root level")
    
    # Check 3: token_usage (older LangChain versions)
    elif "token_usage" in metadata:
        usage_meta = metadata["token_usage"]
        print(f"DEBUG: Found token_usage at root level")
    
    # Check 4: Gemini specific nested in metadata
    if not usage_meta and isinstance(metadata, dict):
        inner_meta = metadata.get("metadata", {})
        if isinstance(inner_meta, dict):
            if "usage_metadata" in inner_meta:
                usage_meta = inner_meta["usage_metadata"]
                print(f"DEBUG: Found usage_metadata in metadata.metadata")
            elif "usage" in inner_meta:
                usage_meta = inner_meta["usage"]
                print(f"DEBUG: Found usage in metadata.metadata")
    
    # Check 5: Check for direct token_count
    if not usage_meta and isinstance(metadata, dict):
        if "token_count" in metadata:
            usage_meta = {"total_token_count": metadata["token_count"]}
            print(f"DEBUG: Found token_count at root")
        # Check in candidates (Gemini format)
        candidates = metadata.get("candidates", [])
        if candidates and isinstance(candidates, list):
            first_candidate = candidates[0]
            if isinstance(first_candidate, dict):
                candidate_meta = first_candidate.get("metadata", {})
                if "token_count" in candidate_meta:
                    usage_meta = {"total_token_count": candidate_meta["token_count"]}
                    print(f"DEBUG: Found token_count in candidates[0].metadata")
    
    print(f"DEBUG: usage_meta found: {usage_meta}")
    
    # If we got usage from API, normalize it
    if isinstance(usage_meta, dict) and usage_meta:
        # Gemini/Google GenAI specific field names
        prompt_tokens = (
            _safe_int(usage_meta.get("prompt_token_count"))
            or _safe_int(usage_meta.get("promptTokens"))
            or _safe_int(usage_meta.get("input_tokens"))
            or _safe_int(usage_meta.get("prompt_tokens"))
            or _safe_int(usage_meta.get("inputTokenCount"))
        )
        completion_tokens = (
            _safe_int(usage_meta.get("candidates_token_count"))
            or _safe_int(usage_meta.get("completion_token_count"))
            or _safe_int(usage_meta.get("output_tokens"))
            or _safe_int(usage_meta.get("candidatesTokens"))
            or _safe_int(usage_meta.get("completion_tokens"))
            or _safe_int(usage_meta.get("outputTokenCount"))
        )
        total_tokens = (
            _safe_int(usage_meta.get("total_token_count"))
            or _safe_int(usage_meta.get("totalTokens"))
            or _safe_int(usage_meta.get("total_tokens"))
            or _safe_int(usage_meta.get("totalTokenCount"))
            or (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            )
        )
        
        print(f"DEBUG: Extracted - prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        if prompt_tokens is not None or completion_tokens is not None:
            normalized = {
                key: value
                for key, value in (
                    ("prompt_tokens", prompt_tokens),
                    ("completion_tokens", completion_tokens),
                    ("total_tokens", total_tokens),
                )
                if value is not None
            }
            normalized["raw"] = usage_meta
            print(f"DEBUG: Returning normalized usage from API: {normalized}")
            return normalized
    
    # FALLBACK: Manually count tokens since API didn't provide them
    print(f"DEBUG: API didn't return token usage, using manual counting")
    
    prompt_tokens = count_tokens(input_text) if input_text else 0
    completion_tokens = count_tokens(content) if content else 0
    total_tokens = prompt_tokens + completion_tokens
    
    print(f"DEBUG: Manually counted tokens - prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated": True  # Flag to indicate these are estimated counts
    }


def merge_usage_totals(*usages):
    """Combine multiple usage dicts into summed totals."""
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False

    for usage in usages:
        if not usage:
            continue
        found = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    if not found:
        return None

    # Drop zero entries if they never received data
    cleaned = {k: v for k, v in totals.items() if v}
    return cleaned or None


class AgentConfig:
    def __init__(self, index_name: str = "askadocument-dot", namespace: str = "test-2"):
        # Initialize LLM Manager
        self.llm_manager = LLMManager(model_name="gemini-2.5-flash")  # Updated to 2.0 flash lite
        self.llm = self.llm_manager.get_llm()
        
        # Initialize Vector Search Manager
        self.vector_manager = VectorSearchManager(index_name, namespace)

        # Structured MCQ detector prompt
        mcq_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Extract only the multiple-choice questions explicitly present in the text. "
             "Each MCQ must list exactly 4 options and one correct answer pulled from those options. "
             "If no MCQ exists, return an empty list."),
            ("human", "{text}")
        ])
        self.mcq_chain = mcq_prompt | self.llm.with_structured_output(MCQList)

# ==================== RETRIEVAL TOOL ====================
def create_retrieval_tool(config: AgentConfig):
    """Create the document retrieval tool"""
    
    @tool
    def retrieve_docs(query: str) -> str:
        """
        Retrieve relevant documents from the vector database based on the query.
        Use this tool for ANY question about documents, facts, news, tables, or figures.
        """
        try:
            print(f"🔍 Searching for: {query}")
            print(f"🔍 Namespace: {config.vector_manager.namespace}")
            print(f"🔍 Index: {config.vector_manager.index_name}")
            
            # Search the vector store
            results = config.vector_manager.search(query=query, k=20)
            
            print(f"🔍 Found {len(results)} results")
            
            if not results:
                return "No relevant documents found in the database."
            
            # Group chunks by document identifiers (doc_id + s3_key)
            results_by_doc = defaultdict(list)
            for doc in results:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                doc_id = metadata.get('doc_id', metadata.get('s3_key', 'unknown_doc'))
                s3_key = metadata.get('s3_key', 'unknown_source')
                namespace_hint = metadata.get('namespace', config.vector_manager.namespace)
                key = (doc_id, s3_key, namespace_hint)
                results_by_doc[key].append({
                    "content": doc.page_content,
                    "metadata": metadata,
                    "content_type": metadata.get('content_type', 'unknown')
                })

            # Merge chunks per document and format
            formatted_results = []
            for i, ((doc_id, s3_key, namespace_hint), chunks) in enumerate(results_by_doc.items(), 1):
                chunks.sort(key=lambda c: c["metadata"].get('page_number', 0))
                combined_text = "\n\n".join(chunk["content"] for chunk in chunks)
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
                    f"Namespace: {namespace_hint}\n"
                    f"Type: {content_type}"
                )
                section_header += f"\nPages: {page_span}\n"

                formatted_results.append(
                    f"{section_header}Content: {combined_text}\n" + "=" * 50 + "\n"
                )

            final_result = "\n".join(formatted_results)
            print(f"📄 Retrieved content preview: {final_result[:500]}...")
            return final_result
            
        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return f"Error retrieving documents: {str(e)}"
    
    return retrieve_docs

# ==================== AGENT LOGIC ====================
def create_agent(config: AgentConfig):
    """Create the RAG agent with LangGraph"""
    
    # Create tools
    retrieve_docs = create_retrieval_tool(config)
    tools = [retrieve_docs]
    
    # Bind tools to LLM
    llm_with_tools = config.llm.bind_tools(tools)
    
    def agent(state: State):
        """Agent function that decides whether to use tools or respond directly"""
        
        # Get the latest user message (HumanMessage). When tools run, the latest
        # message may be a ToolMessage, so we must find the most recent HumanMessage.
        user_input = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break
        print(f"DEBUG: Agent received user input: '{user_input}'")
        
        # Check if we just got tool results - if so, generate final response
        print(f"DEBUG: Message count: {len(state['messages'])}")
        if len(state["messages"]) > 1:
            print(f"DEBUG: Previous message type: {type(state['messages'][-2])}")
            print(f"DEBUG: Previous message has tool_calls: {hasattr(state['messages'][-2], 'tool_calls')}")
            if hasattr(state['messages'][-2], 'tool_calls'):
                print(f"DEBUG: Tool calls: {state['messages'][-2].tool_calls}")
        
        if len(state["messages"]) > 1 and hasattr(state["messages"][-2], 'tool_calls'):
            print("DEBUG: Agent got tool results, generating final response")
            # We just got tool results, now generate final answer
            tool_results = state["messages"][-1].content
            
            # DEBUG: Show what documents were retrieved
            print(f"DEBUG: Retrieved documents content: {tool_results[:500]}...")
            
            final_prompt = f"""
            You MUST answer based ONLY on the retrieved documents below. Do not use any outside knowledge.
            
            User Question: "{user_input}"
            
            Retrieved Documents:
            {tool_results}
            
            INSTRUCTIONS:
            1. Base your answer ENTIRELY on the retrieved documents
            2. For all facts, figures, and specific information from documents, wrap the exact text with <doc> and </doc> tags
            3. Example: "According to the document, <doc>The revenue increased by 25%</doc> in Q3."
            4. Use correct spellings and match exact wording from the documents when quoting or referencing them.
            5. If the documents don't contain the answer, say "I couldn't find this information in your documents."
            
            Answer:
            """
            
            final_response = config.llm.invoke([SystemMessage(content="You answer questions based ONLY on retrieved documents. Always wrap document information with <doc> and </doc> tags."), 
                                               HumanMessage(content=final_prompt)])
            
            return {"messages": [final_response]}
        
        # Create system instructions
        instructions = SystemMessage(content=f"""
        SYSTEM ROLE:
        You are a RAG (Retrieval-Augmented Generation) assistant. You must decide whether to use a tool or speak directly.

        CORE LOGIC:
        1. GREETINGS: If the user says 'hi', 'hello', 'how are you', or similar, just reply politely. DO NOT use 'retrieve_docs'.
        2. DATA QUERIES: For ANY question about documents, news, facts, table or figure, you MUST use the 'retrieve_docs' tool.
        3. MCQ REQUESTS: If the user explicitly asks for multiple-choice questions, generate well-formatted MCQs with a clear question, exactly four answer options (A/B/C/D), and identify the correct option in your explanation.

        CRITICAL GUARDRAIL:
        Never say 'I don't know which documents you mean' or 'Please specify documents'. 
        Instead, just CALL the 'retrieve_docs' tool with the user's query. 
        The tool will search the database for you.

        USER INPUT: "{user_input}"
        """)
        
        # Prepend instructions to message history
        messages = [instructions] + state["messages"]
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        
        # DEBUG: Check if tool was called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"DEBUG: Agent is calling tool: {response.tool_calls}")
        else:
            print("DEBUG: Agent is NOT calling tool, responding directly")
        
        return {"messages": [response]}
    
    # Build the graph
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    # Compile the graph
    graph = builder.compile()
    
    return graph

# ==================== MAIN RAG AGENT CLASS ====================
class RAGAgent:
    """Main RAG Agent class with Firestore conversation persistence"""
    
    def __init__(self, index_name: str = "askadocument-dot", namespace: str = "test-2"):
        self.config = AgentConfig(index_name, namespace)
        self.graph = create_agent(self.config)
        
        # Extract user_id from namespace for conversation management
        # Namespace format: "user_{user_id}"
        namespace_parts = namespace.split("_")
        if len(namespace_parts) > 1 and namespace_parts[0] == "user":
            # User-wide namespace: user_{user_id}
            self.user_id = "_".join(namespace_parts[1:])
        else:
            self.user_id = "unknown"
        
        # No folder_id anymore since we use single namespace per user
        self.folder_id = None
        self.conversation_manager = FirestoreConversationManager(self.user_id, self.folder_id)
    
    def query(self, user_input: str) -> dict:
        assert isinstance(user_input, str)

        # Disable feeding prior conversation context into the agent so answers
        # always depend on current question + retrieved documents.
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }

        result = self.graph.invoke(initial_state)
        final_message = result["messages"][-1]

        try:
            print("DEBUG: final_message type:", type(final_message))
            print("DEBUG: final_message.response_metadata:", getattr(final_message, "response_metadata", None))
            print("DEBUG: final_message.additional_kwargs:", getattr(final_message, "additional_kwargs", None))
        except Exception as meta_exc:
            print(f"DEBUG: Unable to log final_message metadata: {meta_exc}")

        if hasattr(final_message, 'content'):
            response_content = final_message.content
        else:
            response_content = str(final_message)

        if isinstance(response_content, list):
            response_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in response_content
            )

        # Detect MCQs using structured output - ONLY if user asked for MCQs or response is long enough
        mcqs: List[MCQ] = []
        mcq_usage = None
        
        # Check if user explicitly asked for MCQs
        user_asked_mcq = any(keyword in user_input.lower() for keyword in 
                            ['mcq', 'quiz', 'test', 'question', 'exam', 'practice'])
        
        # Only run MCQ detection if user asked for it AND response is substantial
        if user_asked_mcq and len(response_content) > 200:
            try:
                detection = self.config.mcq_chain.invoke({"text": response_content})
                mcqs = detection.mcqs
                mcq_usage = extract_usage_from_message(getattr(detection, "_lc_generation", None), user_input)
            except Exception as exc:
                print(f"MCQ detection error: {exc}")

        answer_usage = extract_usage_from_message(final_message, user_input)

        usage_payload = {
            "answer": answer_usage,
            "mcq_detection": mcq_usage,
            "totals": merge_usage_totals(answer_usage, mcq_usage)
        }
        # Remove empty sections
        usage_payload = {
            key: value for key, value in usage_payload.items() if value
        }

        payload = {
            "answer": response_content,
            "mcq_is_true": bool(mcqs),
            "mcqs": [mcq.model_dump() for mcq in mcqs],
            "usage": usage_payload or None,
        }

        return payload
    
    def clear_memory(self):
        """Clear conversation memory from Firestore"""
        success = self.conversation_manager.clear_conversation()
        if success:
            print("🧹 Conversation memory cleared from Firestore")
        else:
            print("❌ Failed to clear conversation memory from Firestore")
    
    def get_memory_summary(self) -> str:
        """Get summary of conversation memory from Firestore"""
        return self.conversation_manager.get_conversation_summary()
    
    def add_documents(self, documents, metadatas=None, s3_key=None):
        """Add documents to the vector store"""
        # The namespace is already set during initialization
        return self.config.vector_manager.add_documents(documents, metadatas, s3_key)
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete documents from the vector store by ID"""
        return self.config.vector_manager.delete_documents(doc_ids)

# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Initialize the RAG agent
    rag_agent = RAGAgent()
    
    # Example queries
    test_queries = [
        "Hello, how are you?",
        "What information do you have about machine learning?",
        "Show me data about recent trends",
        "What are the key findings in the documents?"
    ]
    
    print("🚀 Testing RAG Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        response = rag_agent.query(query)
        print(f"🤖 Response: {response}")
        print("-" * 50)
    
    # Optional: Add some test documents
    test_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on neural networks and deep learning.",
        "Recent trends in AI include large language models, computer vision, and reinforcement learning.",
        "Data science combines statistics, programming, and domain expertise to extract insights from data."
    ]
    
    test_metadatas = [
        {"content_type": "text", "source": "ml_basics"},
        {"content_type": "text", "source": "ai_trends"},
        {"content_type": "text", "source": "data_science"}
    ]
    
    print("\n📚 Adding test documents...")
    rag_agent.add_documents(test_docs, test_metadatas)
    
    print("\n🔄 Testing with added documents...")
    response = rag_agent.query("What is machine learning?")
    print(f"🤖 Response: {response}")
