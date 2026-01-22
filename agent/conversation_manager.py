"""
Firestore Conversation Manager
- Manages conversation history in Firestore
- Provides conversation context for RAG agent
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import json

class FirestoreConversationManager:
    """Manages conversation history using Firestore"""
    
    def __init__(self, user_id: str, folder_id: Optional[str] = None):
        """
        Initialize conversation manager
        
        Args:
            user_id: Firebase user ID
            folder_id: Optional folder ID for folder-specific conversations
        """
        self.user_id = user_id
        self.folder_id = folder_id
        self.db = self._get_firestore()
    
    def _get_firestore(self):
        """Get Firestore instance"""
        try:
            from config.firebase import get_firestore
            return get_firestore()
        except Exception as e:
            print(f"❌ Failed to initialize Firestore: {e}")
            return None
    
    def _get_conversation_ref(self):
        """Get reference to conversation collection"""
        if not self.db:
            return None
            
        base_ref = (self.db
                    .collection("users")
                    .document(self.user_id))

        if self.folder_id:
            # Folder-specific conversation stored alongside UI chats
            return (base_ref
                    .collection("folders")
                    .document(self.folder_id)
                    .collection("chats")
                    .document("main_chat")
                    .collection("messages"))

        # User-wide conversation fallback (no folder scope)
        return (base_ref
                .collection("chats")
                .document("main_chat")
                .collection("messages"))
    
    def get_recent_messages(self, limit: int = 10) -> List[Any]:
        """
        Get recent conversation messages from Firestore
        
        Args:
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of LangChain message objects
        """
        if not self.db:
            print("⚠️ Firestore not available, using empty conversation")
            return []
        
        try:
            conversation_ref = self._get_conversation_ref()
            if not conversation_ref:
                return []
            
            # Get recent messages, ordered by timestamp
            docs = (conversation_ref
                   .order_by("timestamp", direction="DESCENDING")
                   .limit(limit)
                   .get())
            
            messages = []
            for doc in docs:
                data = doc.to_dict()
                
                # Convert Firestore message back to LangChain format
                content = data.get("text", "")
                is_user = data.get("isUser", False)
                role = data.get("role", "user")
                
                if role == "user" or is_user:
                    messages.append(HumanMessage(content=content))
                elif role == "assistant" or not is_user:
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
            
            # Reverse to get chronological order (oldest first)
            messages.reverse()
            
            print(f"📚 Loaded {len(messages)} messages from Firestore")
            return messages
            
        except Exception as e:
            print(f"❌ Failed to load conversation from Firestore: {e}")
            return []
    
    def save_message(self, content: str, is_user: bool, role: str = "user", 
                     processing_time: Optional[float] = None):
        """
        Save a message to Firestore
        
        Args:
            content: Message content
            is_user: Whether this is a user message
            role: Message role (user, assistant, system)
            processing_time: Optional processing time for AI responses
        """
        if not self.db:
            print("⚠️ Firestore not available, skipping save")
            return
        
        try:
            conversation_ref = self._get_conversation_ref()
            if not conversation_ref:
                return
            
            message_data = {
                "text": content,
                "isUser": is_user,
                "role": role,
                "timestamp": datetime.utcnow(),
            }
            
            if processing_time is not None:
                message_data["processing_time"] = processing_time
            
            conversation_ref.add(message_data)
            print(f"💾 Saved {role} message to Firestore")
            
        except Exception as e:
            print(f"❌ Failed to save message to Firestore: {e}")
    
    def save_user_message(self, content: str):
        """Save a user message"""
        self.save_message(content, is_user=True, role="user")
    
    def save_ai_response(self, content: str, processing_time: Optional[float] = None):
        """Save an AI response"""
        self.save_message(content, is_user=False, role="assistant", 
                        processing_time=processing_time)
    
    def clear_conversation(self) -> bool:
        """
        Clear all messages in the conversation
        
        Returns:
            True if successful, False otherwise
        """
        if not self.db:
            print("⚠️ Firestore not available, cannot clear conversation")
            return False
        
        try:
            conversation_ref = self._get_conversation_ref()
            if not conversation_ref:
                return False
            
            # Get all messages and delete them
            docs = conversation_ref.get()
            
            # Delete in batches (Firestore limit is 500 operations per batch)
            batch_size = 400
            batch = self.db.batch()
            
            for i, doc in enumerate(docs):
                batch.delete(doc.reference)
                
                # Execute batch when it reaches the limit
                if (i + 1) % batch_size == 0:
                    batch.commit()
                    batch = self.db.batch()
            
            # Execute any remaining operations
            if len(docs) % batch_size != 0:
                batch.commit()
            
            print(f"🗑️ Cleared {len(docs)} messages from Firestore")
            return True
            
        except Exception as e:
            print(f"❌ Failed to clear conversation from Firestore: {e}")
            return False
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        messages = self.get_recent_messages(limit=5)
        
        if not messages:
            return "No conversation history"
        
        summary = f"Conversation has {len(messages)} recent messages:\n"
        for i, msg in enumerate(messages, 1):
            msg_type = type(msg).__name__
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary += f"{i}. {msg_type}: {content_preview}\n"
        
        return summary
