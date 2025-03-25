import os
from typing import Dict, List, Optional, Any
import logging
from dotenv import load_dotenv
import anthropic
from odoo_rag.vectorstore import OdooVectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OdooRAG:
    """Retrieval-Augmented Generation system for Odoo code and documentation using ChromaDB and Anthropic"""
    
    def __init__(self, vector_store: OdooVectorStore, 
                 model_name: str = "claude-3-5-haiku-20241022", 
                 temperature: float = 0.0):
        """
        Initialize the RAG system
        
        Args:
            vector_store: The vector store to use for retrieval
            model_name: The Claude model to use (default is claude-3-sonnet)
            temperature: Temperature for the LLM
        """
        self.vector_store = vector_store
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize the Anthropic client
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        # Create the default prompt template
        self.default_template = (
            "You are an Odoo expert assistant that helps developers understand Odoo code and modules.\n"
            "Use the following pieces of context to answer the question at the end.\n"
            "If you don't know the answer, just say you don't know. Don't try to make up an answer.\n"
            "Keep your answers technical and focused on Odoo development.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
        
        # Create specialized prompt templates
        self.model_template = (
            "You are an Odoo expert assistant that helps developers understand Odoo models and fields.\n"
            "Use the following pieces of context about Odoo models to answer the question at the end.\n"
            "Explain Odoo ORM concepts clearly and provide examples when relevant.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
        
        self.view_template = (
            "You are an Odoo expert assistant that helps developers understand Odoo views and UI.\n"
            "Use the following pieces of context about Odoo views to answer the question at the end.\n"
            "Explain view inheritance and XML structure clearly when relevant.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
    
    def _format_context(self, documents: List[Dict]) -> str:
        """
        Format the retrieved documents into a context string
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get("metadata", {})
            module = metadata.get("module", "unknown")
            file_path = metadata.get("file_path", "unknown")
            doc_type = metadata.get("type", "unknown")
            
            header = f"[Document {i}] Module: {module}, Path: {file_path}, Type: {doc_type}"
            content = doc.get("content", "")
            
            context_parts.append(f"{header}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _select_prompt_for_question(self, question: str) -> str:
        """
        Select the most appropriate prompt template based on the question
        
        Args:
            question: The user's question
            
        Returns:
            The selected prompt template
        """
        question_lower = question.lower()
        
        # Check for model-related questions
        model_keywords = ["model", "field", "orm", "inheritance", "method", "record", "database"]
        view_keywords = ["view", "form", "tree", "kanban", "xml", "qweb", "ui", "button", "action"]
        
        if any(keyword in question_lower for keyword in model_keywords):
            return self.model_template
        elif any(keyword in question_lower for keyword in view_keywords):
            return self.view_template
        else:
            return self.default_template
    
    def answer_question(self, question: str, filter: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Answer a question using the RAG system
        
        Args:
            question: Question to answer
            filter: Optional filter to apply to the retrieval
            
        Returns:
            Dict containing the answer and source documents
        """
        # Retrieve relevant documents
        docs = self.vector_store.search(query=question, filter=filter, k=5)
        
        # Format the context
        context_str = self._format_context(docs)
        
        # Select the appropriate prompt template
        prompt_template = self._select_prompt_for_question(question)
        
        # Fill in the prompt template
        prompt = prompt_template.format(context_str=context_str, query_str=question)
        
        # Get response from Claude
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Process the response
        answer = message.content[0].text
        
        # Return the result
        return {
            "result": answer,
            "source_documents": docs
        }
    
    def answer_about_module(self, question: str, module_name: str) -> Dict[str, Any]:
        """
        Answer a question about a specific module
        
        Args:
            question: Question to answer
            module_name: Name of the module to focus on
            
        Returns:
            Dict containing the answer and source documents
        """
        return self.answer_question(
            question=question,
            filter={"module": module_name}
        )
    
    def answer_about_model(self, question: str, model_name: str) -> Dict[str, Any]:
        """
        Answer a question about a specific model
        
        Args:
            question: Question to answer
            model_name: Technical name of the model (e.g. 'res.partner')
            
        Returns:
            Dict containing the answer and source documents
        """
        # Use the vector store's specialized search method
        docs = self.vector_store.search_by_model(query=question, model_name=model_name, k=5)
        
        # Format the context
        context_str = self._format_context(docs)
        
        # Select the model prompt template
        prompt_template = self.model_template
        
        # Fill in the prompt template
        prompt = prompt_template.format(context_str=context_str, query_str=question)
        
        # Get response from Claude
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1000,
            temperature=self.temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Process the response
        answer = message.content[0].text
        
        # Return the result
        return {
            "result": answer,
            "source_documents": docs
        } 