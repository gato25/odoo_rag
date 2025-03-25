import os
import argparse
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from odoo_rag.indexer import OdooModuleParser
from odoo_rag.vectorstore import OdooVectorStore
from odoo_rag.rag import OdooRAG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_argparse() -> argparse.ArgumentParser:
    """Setup the argument parser"""
    parser = argparse.ArgumentParser(description='Odoo RAG System CLI')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index Odoo modules')
    index_parser.add_argument('--modules-path', type=str, required=True, help='Path to Odoo modules directory')
    index_parser.add_argument('--persist-dir', type=str, default='chroma_db', help='Directory to persist the vector store')
    index_parser.add_argument('--embedding-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Embedding model to use')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('--question', type=str, required=True, help='Question to answer')
    query_parser.add_argument('--persist-dir', type=str, default='chroma_db', help='Directory where the vector store is persisted')
    query_parser.add_argument('--module', type=str, help='Optional module to restrict the search to')
    query_parser.add_argument('--model', type=str, help='Optional model to restrict the search to')
    query_parser.add_argument('--llm-model', type=str, default='claude-3-sonnet-20240229', 
                             help='Claude model to use (e.g., claude-3-sonnet-20240229, claude-3-opus-20240229)')
    query_parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the LLM')
    query_parser.add_argument('--output-format', type=str, choices=['text', 'json'], default='text', help='Output format')
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start an interactive session')
    interactive_parser.add_argument('--persist-dir', type=str, default='chroma_db', help='Directory where the vector store is persisted')
    interactive_parser.add_argument('--llm-model', type=str, default='claude-3-sonnet-20240229',
                                  help='Claude model to use (e.g., claude-3-sonnet-20240229, claude-3-opus-20240229)')
    interactive_parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the LLM')
    
    return parser

def index_modules(args: argparse.Namespace) -> None:
    """Index Odoo modules and create the vector store"""
    logger.info(f"Indexing modules from {args.modules_path}")
    
    # Create the module parser
    parser = OdooModuleParser(args.modules_path)
    
    # Index all modules
    modules = parser.index_all_modules()
    logger.info(f"Indexed {len(modules)} modules")
    
    # Extract chunks for embedding
    chunks = parser.extract_chunks_for_embedding()
    logger.info(f"Extracted {len(chunks)} chunks for embedding")
    
    # Create the vector store and add the chunks
    vector_store = OdooVectorStore(persist_directory=args.persist_dir, model_name=args.embedding_model)
    vector_store.add_chunks(chunks)
    logger.info(f"Added chunks to vector store at {args.persist_dir}")

def query_rag(args: argparse.Namespace) -> None:
    """Query the RAG system"""
    logger.info(f"Querying RAG system with question: {args.question}")
    
    # Check if the vector store exists
    if not os.path.exists(args.persist_dir):
        logger.error(f"Vector store not found at {args.persist_dir}. Please index modules first.")
        return
    
    # Check if Anthropic API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set. Please set it in your .env file.")
        return
    
    # Create the vector store
    vector_store = OdooVectorStore(persist_directory=args.persist_dir)
    
    # Create the RAG system
    rag = OdooRAG(vector_store=vector_store, model_name=args.llm_model, temperature=args.temperature)
    
    # Query the RAG system
    if args.module:
        result = rag.answer_about_module(question=args.question, module_name=args.module)
    elif args.model:
        result = rag.answer_about_model(question=args.question, model_name=args.model)
    else:
        result = rag.answer_question(question=args.question)
    
    # Format and output the result
    if args.output_format == 'json':
        print(json.dumps(result, indent=2))
    else:
        print("\nAnswer:")
        print(result.get('result', 'No answer found.'))
        
        if 'source_documents' in result and result['source_documents']:
            print("\nSources:")
            for i, doc in enumerate(result['source_documents'][:3]):  # Show top 3 sources
                metadata = doc.get('metadata', {})
                source = metadata.get('file_path', 'Unknown')
                print(f"{i+1}. {source}")

def start_interactive_session(args: argparse.Namespace) -> None:
    """Start an interactive RAG session"""
    # Check if the vector store exists
    if not os.path.exists(args.persist_dir):
        logger.error(f"Vector store not found at {args.persist_dir}. Please index modules first.")
        return
    
    # Check if Anthropic API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY environment variable not set. Please set it in your .env file.")
        return
    
    # Create the vector store
    vector_store = OdooVectorStore(persist_directory=args.persist_dir)
    
    # Get collection stats
    stats = vector_store.get_stats()
    
    # Create the RAG system
    rag = OdooRAG(vector_store=vector_store, model_name=args.llm_model, temperature=args.temperature)
    
    print("Odoo RAG Interactive Session (powered by Claude and ChromaDB)")
    print(f"Vector store contains {stats.get('total_documents', 0)} documents")
    print("Type 'exit' or 'quit' to end the session")
    print("Special commands:")
    print("  /module <module_name>: Set module filter")
    print("  /model <model_name>: Set model filter")
    print("  /clear: Clear all filters")
    
    # Session state
    module_filter = None
    model_filter = None
    
    while True:
        # Show current filters
        filter_info = []
        if module_filter:
            filter_info.append(f"module: {module_filter}")
        if model_filter:
            filter_info.append(f"model: {model_filter}")
        
        if filter_info:
            print(f"\nActive filters: {', '.join(filter_info)}")
        
        # Get user input
        try:
            user_input = input("\nQuestion: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Process special commands
        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == '/module' and len(parts) > 1:
                module_filter = parts[1]
                print(f"Set module filter to: {module_filter}")
                continue
            elif command == '/model' and len(parts) > 1:
                model_filter = parts[1]
                print(f"Set model filter to: {model_filter}")
                continue
            elif command == '/clear':
                module_filter = None
                model_filter = None
                print("Cleared all filters")
                continue
        
        # Query the RAG system with the current filters
        if module_filter:
            result = rag.answer_about_module(question=user_input, module_name=module_filter)
        elif model_filter:
            result = rag.answer_about_model(question=user_input, model_name=model_filter)
        else:
            result = rag.answer_question(question=user_input)
        
        # Print the answer
        print("\nAnswer:")
        print(result.get('result', 'No answer found.'))
        
        if 'source_documents' in result and result['source_documents']:
            print("\nSources:")
            for i, doc in enumerate(result['source_documents'][:3]):  # Show top 3 sources
                metadata = doc.get('metadata', {})
                source = metadata.get('file_path', 'Unknown')
                doc_type = metadata.get('type', 'Unknown')
                print(f"{i+1}. {source} ({doc_type})")

def main() -> None:
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command == 'index':
        index_modules(args)
    elif args.command == 'query':
        query_rag(args)
    elif args.command == 'interactive':
        start_interactive_session(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 