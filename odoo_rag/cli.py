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
    query_parser.add_argument('--llm-model', type=str, default='claude-3-5-haiku-20241022', 
                             help='Claude model to use (e.g., claude-3-sonnet-20240229, claude-3-opus-20240229)')
    query_parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the LLM')
    query_parser.add_argument('--output-format', type=str, choices=['text', 'json'], default='text', help='Output format')
    
    # Diagram command
    diagram_parser = subparsers.add_parser('diagram', help='Generate a sequence diagram for a business process')
    diagram_parser.add_argument('--process', type=str, required=True, help='Name or description of the business process')
    diagram_parser.add_argument('--module', type=str, help='Optional module to restrict the search to')
    diagram_parser.add_argument('--persist-dir', type=str, default='chroma_db', help='Directory where the vector store is persisted')
    diagram_parser.add_argument('--llm-model', type=str, default='claude-3-5-haiku-20241022',
                             help='Claude model to use (e.g., claude-3-5-haiku-20241022, claude-3-sonnet-20240229)')
    diagram_parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for the LLM')
    diagram_parser.add_argument('--output-file', type=str, help='Optional file to save the diagram to')
    
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

def generate_diagram(args: argparse.Namespace) -> None:
    """Generate a sequence diagram for a business process"""
    logger.info(f"Generating sequence diagram for process: {args.process}")
    
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
    
    # Generate the sequence diagram
    result = rag.generate_sequence_diagram(process_name=args.process, module_name=args.module)
    
    # Extract the diagram content
    diagram = result.get('result', 'Could not generate diagram.')
    
    # Display or save the diagram
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(diagram)
        logger.info(f"Diagram saved to {args.output_file}")
        print(f"Diagram saved to {args.output_file}")
    else:
        print("\nSequence Diagram:")
        print(diagram)
        
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
    print("  /modules: List all available modules")
    print("  /diagram <process_name>: Generate sequence diagram for business process")
    
    # Session state
    current_filter = None
    
    while True:
        # Show current filters
        filter_info = []
        if current_filter:
            filter_info.append(f"module: {current_filter.get('module')}")
            filter_info.append(f"model: {current_filter.get('model_name')}")
        
        if filter_info:
            print(f"\nActive filters: {', '.join(filter_info)}")
        
        # Get user input
        try:
            question = input("\nQuestion: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        
        if question.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        # Check for special commands
        if question.startswith('/'):
            parts = question.split()
            command = parts[0].lower()
            
            if command == '/clear':
                current_filter = None
                print("Filters cleared")
                continue
            elif command == '/module' and len(parts) > 1:
                module_name = parts[1]
                current_filter = {"module": module_name}
                print(f"Set filter to module: {module_name}")
                continue
            elif command == '/model' and len(parts) > 1:
                model_name = parts[1]
                current_filter = {"model_name": model_name}
                print(f"Set filter to model: {model_name}")
                continue
            elif command == '/modules':
                # Use the new list_all_modules method
                result = rag.list_all_modules()
                print("\n" + result["result"])
                continue
            elif command == '/diagram' and len(parts) > 1:
                # Generate a sequence diagram
                process_name = ' '.join(parts[1:])
                module_filter = current_filter.get('module') if current_filter else None
                result = rag.generate_sequence_diagram(process_name=process_name, module_name=module_filter)
                print("\nSequence Diagram:")
                print(result["result"])
                continue
        
        # Query the RAG system with the current filters
        if current_filter:
            result = rag.answer_about_module(question=question, module_name=current_filter.get('module'))
        else:
            result = rag.answer_question(question=question)
        
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
    elif args.command == 'diagram':
        generate_diagram(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 