# Odoo RAG System

A Retrieval-Augmented Generation (RAG) system specifically designed for Odoo ERP custom modules. This tool helps developers understand, navigate, and work with Odoo codebases by providing AI-powered answers to questions about the code.

## Features

- **Smart indexing** of Odoo modules, including:
  - Python models and fields
  - XML views and their inheritance structure
  - JavaScript files and data files
- **Semantic search** using embeddings for accurate retrieval
- **Specialized prompts** for different types of Odoo questions
- **Context-aware answers** that understand Odoo's architecture
- **Interactive CLI** with filtering by module or model
- **Powered by Claude AI** and **LlamaIndex** for high-quality responses

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/odoo_rag.git
   cd odoo_rag
   ```

2. Create a virtual environment and install the package:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your-anthropic-api-key-here
   ```

## Usage

### Indexing Odoo Modules

Before you can query the system, you need to index your Odoo modules:

```bash
odoo-rag index --modules-path /path/to/your/odoo/addons
```

This will create a vector database in the `chroma_db` directory by default.

### Querying the System

You can ask questions about your Odoo modules in several ways:

1. One-off query:
   ```bash
   odoo-rag query --question "How is the invoice workflow implemented?"
   ```

2. Query about a specific module:
   ```bash
   odoo-rag query --question "What fields are defined in the partner model?" --module sale
   ```

3. Query about a specific model:
   ```bash
   odoo-rag query --question "What are the computed fields?" --model res.partner
   ```

### Interactive Mode

For multiple questions in a session, use interactive mode:

```bash
odoo-rag interactive
```

In interactive mode, you can use special commands:
- `/module sale`: Set a module filter
- `/model res.partner`: Set a model filter
- `/clear`: Clear all filters

## Examples

### Understanding Model Definitions

```
Question: How is the invoicing flow implemented in the sale module?
```

### Exploring Field Definitions

```
Question: What fields are tracked for changes in the CRM lead model?
```

### Understanding View Inheritance

```
Question: How is the partner form view customized in the CRM module?
```

## Advanced Configuration

You can customize various aspects of the system:

- **Embedding model**: `--embedding-model sentence-transformers/all-mpnet-base-v2`
- **Claude model**: `--llm-model claude-3-opus-20240229`
- **Vector store location**: `--persist-dir ./my_vectors`

## Technology Stack

This RAG system is built using:
- **LlamaIndex**: For document indexing, retrieval, and query processing
- **Claude by Anthropic**: For generating high-quality responses
- **HuggingFace Embeddings**: For semantic understanding of code
- **ChromaDB**: For efficient vector storage

## Available Claude Models

The system supports the following Claude models:
- `claude-3-sonnet-20240229` (default) - Fast and cost-effective
- `claude-3-opus-20240229` - Most powerful for complex code understanding
- `claude-3-haiku-20240307` - Fastest for simpler queries

## Development

To set up a development environment:

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 