from setuptools import setup, find_packages

setup(
    name="odoo_rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chromadb==0.4.17",
        "anthropic==0.49.0",
        "python-dotenv==1.0.0",
        "sentence-transformers==2.2.2",
        "requests==2.31.0",
        "tqdm==4.66.1",
        "huggingface-hub==0.16.4",
        "torch>=1.6.0",
    ],
    entry_points={
        "console_scripts": [
            "odoo-rag=odoo_rag.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="RAG system for Odoo ERP custom modules",
    keywords="odoo, rag, ai, claude",
    python_requires=">=3.8",
) 