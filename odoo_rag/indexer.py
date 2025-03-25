import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OdooModuleParser:
    """Parser for Odoo modules to extract structured information for RAG system"""
    
    def __init__(self, modules_path: str):
        """Initialize with path to Odoo modules directory"""
        self.modules_path = Path(modules_path)
        self.modules = {}  # Will store parsed module data
        
    def discover_modules(self) -> List[str]:
        """Find all Odoo modules in the specified directory"""
        modules = []
        for item in os.listdir(self.modules_path):
            module_path = self.modules_path / item
            if module_path.is_dir() and (module_path / '__manifest__.py').exists():
                modules.append(item)
        logger.info(f"Discovered {len(modules)} Odoo modules")
        return modules
    
    def parse_manifest(self, module_name: str) -> Dict:
        """Extract information from the module manifest"""
        manifest_path = self.modules_path / module_name / '__manifest__.py'
        if not manifest_path.exists():
            return {}
        
        # Read the file content
        with open(manifest_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a simple metadata dict (without evaluating the Python code)
        manifest_dict = {
            'raw_content': content,
            'file_path': str(manifest_path)
        }
        
        # Extract basic info with regex
        name_match = re.search(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
        if name_match:
            manifest_dict['name'] = name_match.group(1)
        
        version_match = re.search(r"['\"]version['\"]\s*:\s*['\"]([^'\"]+)['\"]", content)
        if version_match:
            manifest_dict['version'] = version_match.group(1)
            
        depends_match = re.search(r"['\"]depends['\"]\s*:\s*\[(.*?)\]", content, re.DOTALL)
        if depends_match:
            # Extract module names from depends list
            depends_str = depends_match.group(1)
            depends = re.findall(r"['\"]([^'\"]+)['\"]", depends_str)
            manifest_dict['depends'] = depends
            
        return manifest_dict
    
    def extract_file_content(self, file_path: Path, file_type: str) -> Dict:
        """Extract the raw content from a file with basic metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Get the module name from the path
            parts = file_path.parts
            module_index = next((i for i, part in enumerate(parts) if part in self.modules), -1)
            module_name = parts[module_index] if module_index >= 0 else "unknown"
            
            # Create a simple content dict
            content_dict = {
                'content': content,
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': file_type,
                'module': module_name,
                'size': len(content)
            }
            
            # Add type-specific metadata
            if file_type == 'python':
                # Basic regex extraction of classes and models
                model_match = re.search(r"_name\s*=\s*['\"]([^'\"]+)['\"]", content)
                if model_match:
                    content_dict['model_name'] = model_match.group(1)
                
                # Extract class definitions
                class_defs = re.findall(r"class\s+(\w+)\s*\(([^)]+)\):", content)
                content_dict['classes'] = [
                    {'name': cls_name, 'base': base.strip()} 
                    for cls_name, base in class_defs
                ]
                
            elif file_type == 'xml':
                # For XML files, try to identify view types, models, etc.
                if '<record' in content and 'ir.ui.view' in content:
                    content_dict['has_views'] = True
                if '<template' in content:
                    content_dict['has_templates'] = True
                if '<menuitem' in content:
                    content_dict['has_menus'] = True
                    
                # Try to extract model information from views
                model_matches = re.findall(r"model=['\"]([^'\"]+)['\"]", content)
                if model_matches:
                    content_dict['referenced_models'] = list(set(model_matches))
            
            return content_dict
            
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return {
                'content': f"Error: {e}",
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': file_type,
                'error': str(e)
            }
    
    def index_module(self, module_name: str) -> Dict:
        """Index a single module and return its structure"""
        module_path = self.modules_path / module_name
        module_data = {
            'name': module_name,
            'path': str(module_path),
            'manifest': self.parse_manifest(module_name),
            'files': []
        }
        
        # Walk through the module directory and process all files
        for root, _, files in os.walk(module_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(module_path)
                
                # Skip some files we don't want to index
                if file.startswith('.') or file.endswith('.pyc'):
                    continue
                
                # Determine file type
                if file.endswith('.py'):
                    file_type = 'python'
                elif file.endswith('.xml'):
                    file_type = 'xml'
                elif file.endswith('.js'):
                    file_type = 'javascript'
                elif file.endswith('.css'):
                    file_type = 'css'
                elif file.endswith('.scss'):
                    file_type = 'scss'
                elif file.endswith('.csv'):
                    file_type = 'csv'
                else:
                    file_type = 'other'
                
                # Extract content
                content_dict = self.extract_file_content(file_path, file_type)
                content_dict['relative_path'] = str(relative_path)
                module_data['files'].append(content_dict)
        
        logger.info(f"Indexed module {module_name} with {len(module_data['files'])} files")
        self.modules[module_name] = module_data
        return module_data
    
    def index_all_modules(self) -> Dict:
        """Index all discovered modules"""
        modules = self.discover_modules()
        for module_name in modules:
            self.index_module(module_name)
        return self.modules
    
    def create_markdown_chunk(self, content_dict: Dict) -> str:
        """Create a Markdown representation of a content chunk"""
        file_path = content_dict.get('file_path', 'unknown')
        file_type = content_dict.get('file_type', 'unknown')
        module = content_dict.get('module', 'unknown')
        content = content_dict.get('content', '')
        
        # Create header with metadata
        header = f"# {os.path.basename(file_path)}\n\n"
        header += f"**Module:** {module}\n"
        header += f"**Path:** {file_path}\n"
        header += f"**Type:** {file_type}\n"
        
        # Add model info if available
        if 'model_name' in content_dict:
            header += f"**Model:** {content_dict['model_name']}\n"
        
        # Add class info if available
        if 'classes' in content_dict and content_dict['classes']:
            header += "\n**Classes:**\n"
            for cls in content_dict['classes']:
                header += f"- {cls['name']} (inherits {cls['base']})\n"
        
        # Add referenced models if available
        if 'referenced_models' in content_dict and content_dict['referenced_models']:
            header += "\n**Referenced Models:**\n"
            for model in content_dict['referenced_models']:
                header += f"- {model}\n"
        
        # Add a divider
        header += "\n---\n\n"
        
        # Add code block with appropriate syntax highlighting
        code_block = f"```{file_type}\n{content}\n```"
        
        return header + code_block
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split a long content into overlapping chunks"""
        chunks = []
        
        # If content is shorter than chunk_size, return it as is
        if len(content) <= chunk_size:
            return [content]
        
        # Split content into chunks with overlap
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    def extract_chunks_for_embedding(self, chunk_size: int = 2000, overlap: int = 400) -> List[Dict]:
        """Extract chunks of code/content suitable for embedding"""
        chunks = []
        
        for module_name, module_data in self.modules.items():
            # Add manifest as a chunk
            manifest = module_data.get('manifest', {})
            if manifest and 'raw_content' in manifest:
                manifest_content = self.create_markdown_chunk({
                    'file_path': manifest.get('file_path', f"{module_name}/__manifest__.py"),
                    'file_type': 'python',
                    'module': module_name,
                    'content': manifest.get('raw_content', ''),
                    'type': 'manifest'
                })
                
                chunks.append({
                    'content': manifest_content,
                    'metadata': {
                        'module': module_name,
                        'type': 'manifest',
                        'file_path': manifest.get('file_path', f"{module_name}/__manifest__.py")
                    }
                })
            
            # Process each file in the module
            for file_dict in module_data.get('files', []):
                # Create Markdown representation
                md_content = self.create_markdown_chunk(file_dict)
                
                # Split into chunks if needed
                if len(md_content) > chunk_size:
                    content_chunks = self.chunk_content(md_content, chunk_size, overlap)
                    
                    for i, content_chunk in enumerate(content_chunks):
                        chunks.append({
                            'content': content_chunk,
                            'metadata': {
                                'module': module_name,
                                'type': file_dict.get('file_type', 'unknown'),
                                'file_path': file_dict.get('file_path', 'unknown'),
                                'chunk_index': i,
                                'total_chunks': len(content_chunks),
                                'model_name': file_dict.get('model_name', None)
                            }
                        })
                else:
                    # Add as a single chunk
                    chunks.append({
                        'content': md_content,
                        'metadata': {
                            'module': module_name,
                            'type': file_dict.get('file_type', 'unknown'),
                            'file_path': file_dict.get('file_path', 'unknown'),
                            'model_name': file_dict.get('model_name', None)
                        }
                    })
        
        logger.info(f"Generated {len(chunks)} chunks for embedding")
        return chunks

if __name__ == "__main__":
    # Example usage
    parser = OdooModuleParser("./addons")
    parser.index_all_modules()
    chunks = parser.extract_chunks_for_embedding()
    print(f"Generated {len(chunks)} chunks for embedding") 