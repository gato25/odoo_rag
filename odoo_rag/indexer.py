import os
import re
import ast
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
        
        # Read the file and evaluate it safely
        with open(manifest_path, 'r') as f:
            content = f.read()
        
        try:
            manifest_dict = ast.literal_eval(content)
            return manifest_dict
        except (SyntaxError, ValueError) as e:
            logger.error(f"Error parsing manifest for {module_name}: {e}")
            return {}
    
    def parse_model_file(self, file_path: Path) -> List[Dict]:
        """Parse a Python file containing Odoo model definitions"""
        models = []
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        try:
            # Parse the Python file
            tree = ast.parse(content)
            
            # Extract classes that inherit from Odoo models
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class inherits from an Odoo model
                    if self._is_odoo_model(node):
                        model_info = self._extract_model_info(node, file_path)
                        models.append(model_info)
        
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
        
        return models
    
    def _is_odoo_model(self, class_def: ast.ClassDef) -> bool:
        """Check if a class definition inherits from an Odoo model"""
        for base in class_def.bases:
            if isinstance(base, ast.Name) and base.id in ('Model', 'TransientModel', 'AbstractModel'):
                return True
            if isinstance(base, ast.Attribute):
                attr_str = f"{self._get_attribute_full_name(base)}"
                if attr_str in ('models.Model', 'models.TransientModel', 'models.AbstractModel'):
                    return True
        return False
    
    def _get_attribute_full_name(self, attr: ast.Attribute) -> str:
        """Get the full dotted name of an attribute"""
        parts = []
        current = attr
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _extract_model_info(self, class_def: ast.ClassDef, file_path: Path) -> Dict:
        """Extract model information from a class definition"""
        model_info = {
            'name': class_def.name,
            'file_path': str(file_path),
            'line_start': class_def.lineno,
            'line_end': class_def.end_lineno,
            '_name': None,  # Technical name, e.g. 'res.partner'
            '_inherit': None,  # Inherited model(s)
            '_description': None,  # User-friendly description
            'fields': [],
            'methods': []
        }
        
        # Extract class attributes and methods
        for node in class_def.body:
            # Extract class variables like _name, _inherit, etc.
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if targets:
                    name = targets[0]
                    if name in ('_name', '_inherit', '_description'):
                        if isinstance(node.value, ast.Constant):
                            model_info[name] = node.value.value
                    elif self._is_field_definition(node.value):
                        field_info = self._extract_field_info(name, node.value)
                        model_info['fields'].append(field_info)
            
            # Extract methods
            elif isinstance(node, ast.FunctionDef):
                method_info = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)]
                }
                model_info['methods'].append(method_info)
        
        return model_info
    
    def _is_field_definition(self, node: ast.AST) -> bool:
        """Check if an AST node represents an Odoo field definition"""
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == 'fields':
                    return True
        return False
    
    def _extract_field_info(self, name: str, call_node: ast.Call) -> Dict:
        """Extract field information from a fields.X() call"""
        field_info = {
            'name': name,
            'type': None,
            'string': None,
            'required': False,
            'readonly': False,
            'comodel_name': None,  # For relational fields
        }
        
        # Get field type
        if isinstance(call_node.func, ast.Attribute):
            field_info['type'] = call_node.func.attr
        
        # Extract keyword arguments
        for kw in call_node.keywords:
            if kw.arg in field_info:
                if isinstance(kw.value, ast.Constant):
                    field_info[kw.arg] = kw.value.value
        
        return field_info
    
    def parse_view_file(self, file_path: Path) -> List[Dict]:
        """Parse an XML view file to extract view definitions"""
        views = []
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Find all record elements that define views
            for record in root.findall(".//record"):
                model = record.get('model')
                if model == 'ir.ui.view':
                    view_info = {
                        'id': record.get('id'),
                        'name': None,
                        'model': None,
                        'type': None,
                        'arch': None,
                        'inherit_id': None,
                        'file_path': str(file_path)
                    }
                    
                    # Extract field values
                    for field in record.findall('field'):
                        field_name = field.get('name')
                        if field_name in view_info:
                            if field_name == 'arch' and field.get('type') == 'xml':
                                view_info[field_name] = ET.tostring(field[0], encoding='unicode') if len(field) > 0 else None
                            else:
                                view_info[field_name] = field.text
                    
                    views.append(view_info)
        
        except Exception as e:
            logger.error(f"Error parsing view file {file_path}: {e}")
        
        return views
    
    def index_module(self, module_name: str) -> Dict:
        """Index a single module and return its structure"""
        module_path = self.modules_path / module_name
        module_data = {
            'name': module_name,
            'path': str(module_path),
            'manifest': self.parse_manifest(module_name),
            'models': [],
            'views': [],
            'scripts': [],
            'data': []
        }
        
        # Parse Python files for models
        model_dir = module_path / 'models'
        if model_dir.exists():
            for py_file in model_dir.glob('*.py'):
                module_data['models'].extend(self.parse_model_file(py_file))
        
        # Parse XML files for views
        for xml_file in module_path.glob('**/*.xml'):
            module_data['views'].extend(self.parse_view_file(xml_file))
        
        # Index JavaScript files
        for js_file in module_path.glob('**/*.js'):
            module_data['scripts'].append({
                'path': str(js_file),
                'name': js_file.name
            })
        
        # Index CSV/data files
        for data_file in module_path.glob('**/*.csv'):
            module_data['data'].append({
                'path': str(data_file),
                'name': data_file.name,
                'type': 'csv'
            })
            
        self.modules[module_name] = module_data
        return module_data
    
    def index_all_modules(self) -> Dict:
        """Index all discovered modules"""
        modules = self.discover_modules()
        for module_name in modules:
            self.index_module(module_name)
        return self.modules
    
    def extract_chunks_for_embedding(self) -> List[Dict]:
        """Extract chunks of code/content suitable for embedding"""
        chunks = []
        
        for module_name, module_data in self.modules.items():
            # Add module manifest as a chunk
            manifest_str = str(module_data['manifest'])
            chunks.append({
                'content': f"Module: {module_name}\nManifest: {manifest_str}",
                'metadata': {
                    'module': module_name,
                    'type': 'manifest',
                    'path': module_data['path']
                }
            })
            
            # Add model definitions as chunks
            for model in module_data['models']:
                model_content = f"Model: {model.get('_name') or model['name']}\n"
                model_content += f"Class: {model['name']}\n"
                
                if model.get('_description'):
                    model_content += f"Description: {model['_description']}\n"
                
                if model.get('_inherit'):
                    model_content += f"Inherits: {model['_inherit']}\n"
                
                # Add fields
                if model['fields']:
                    model_content += "Fields:\n"
                    for field in model['fields']:
                        model_content += f"  - {field['name']} ({field['type']}): {field.get('string')}\n"
                
                # Add methods
                if model['methods']:
                    model_content += "Methods:\n"
                    for method in model['methods']:
                        model_content += f"  - {method['name']}"
                        if method['decorators']:
                            model_content += f" (decorated with: {', '.join(method['decorators'])})"
                        model_content += "\n"
                
                chunks.append({
                    'content': model_content,
                    'metadata': {
                        'module': module_name,
                        'type': 'model',
                        'model_name': model.get('_name') or model['name'],
                        'path': model['file_path'],
                        'line_start': model['line_start'],
                        'line_end': model['line_end']
                    }
                })
            
            # Add views as chunks
            for view in module_data['views']:
                view_content = f"View: {view.get('name')}\n"
                view_content += f"Model: {view.get('model')}\n"
                view_content += f"Type: {view.get('type')}\n"
                
                if view.get('inherit_id'):
                    view_content += f"Inherits view: {view.get('inherit_id')}\n"
                
                if view.get('arch'):
                    view_content += f"Architecture:\n{view['arch']}"
                
                chunks.append({
                    'content': view_content,
                    'metadata': {
                        'module': module_name,
                        'type': 'view',
                        'view_id': view.get('id'),
                        'path': view['file_path']
                    }
                })
        
        return chunks

if __name__ == "__main__":
    # Example usage
    parser = OdooModuleParser("./addons")
    parser.index_all_modules()
    chunks = parser.extract_chunks_for_embedding()
    print(f"Generated {len(chunks)} chunks for embedding") 