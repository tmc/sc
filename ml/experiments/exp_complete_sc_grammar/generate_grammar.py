import re
import json
import os
from typing import Dict, Any, List, Optional

# Constants
PROTO_DIR = "../../../proto/statecharts/v1"
OUTPUT_FILE = "full_grammar.json"
STATS_FILE = "grammar_stats.py"
SAMPLER_CONFIG_FILE = "sampler_config.py"

def parse_proto_file(filepath: str) -> Dict[str, Any]:
    """
    Parses a .proto file and extracts message and enum definitions.
    Returns a dictionary of definitions.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    definitions = {
        "messages": {},
        "enums": {}
    }

    # Remove comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Parse Enums
    enum_matches = re.finditer(r'enum\s+(\w+)\s*{([^}]*)}', content, re.DOTALL)
    for match in enum_matches:
        enum_name = match.group(1)
        body = match.group(2)
        values = []
        for line in body.split(';'):
            line = line.strip()
            if '=' in line:
                key = line.split('=')[0].strip()
                values.append(key)
        definitions["enums"][enum_name] = values

    # Parse Messages
    # This is a simplified parser; handles nested messages by extracting them first if needed, 
    # but here we assume flat or top-level mainly.
    # We will do a recursive-like regex or just find all top-level messages.
    # Proto messages can be nested, but for SC proto they are mostly top-level.
    
    msg_matches = re.finditer(r'message\s+(\w+)\s*{', content)
    
    # We need to extract the body. A simple regex for body is hard due to nesting braces.
    # We'll use a brace counter approach.
    
    def extract_block(start_idx, text):
        brace_count = 0
        found_start = False
        body_start = -1
        
        for i in range(start_idx, len(text)):
            char = text[i]
            if char == '{':
                if not found_start:
                    found_start = True
                    brace_count = 1
                    body_start = i + 1
                else:
                    brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[body_start:i], i + 1
        return "", -1

    cursor = 0
    # Re-read to handle brace counting manually
    while True:
        match = re.search(r'message\s+(\w+)\s*', content[cursor:])
        if not match:
            break
        
        msg_name = match.group(1)
        start_brace_search = cursor + match.end()
        # Find the opening brace
        open_brace_idx = content.find('{', start_brace_search)
        if open_brace_idx == -1:
            break
            
        body, end_idx = extract_block(open_brace_idx, content)
        cursor = end_idx
        
        fields = []
        # Parse fields: type name = id;
        # repeated type name = id;
        # map<k,v> name = id;
        
        # Split by ;
        for line in body.split(';'):
            line = line.strip()
            if not line: continue
            
            # Check for map
            map_match = re.match(r'map\s*<\s*(\w+)\s*,\s*(\w+)\s*>\s+(\w+)\s*=', line)
            if map_match:
                fields.append({
                    "name": map_match.group(3),
                    "type": "map",
                    "key_type": map_match.group(1),
                    "value_type": map_match.group(2),
                    "repeated": False
                })
                continue

            # Standard fields
            # repeated Type name = ID [options];
            parts = line.split('=')[0].strip().split()
            if not parts: continue
            
            is_repeated = False
            field_type = ""
            field_name = ""
            
            if parts[0] == 'repeated':
                is_repeated = True
                field_type = parts[1]
                field_name = parts[2]
            else:
                # Handle optional/required (in proto3 just type name)
                # But could be "optional Type name"
                if parts[0] in ['optional', 'required']:
                     field_type = parts[1]
                     field_name = parts[2]
                else:
                     field_type = parts[0]
                     field_name = parts[1]
            
            fields.append({
                "name": field_name,
                "type": field_type,
                "repeated": is_repeated
            })
            
        definitions["messages"][msg_name] = fields

    return definitions

def build_grammar(sc_defs: Dict, expr_defs: Dict) -> Dict:
    """
    Constructs the JSON grammar based on parsed definitions.
    """
    # Merge definitions
    messages = {**sc_defs["messages"], **expr_defs["messages"]}
    enums = {**sc_defs["enums"], **expr_defs["enums"]}
    
    grammar = {
        "root": "Statechart",
        "definitions": {}
    }

    # Helper maps
    type_mapping = {
        "string": {"type": "string"},
        "bool": {"type": "boolean"},
        "int32": {"type": "integer"},
        "int64": {"type": "integer"},
        "bytes": {"type": "string", "encoding": "base64"}, # Simplified
        "google.protobuf.Struct": {"type": "object", "additionalProperties": True},
        "google.protobuf.Any": {"type": "object", "additionalProperties": True}
    }

    def process_field(field):
        ftype = field['type']
        schema = {}

        if ftype in type_mapping:
            schema = type_mapping[ftype].copy()
        elif ftype in messages:
            schema = {"$ref": f"#/definitions/{ftype}"}
        elif ftype in enums:
            schema = {"type": "string", "enum": enums[ftype]}
        elif ftype == "map":
             schema = {
                 "type": "object",
                 "additionalProperties": get_type_schema(field['value_type'])
             }
        else:
            # Fallback for imported types or primitives not mapped
            schema = {"type": "string", "description": f"Unknown type: {ftype}"}
        
        if field['repeated']:
            return {"type": "array", "items": schema}
        return schema

    def get_type_schema(typename):
        # Re-use process_field logic for simple type lookups
        return process_field({"type": typename, "repeated": False})

    for msg_name, fields in messages.items():
        # SC Grammar specific constraint:
        # We want to represent the message as an object properties
        properties = {}
        required = [] # Proto3 fields are optional, but for grammar generation we might enforce some structural fields
        
        for f in fields:
            properties[f['name']] = process_field(f)
            
        def_schema = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False
        }
        grammar["definitions"][msg_name] = def_schema

    # Refine Specific Constraints (Manual Overrides for "Constraints")
    # 1. Statechart root needs root_state
    if "Statechart" in grammar["definitions"]:
        grammar["definitions"]["Statechart"]["required"] = ["root_state", "events"]
    
    # 2. State needs label and type
    if "State" in grammar["definitions"]:
        grammar["definitions"]["State"]["required"] = ["label", "type"]
    
    # 3. Transition needs event, from, to
    if "Transition" in grammar["definitions"]:
         grammar["definitions"]["Transition"]["required"] = ["label", "from", "to", "event"]

    return grammar

def main():
    sc_proto_path = os.path.join(PROTO_DIR, "statecharts.proto")
    expr_proto_path = os.path.join(PROTO_DIR, "expressions.proto")
    
    print(f"Parsing {sc_proto_path}...")
    sc_defs = parse_proto_file(sc_proto_path)
    
    print(f"Parsing {expr_proto_path}...")
    expr_defs = parse_proto_file(expr_proto_path)
    
    print("Building Grammar...")
    grammar = build_grammar(sc_defs, expr_defs)
    
    # Save grammar
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(grammar, f, indent=2)
    print(f"Saved grammar to {OUTPUT_FILE}")

    # Calculate Stats
    num_states = len(grammar["definitions"])
    num_refs = str(grammar).count("$ref")
    
    stats_content = f"""
# Grammar Coverage Stats
TOTAL_DEFINITIONS = {num_states}
TOTAL_REFERENCES = {num_refs}
"""
    with open(STATS_FILE, 'w') as f:
        f.write(stats_content)
        
    # Generate Sampler Config (Simple)
    config_content = f"""
# Constrained Sampler Configuration
GRAMMAR_PATH = "{os.path.abspath(OUTPUT_FILE)}"
ROOT_TYPE = "Statechart"
"""
    with open(SAMPLER_CONFIG_FILE, 'w') as f:
        f.write(config_content)
        
    print(f"Report: [F624]: COMPLETE_SC_GRAMMAR states={num_states}, transitions={num_refs} (approx refs)")

if __name__ == "__main__":
    main()
