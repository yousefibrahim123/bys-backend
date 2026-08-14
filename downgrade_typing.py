import os
import re

def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Add typing imports if needed
    if 'from __future__ import annotations' in content:
        if 'from typing import ' not in content:
            content = content.replace('from __future__ import annotations', 
                                      'from __future__ import annotations\nfrom typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal', 1)
        else:
            if 'List' not in content or 'Optional' not in content:
                content = content.replace('from typing import ', 'from typing import List, Dict, Tuple, Set, Optional, Union, Any, Type, Literal, ')

    # Fix list[, dict[, tuple[, set[, type[
    content = re.sub(r'\blist\[', 'List[', content)
    content = re.sub(r'\bdict\[', 'Dict[', content)
    content = re.sub(r'\btuple\[', 'Tuple[', content)
    content = re.sub(r'\bset\[', 'Set[', content)
    content = re.sub(r'\btype\[', 'Type[', content)

    # Fix X | None -> Optional[X]
    # Handle cases like `: str | None`, `-> str | None`, `[str | None]`
    content = re.sub(r':\s*([A-Za-z0-9_\[\]\"\',\s]+?)\s*\|\s*None', r': Optional[\1]', content)
    content = re.sub(r'->\s*([A-Za-z0-9_\[\]\"\',\s]+?)\s*\|\s*None', r'-> Optional[\1]', content)
    content = re.sub(r'\[\s*([A-Za-z0-9_\[\]\"\',\s]+?)\s*\|\s*None', r'[Optional[\1]', content)

    # Fix any reversed ones: None | X -> Optional[X]
    content = re.sub(r':\s*None\s*\|\s*([A-Za-z0-9_\[\]\"\',\s]+)', r': Optional[\1]', content)
    content = re.sub(r'->\s*None\s*\|\s*([A-Za-z0-9_\[\]\"\',\s]+)', r'-> Optional[\1]', content)
    content = re.sub(r'\[\s*None\s*\|\s*([A-Za-z0-9_\[\]\"\',\s]+)', r'[Optional[\1]', content)

    # A special case for the auth router / security (e.g. `str | None = Depends(...)`)
    content = re.sub(r'\(\s*([A-Za-z0-9_\[\]\"\',\s]+?)\s*\|\s*None', r'(Optional[\1]', content)
    
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {path}")

def main():
    app_dir = 'app'
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith('.py'):
                process_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
