import os
import re

def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix Optional[Mapped[X]] -> Mapped[Optional[X]]
    # It might also have spaces or extra brackets depending on how the regex messed it up.
    # Let's just catch Optional[Mapped[...]]
    content = re.sub(r'Optional\[\s*Mapped\[([^\]]+)\]\s*\]', r'Mapped[Optional[\1]]', content)

    # Also catch cases where there might be a missing bracket? 
    # Optional[Mapped[str]]] -> Mapped[Optional[str]]] which has an extra bracket at the end.
    content = re.sub(r'Optional\[\s*Mapped\[([^\]]+)\]\s*\]\]', r'Mapped[Optional[\1]]]', content)
    
    # Let's fix `Optional[Mapped[X]]` safely:
    content = re.sub(r'Optional\[\s*Mapped\[([^\]]+)\]\s*\]', r'Mapped[Optional[\1]]', content)

    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {path}")

def main():
    app_dir = 'app'
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith('.py'):
                process_file(os.path.join(root, file))

if __name__ == '__main__':
    main()
