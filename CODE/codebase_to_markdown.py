import os

def create_markdown_from_codebase(root_dir, output_file):
    """
    Traverses the directory tree starting from root_dir, identifies all Python files,
    and consolidates their content into a single Markdown file.
    Each file section is prefixed with its full path and wrapped in a code block.
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    full_path = os.path.join(dirpath, filename)
                    
                    # Optional: Skip the output file itself if it happens to be a .py file (unlikely for .md)
                    # or skip this script itself if desired. Currently including everything as requested.
                    
                    try:
                        with open(full_path, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()
                            
                        # Write the full path
                        md_file.write(f"{full_path}\n")
                        # Write the code block start
                        md_file.write("```\n")
                        # Write the file content
                        md_file.write(content)
                        # Ensure there's a newline before closing the block
                        if content and not content.endswith('\n'):
                            md_file.write('\n')
                        # Write the code block end
                        md_file.write("```\n\n")
                        
                        print(f"[REDACTED_BY_SCRIPT]")
                        
                    except Exception as e:
                        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    # define the root directory as the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output file name
    output_md_path = os.path.join(current_dir, 'full_codebase.md')
    
    print(f"[REDACTED_BY_SCRIPT]")
    print(f"[REDACTED_BY_SCRIPT]")
    
    create_markdown_from_codebase(current_dir, output_md_path)
    
    print("Done!")
