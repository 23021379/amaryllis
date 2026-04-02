
import os

def extract_python_scripts(root_dir, output_file):
    # Get the absolute path of the current script and output file to avoid including them
    current_script = os.path.abspath(__file__)
    output_file_abs = os.path.abspath(output_file)

    with open(output_file, 'w', encoding='utf-8') as md_file:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    abs_path = os.path.abspath(file_path)

                    # Skip this script and the output file (though output file usually isn't .py)
                    if abs_path == current_script:
                        continue
                    
                    # Calculate relative path for display
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as source_file:
                            content = source_file.read()
                            
                        # Write to markdown
                        md_file.write(f"{relative_path}\n")
                        md_file.write("```\n")
                        md_file.write(content)
                        if not content.endswith('\n'):
                            md_file.write('\n')
                        md_file.write("```\n\n")
                        
                    except Exception as e:
                        print(f"[REDACTED_BY_SCRIPT]")

if __name__ == "__main__":
    extract_python_scripts('.', 'codebase_summary.md')
    print("[REDACTED_BY_SCRIPT]")
