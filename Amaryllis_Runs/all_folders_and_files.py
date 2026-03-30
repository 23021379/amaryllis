import os
import sys

def list_files_recursive(startpath):
    # Print the root directory first
    print(f"{startpath}/")
    
    for root, dirs, files in os.walk(startpath):
        # Calculate the depth to determine indentation
        level = root.replace(startpath, '').count(os.sep)
        indent = '    ' * (level + 1)
        
        # Sort directories and files for consistent output
        dirs.sort()
        files.sort()
        
        # Print directories
        for d in dirs:
            print(f"{indent}{d}/")
            
        # Print files inside the current root, but we need to ensure 
        # we aren't printing files for the subdirectories handled in the loop above.
        # Actually, os.walk goes deeper automatically.
        # The standard approach for a tree view using os.walk is slightly different
        # because os.walk is breadth-first per directory but depth-first overall.
        
        # A clearer visual approach is often better achieved ensuring files are printed
        # "under" their folders. The standard os.walk tuple (root, dirs, files) lists
        # files immediately under 'root'.
        
        subindent = '    ' * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def print_directory_tree(startpath):
    print(f"{os.path.abspath(startpath)}/")
    
    # We use os.walk, but we need to manipulate the logic to print visibly like a tree.
    # Standard os.walk travels top-down.
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = '    ' * (level)
        
        # Don't print the startpath again inside the loop since we printed it at the top
        if root != startpath:
            print(f"[REDACTED_BY_SCRIPT]")
        
        subindent = '    ' * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    # Change '.' to a specific path string if you want to scan a different folder
    # e.g., root_dir = r"c:/renewables"
    root_dir = os.getcwd() 
    output_filename = "directory_tree.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        try:
            print_directory_tree(root_dir)
        finally:
            sys.stdout = original_stdout
    print(f"[REDACTED_BY_SCRIPT]")