import os

def combine_python_scripts(script_paths, output_file):
    """
    Reads multiple Python scripts and combines their contents into a single text file.

    Args:
        script_paths (list): A list of strings, where each string is a path to a Python script.
        output_file (str): The path to the output text file.
    """
    combined_content = []
    for script_path in script_paths:
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                # Add a separator with the original filename
                combined_content.append(f'# {"-"[REDACTED_BY_SCRIPT]"-"*20}\n\n')
                combined_content.append(f.read())
                combined_content.append(f'\n\n# {"-"[REDACTED_BY_SCRIPT]"-"*20}\n\n')
        except FileNotFoundError:
            print(f"Warning: The file '{script_path}'[REDACTED_BY_SCRIPT]")
        except Exception as e:
            print(f"[REDACTED_BY_SCRIPT]'{script_path}': {e}")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("".join(combined_content))
        print(f"[REDACTED_BY_SCRIPT]'{output_file}'.")
    except Exception as e:
        print(f"[REDACTED_BY_SCRIPT]'{output_file}': {e}")

if __name__ == '__main__':
    # --- User Configuration ---
    # 1. Add the paths to your Python scripts here
    #    Example for Windows: '[REDACTED_BY_SCRIPT]'
    #    Example for macOS/Linux: '[REDACTED_BY_SCRIPT]'
    paths_to_scripts = [
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]",
        "[REDACTED_BY_SCRIPT]"
    ]

    # 2. Specify the name of the output file
    output_filename = '[REDACTED_BY_SCRIPT]'
    # --- End of User Configuration ---

    if not paths_to_scripts:
        print("[REDACTED_BY_SCRIPT]'paths_to_scripts' list.")
    else:
        combine_python_scripts(paths_to_scripts, output_filename)