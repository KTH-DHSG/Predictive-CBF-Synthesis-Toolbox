"""

    This script defines a function to count the total number of lines in all files within a specified folder, optionally filtering by file extension. It then demonstrates the usage of this function by counting the lines of Python code in the current directory.

    This script is useful for quickly assessing the size of a codebase.

    Automatically generated code by Adrian Wiltz, 2025
    
"""

import os

def count_lines_in_folder(folder_path, file_extension=None):
    total_lines = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file_extension and not file.endswith(file_extension):
                continue
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines += sum(1 for _ in f)
    return total_lines

# Example usage
folder_path = "./"  # Change to your folder path
file_extension = ".py"  # Set to None for all file types
lines = count_lines_in_folder(folder_path, file_extension)
print(f"Total lines of code: {lines}")
print(f"Considered folder: {folder_path}")