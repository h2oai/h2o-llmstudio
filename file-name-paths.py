import os

def find_markdown_files(folder_path):
    markdown_files = []
    # Walk through all files and directories in the folder recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a Markdown extension (.md or .markdown)
            if file.endswith('.md') or file.endswith('.markdown'):
                file_path = os.path.join(root, file)
                # Wrap the file path with '' and add a comma
                markdown_files.append(f"'{file_path}',")
    # Add a comma to the last element (to avoid extra newline)
    if markdown_files:
        markdown_files[-1] = markdown_files[-1].rstrip(",") + ","

    return markdown_files


# Example usage:
if __name__ == '__main__':
    folder_path = '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs'  # Replace with your folder path
    markdown_files = find_markdown_files(folder_path)
    print('Markdown files found:')
    # Print each file path on a separate line
    for file_path in markdown_files:
        print(file_path)
