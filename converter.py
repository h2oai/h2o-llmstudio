import os
import subprocess
import re

def combine_markdown_files(input_files, combined_file):
    try:
        combined_md_file = 'combined_docs.md'

        with open(combined_md_file, 'w', encoding='utf-8') as outfile:
            outfile.write('\\newpage\n\n')  # Add a LaTeX page break after the additional content

            for file_path in input_files:
                if file_path.endswith('.md'):
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read().strip()
                        # Replace dynamic tags with actual MDX content and remove import statements
                        content = replace_dynamic_tags(content, file_path)
                        outfile.write(content + '\n\n\\newpage\n\n')  # Add a LaTeX page break after each Markdown file

        print(f'Combined {len(input_files)} Markdown files into {combined_md_file}')
        return combined_md_file

    except Exception as e:
        print(f'Error: Failed to combine Markdown files into {combined_file}.')
        print(e)
        raise

def replace_dynamic_tags(content, markdown_path):
    try:
        # Regex pattern to match dynamic tags like <Dataset />
        pattern = r'<(\w+)\/>'

        # Find all occurrences of dynamic tags in the content
        matches = re.findall(pattern, content)

        for tag_name in matches:
            # Construct the import statement pattern to find corresponding MDX file
            import_pattern = rf'import\s+{tag_name}\s+from\s+[\'"]([^\'"]+)[\'"]\s*;'
            import_match = re.search(import_pattern, content)
            
            if import_match:
                mdx_file_path = os.path.join(os.path.dirname(markdown_path), import_match.group(1))
                
                if os.path.isfile(mdx_file_path):
                    with open(mdx_file_path, 'r', encoding='utf-8') as mdxfile:
                        mdx_content = mdxfile.read().strip()
                        # Replace <Tag /> with actual MDX content
                        content = content.replace(f'<{tag_name}/>', mdx_content)
                        # Remove the entire import statement from the content
                        content = re.sub(import_pattern, '', content)
                else:
                    print(f"Warning: MDX file '{mdx_file_path}' referenced in '{markdown_path}' not found.")
            else:
                print(f"Warning: Import statement not found for '{tag_name}' tag in '{markdown_path}'.")

        return content

    except Exception as e:
        print(f'Error: Failed to replace dynamic tags in {markdown_path}.')
        print(e)
        return content


def convert_markdown_links(content):
    try:
        # Regex pattern to match Markdown URLs and extract anchor text
        pattern = r'\[([^\]]+)\]\(([^)\n]+)\.md\)'
        modified_content = re.sub(pattern, lambda match: f"[{match.group(1)}](#{extract_anchor(match.group(2))})", content)

        return modified_content

    except Exception as e:
        print(f'Error: Failed to convert Markdown links.')
        print(e)

def extract_anchor(url):
    parts = url.split('/')
    return parts[-1]

def remove_alt_text_from_images(content):
    try:
        # Regex pattern to remove alt text from image links, excluding those followed by ](#
        pattern = r'!\[([^\]]*)\]\((?!\]\()([^)]+)\)'

        # Replace matches with simplified image syntax
        modified_content = re.sub(pattern, r'![](\2)', content)

        return modified_content

    except Exception as e:
        print(f'Error: Failed to remove alt text from images.')
        print(e)
        return content  # Return original content if there's an error

def remove_info_caution_tip_tags(content):
    try:
        # Remove ":::info", ":::caution", ":::tip" tags 
        modified_content = re.sub(r':::info\s+(Note|note)', '- **Note:**', content, flags=re.IGNORECASE)
        modified_content = re.sub(r':::info', '- **Note:**', modified_content)
        modified_content = re.sub(r':::caution', '**Caution:**', modified_content)
        modified_content = re.sub(r':::tip', '**Tip:**', modified_content)
        modified_content = re.sub(r':::note\s+(Examples|examples)', '**Examples:**', modified_content)
        modified_content = re.sub(r':::', '', modified_content)

        return modified_content

    except Exception as e:
        print(f'Error: Failed to remove ":::" tags.')
        print(e)

def remove_import_statements(content):
    try:
        # Identify and remove the specific imports
        modified_content = re.sub(r'import (Tabs|TabItem|Icon|Admonition) from .*', '', content)

        return modified_content

    except Exception as e:
        print(f'Error: Failed to remove import statements.')
        print(e)

def remove_and_replace_html_tags(content):
    # Define the regex pattern to find the <Tabs>...</Tabs> block
    tabs_pattern = re.compile(r'<Tabs(?: className="unique-tabs")?>(.*?)</Tabs>', re.DOTALL)
    
    # Function to replace HTML tags with specific markdown equivalents
    def replace_html_tags(text):
        # Replace <br> tags with an empty string
        text = re.sub(r'<br\s*/?>', '', text)
        # Replace <b> tags with **
        text = re.sub(r'<b>(.*?)</b>', r'**\1**', text)
        # Replace <li> tags with "- "
        text = re.sub(r'<li>', '- ', text)
        # Remove </li> tags
        text = re.sub(r'</li>', '', text)
        # Convert <img> tags to markdown format
        text = re.sub(r'<img\s+src={require\([\'"](.*?)[\'"]\).default}\s+alt="(.*?)"\s*/?>',
                      r'![\2](\1)', text)
        # Handle <TabItem> tags with value and label attributes
        text = re.sub(r'<TabItem\s+value="(.*?)"\s+label="(.*?)"\s*(?:default)?>',
                      lambda match: f'**{match.group(2)}**', text, flags=re.DOTALL)
        # Convert <pre><code>...</code></pre> blocks to markdown code blocks
        text = re.sub(r'<pre><code>(.*?)</code></pre>', 
                      lambda match: f'\n```\n{match.group(1)}\n```\n', text, flags=re.DOTALL)

        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text

    # Function to process content
    def process_content(content):
        # Find the <Tabs> block and process its content
        def process_tabs_block(match):
            inner_content = match.group(1)
            cleaned_inner_content = replace_html_tags(inner_content)
            # Ensure there's a space between lines
            cleaned_inner_content = '\n'.join(line.strip() for line in cleaned_inner_content.splitlines())
            return cleaned_inner_content

        # Replace the <Tabs> block with cleaned content
        cleaned_content = tabs_pattern.sub(process_tabs_block, content)
        
        return cleaned_content

    return process_content(content)

def ask_for_confirmation():
    response = input("Make any changes to the combined_docs MD file. Ready to continue? (yes/no): ").strip().lower()
    if response not in ["yes", "y"]:
        print("Script terminated by user.")
        exit()

def markdown_to_pdf(input_files, output_file):
    try:
        combined_md_file = combine_markdown_files(input_files, 'combined_docs.md')

        with open(combined_md_file, 'r', encoding='utf-8') as file:
            markdown_content = file.read()

        # Apply transformations
        markdown_content = convert_markdown_links(markdown_content)
        markdown_content = remove_info_caution_tip_tags(markdown_content)
        markdown_content = remove_import_statements(markdown_content)
        markdown_content = remove_and_replace_html_tags(markdown_content)
        markdown_content = remove_alt_text_from_images(markdown_content)

        # Write modified content back to the combined Markdown file
        with open(combined_md_file, 'w', encoding='utf-8') as file:
            file.write(markdown_content)

        # Collect all directories from the input files
        directories = set(os.path.dirname(os.path.abspath(file_path)) for file_path in input_files)
        resource_path = ':'.join(directories)  # Use ':' as a separator for Unix-based systems

        margin_size = '0.6in'
        title = "H2O LLM Studio | Documentation"
        subtitle = "A framework and no-code GUI designed for fine-tuning state-of-the-art large language models (LLMs)"

        ask_for_confirmation()

        # Pandoc command to convert Markdown to PDF with additional options
        subprocess.run([
            'pandoc',
            os.path.abspath(combined_md_file),
            '--from', 'markdown',
            '--to', 'pdf',
            '--output', output_file,
            '--pdf-engine=pdflatex',
            '--pdf-engine-opt=-shell-escape',
            '--pdf-engine-opt=-interaction=nonstopmode',
            '--pdf-engine-opt=-halt-on-error',
            '--pdf-engine-opt=-file-line-error',
            f'--resource-path={resource_path}',  # Set the resource path for relative images
            '--variable', f'geometry:top={margin_size}, bottom={margin_size}, left={margin_size}, right={margin_size}', # Set margins to 0.5 inch
            '--toc',  # Add table of contents
            '--variable', f'title={title}',  # Pass the title variable
            '--variable', f'subtitle={subtitle}',  # Pass the subtitle variable
            '--variable', 'linkcolor:blue',  # Set link color to blue
        ], check=True)

        print(f'Converted {combined_md_file} to {output_file}')

    except subprocess.CalledProcessError as e:
        print(f'Error: Failed to convert {combined_md_file} to PDF.')
        print(e)
        raise

if __name__ == '__main__':
    input_md_files = [
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/get-started/what-is-h2o-llm-studio.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/get-started/set-up-llm-studio.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/get-started/llm-studio-performance.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/get-started/llm-studio-flow.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/get-started/core-features.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/concepts.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/datasets/data-connectors-format.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/datasets/import-dataset.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/datasets/view-dataset.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/datasets/merge-datasets.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/experiment-settings.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/create-an-experiment.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/view-an-experiment.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/compare-experiments.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/export-trained-model.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/evaluate-model-using-llm.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/guide/experiments/import-to-h2ogpt.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/faqs.md',
        '/Users/Sher/Documents/GitHub/h2o-llmstudio/documentation/docs/key-terms.md',
    ]
    output_pdf = 'h2o-llmstudio-docs.pdf'

    try:
        # Convert Markdown files to a single PDF
        markdown_to_pdf(input_md_files, output_pdf)

    except Exception as e:
        print(f'Error: Failed to generate PDF from Markdown files.')
        print(e)
