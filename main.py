# if product doesn't exist append
# keyword: what's in the search


import os
from bs4 import BeautifulSoup

repo_path = "./Pages"


def process_files_in_repo(repo_path):
    for root, _, files in os.walk(repo_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()  # Read the file
                    print(f"Read {len(content)} characters from {file_path}")
                    soup = BeautifulSoup(content, 'html.parser')
                    # print(soup.prettify())
                    print(soup.find(class_='search-query-result').text)
            
            except Exception as e:
                print(f"Could not open {file_path}: {e}")


def handle_markets():
    if text==='01':


# Example usage
process_files_in_repo(repo_path)
