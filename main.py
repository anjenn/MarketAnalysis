# if product doesn't exist append
# keyword: what's in the search


import os
import utils
from bs4 import BeautifulSoup

html_path = "./Html"
product_path = "./Products"
markets_json = "./markets.json"
market_objs = utils.read_json(markets_json)

def return_market_params(soup):
    for i in market_objs:
        if i['market_name'] in soup.get_text():
            return [i['search_id'], i['priceEl_id'], i['quantityEl_id'], i['reviewEl_id'], i['unit_price']]

def make_file(item_name):
    for root, _, files in os.walk(product_path):
        if item_name + '.txt' not in files:
            print('Error1: File not found')
            with open(product_path + '/' + item_name + '.txt', 'w') as f:
                f.write('keyword: what\'s in the search')


def process_files_in_repo(html_path):
    for root, _, files in os.walk(html_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()  # Read the file
                    print(f"Read {len(content)} characters from {file_path}")
                    soup = BeautifulSoup(content, 'html.parser')
                    keywords = return_market_params(soup)
                    # make_file(keywords[0])


                    # print(soup.prettify())
                    product_el = soup.select_one(keywords[0])
                    if soup.select_one(keywords[0]):
                        product_name = product_el.get("value") 
                        make_file(product_name)
                    else:
                        print('Error2: Product not found')

            
            except Exception as e:
                print(f"Could not open {file_path}: {e}")





# Example usage
# process_files_in_repo(repo_path)

def main():
    process_files_in_repo(html_path)
    # check_market()



main()
