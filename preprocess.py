# if product doesn't exist append
# keyword: what's in the search


import os
import utils
from bs4 import BeautifulSoup
import json

html_path = "./Html"
product_path = "./Products"
markets_json = "./markets.json"
market_objs = utils.read_json(markets_json)

def return_product_temp(soup):
    for i in market_objs:
        if i['MARKET_NAME'] in soup.get_text():
            return i

def make_file(product_name):
    for root, _, files in os.walk(product_path):
        if product_name + '.txt' not in files:
            print('Info1: File not found')
            with open(product_path + '/' + product_name + '.txt', 'w') as f:
                # content= json.dumps()
                f.write(content)

def find_products(market_id):
    case = market_id
    
    if case == "01":
        headerSearchKeyword

        return "Apple"
    elif case == "02":
        return "Banana"
    elif case == "03":
        return "Cherry"
    else:
        return "Invalid choice"



def process_files_in_repo(html_path):
    for root, _, files in os.walk(html_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()  # Read the file
                    print(f"Read {len(content)} characters from {file_path}")
                    soup = BeautifulSoup(content, 'html.parser')
                    product_temp = return_product_temp(soup)

                    # print(soup.prettify())
                    search_el = soup.select_one(product_temp['SEARCH_ID'])
                    print(search_el)
                    if search_el:
                        product_name = search_el.get("value") 
                        make_file(product_name, product_temp)
                    else:
                        print('Error1: Product not found')

            
            except Exception as e:
                print(f"Could not open {file_path}: {e}")





# Example usage
# process_files_in_repo(repo_path)

def main():
    process_files_in_repo(html_path)
    # check_market()



main()
