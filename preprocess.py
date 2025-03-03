import os
import utils
from bs4 import BeautifulSoup
import json

html_path = "./Html"
product_path = "./Products"
markets_json = "./markets.json"
market_objs = utils.read_json(markets_json)

def return_product_src_temp(soup):
    for i in market_objs:
        if i['MARKET_NAME'] in soup.get_text():
            return i

def make_file(product_name, product_src_temp, soup):
    content = get_products(product_src_temp, soup)
    content = json.dumps(content)

    for root, _, files in os.walk(product_path):
        if product_name + '.json' not in files:
            print('Info1: File not found')
            with open(product_path + '/' + product_name + '.json', 'w') as f:
                f.write(content)
        else:
            print('Info2: File found')
            with open(product_path + '/' + product_name + '.json', 'r+') as f:
                try:
                # Read and parse the existing content (assumed to be an array of objects)
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                    existing_data.extend(json.loads(content))
                    f.seek(0)  # Move the pointer to the start of the file
                    f.truncate()  # Clear the file content
                    json.dump(existing_data, f, indent=4)

                except json.JSONDecodeError:
                    print("DEBUGGING")

                    # If the file is empty or corrupted, start with an empty list
                    existing_data = content
                    f.seek(0)  # Move to the start of the file to write new content
                    f.truncate()  # Clear the file content
                    json.dump(existing_data, f, indent=4)

def get_products(product_src_temp, soup):
    class_name = product_src_temp['ITEM_PREFIX']
    data = []
    for product_obj in soup.find_all(class_=class_name):
        for i in product_obj:
            try:
                obj = {
                    "PRICE": utils.clean_and_convert_to_int(i.select_one(product_src_temp['PRICE_EL_ID']).get_text()),
                    "REVIEW_COUNT": utils.clean_and_convert_to_int(i.select_one(product_src_temp['REVIEW_EL_ID']).get_text()),
                    "UNIT_PRICE": utils.clean_and_convert_to_int(i.select_one(product_src_temp['UNIT_PRICE_ID']).get_text()),
                    "QUANTITY": utils.convert_to_ml(utils.extract_first_volume(i.select_one(product_src_temp['QUANTITY_ID']).get_text()))
                }
                if 'SOLD_COUNT_ID' in product_src_temp: # TEST THIS
                    print("DEBUGGING")
                    obj['SOLD_COUNT_ID'] = i.select_one(product_src_temp['SOLD_COUNT_ID']).get_text() or ''
            except Exception as e:
                print(f"Error0: {e}")
            data.append(obj)
    return data

def process_files_in_repo(html_path):
    for root, _, files in os.walk(html_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()  # Read the file
                    print(f"Read {len(content)} characters from {file_path}")
                    soup = BeautifulSoup(content, 'html.parser')
                    product_src_temp = return_product_src_temp(soup)

                    # print(soup.prettify())
                    search_el = soup.select_one(product_src_temp['SEARCH_ID'])
                    if search_el:
                        product_name = search_el.get("value") 
                        make_file(product_name, product_src_temp, soup)
                    else:
                        print('Error1: Product not found')

            
            except Exception as e:
                print(f"Could not open {file_path}: {e}")



process_files_in_repo(html_path)