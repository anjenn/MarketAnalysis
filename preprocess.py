import os
import utils
from bs4 import BeautifulSoup
import json
import re

html_path = "./Html"
# html_path = "./Html2"
product_path = "./Products"
rank_path = "./Ranks"
markets_json = "./markets.json"
market_objs = utils.read_json(markets_json)

def return_product_src_temp(soup):
    for i in market_objs:
        if i['MARKET_NAME'] in soup.get_text():
            return i['MARKET_NAME'], i


def make_ranking_report(market_name, product_name, product_data):
    reviews_rank = sorted(product_data, key=lambda x: x["REVIEW_COUNT"], reverse=True)
    ordered_quantity = sorted(product_data, key=lambda x: x["QUANTITY"], reverse=False)

    cheapest_per_quantity = {}
    for item in ordered_quantity:
        qty = item["QUANTITY"]
        if qty not in cheapest_per_quantity or item["UNIT_PRICE"] < cheapest_per_quantity[qty]["UNIT_PRICE"]:
            cheapest_per_quantity[qty] = item

    cheapest_items = list(cheapest_per_quantity.values())
    utils.make_file(rank_path, market_name + '_' + product_name + 'Reviews_rank', reviews_rank, '.json')
    utils.make_file(rank_path, market_name + '_' + product_name + 'Unit_price_by_quantity', cheapest_items, '.json')


def get_products(product_src_temp, soup, product_name):
    class_name = product_src_temp['ITEM_PREFIX']
    data = []
    # for product_obj in soup.find_all(class_=class_name) or :
    product_soups = [BeautifulSoup(str(product_obj), 'html.parser') for product_obj in soup.select(class_name)]

    for idx, product_soup in enumerate(product_soups, 1):
        i = product_soup
        title_element = i.select_one(product_src_temp['TITLE_ID'])
        quantity_element = i.select_one(product_src_temp['QUANTITY_ID'])
        price_element = i.select_one(product_src_temp['PRICE_EL_ID'])
        review_count_element = i.select_one(product_src_temp['REVIEW_EL_ID'])
        unit_price_element = i.select_one(product_src_temp['UNIT_PRICE_ID'])
        
        if title_element and quantity_element and price_element and utils.contains_first_four(product_name, quantity_element.get_text()) and '세트' not in quantity_element.get_text():
            try:
                obj = {
                    "TITLE": title_element.get_text(),
                    "ITEM_COUNT": utils.extract_item_count(quantity_element.get_text()) if quantity_element else 1,
                    "PRICE": utils.clean_and_convert_to_int(price_element.get_text()),
                    "REVIEW_COUNT": utils.clean_and_convert_to_int(review_count_element.get_text()) if review_count_element else 1,
                    "QUANTITY": utils.convert_to_ml(utils.extract_volume(quantity_element.get_text())) if quantity_element else 1,
                    "UNIT_PRICE": utils.clean_and_convert_to_int(unit_price_element.get_text()) if unit_price_element else 1
                }
                if obj["UNIT_PRICE"] == 1 and obj["PRICE"] != 1:
                    obj["UNIT_PRICE"] = obj["PRICE"] // (obj["QUANTITY"]*obj["ITEM_COUNT"]) * 100
            except Exception as e:
                print(f"Error in GET_PRODUCTS: {e}")
            data.append(obj)
    return data

def get_review_counts(product_path, rank_path):
    Revew_data = {}
    for root, _, files in os.walk(rank_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            # pattern = rf'\b{re.escape(file_name)}\b' 
            if re.search('rank', os.path.splitext(file_path)[0]):

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)  # Read JSON content
                        if isinstance(data, list) and len(data) > 0 and "REVIEW_COUNT" in data[0]:  
                            Revew_data[file_name] = data[0]["REVIEW_COUNT"]  # Store first item's rank
                        else:
                            Revew_data[file_name] = None  # Handle empty or invalid files
                except Exception as e:
                    print("Error in calc_review_ratio {file_name}: {e}")
    
    output_path = os.path.join(product_path, 'review_counts')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(Revew_data, out_f, indent=4, ensure_ascii=False)

    return Revew_data  # Return the dictionary if needed

def add_review_ratio(product_path):
    with open(product_path + '/review_counts', 'r', encoding='utf-8', errors='ignore') as f:
        review_obj = json.load(f)

    for root, _, files in os.walk(product_path):
        for file in files:
            file_name = os.path.splitext(file)[0]  # Remove file extension

            for review_file_key in review_obj.keys():
                if file_name in review_file_key:
                    max_review = review_obj[review_file_key]

            for key in review_obj.keys():
                if file_name in key:
                    try:
                        with open(product_path + '/' + file_name + '.json', 'r', encoding='utf-8') as f:  # Open the file
                            product_objs = json.load(f)  # Load JSON data
                            for obj in product_objs:
                                obj["REVIEW_RATIO"] = round((obj["REVIEW_COUNT"] / max_review) * 100, 2)

                        print(product_objs)
                        with open(product_path + '/' + file_name + '.json', 'w', encoding='utf-8') as f:  # Open the file for writing
                            json.dump(product_objs, f, indent=4, ensure_ascii=False)  # Write the modified data

                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_name}")




def process_files_in_repo(html_path):
    for root, _, files in os.walk(html_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()  # Read the file
                    print(f"Read {len(content)} characters from {file_path}")
                    soup = BeautifulSoup(content, 'html.parser')
                    market_name, product_src_temp = return_product_src_temp(soup)

                    # print(soup.prettify())
                    search_el = soup.select_one(product_src_temp['SEARCH_ID'])
                    # print(search_el.prettify())
                    if search_el:
                        product_name = search_el.get("value")
                        product_data = get_products(product_src_temp, soup, product_name)
                        utils.make_catalogue(market_name, product_name, product_path, product_data)
                        make_ranking_report(market_name, product_name, product_data)
                        
                    else:
                        print('Error1: Product not found')
            
            except Exception as e:
                print(f"Error in {file_path}: {e}")


 # Needed for better modelling

process_files_in_repo(html_path)
get_review_counts(product_path, rank_path)
add_review_ratio(product_path)
