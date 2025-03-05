import os
import utils
from bs4 import BeautifulSoup
import json
import time

html_path = "./Html"
product_path = "./Products"
rank_path = "./Ranks"
markets_json = "./markets.json"
market_objs = utils.read_json(markets_json)

def return_product_src_temp(soup):
    for i in market_objs:
        if i['MARKET_NAME'] in soup.get_text():
            return i


def make_ranking_report(product_name, product_data):
    reviews_rank = sorted(product_data, key=lambda x: x["REVIEW_COUNT"], reverse=True)
    ordered_quantity = sorted(product_data, key=lambda x: x["QUANTITY"], reverse=False)

    cheapest_per_quantity = {}
    for item in ordered_quantity:
        qty = item["QUANTITY"]
        if qty not in cheapest_per_quantity or item["UNIT_PRICE"] < cheapest_per_quantity[qty]["UNIT_PRICE"]:
            cheapest_per_quantity[qty] = item

    cheapest_items = list(cheapest_per_quantity.values())
    utils.make_file(rank_path, product_name + 'Reviews_rank', reviews_rank, '.json')
    utils.make_file(rank_path, product_name + 'Unit_price_by_quantity', cheapest_items, '.json')


def get_products(product_src_temp, soup, product_name):
    class_name = product_src_temp['ITEM_PREFIX']
    data = []
    for product_obj in soup.find_all(class_=class_name):
        for i in product_obj:
            if utils.contains_first_four(product_name, i.select_one(product_src_temp['QUANTITY_ID']).get_text()) and '세트' not in i.select_one(product_src_temp['QUANTITY_ID']).get_text():
                try:
                    obj = {
                        "TITLE": i.select_one(product_src_temp['QUANTITY_ID']).get_text(),
                        "ITEM_COUNT": utils.extract_item_count(i.select_one(product_src_temp['QUANTITY_ID']).get_text()) or 1,
                        "REVIEW_COUNT": utils.clean_and_convert_to_int(i.select_one(product_src_temp['REVIEW_EL_ID']).get_text()) or 0,
                        "UNIT_PRICE": utils.clean_and_convert_to_int(i.select_one(product_src_temp['UNIT_PRICE_ID']).get_text()),
                        "QUANTITY": utils.convert_to_ml(utils.extract_volume(i.select_one(product_src_temp['QUANTITY_ID']).get_text()))
                    }
                    # DEBUG above
                except Exception as e:
                    print(f"Error in GET_PRODUCTS: {e}")
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
                        product_data = get_products(product_src_temp, soup, product_name)
                        utils.make_catalogue(product_name, product_path, product_data)
                        make_ranking_report(product_name, product_data)
                        
                    else:
                        print('Error1: Product not found')
            
            except Exception as e:
                print(f"Error in {file_path}: {e}")



process_files_in_repo(html_path)