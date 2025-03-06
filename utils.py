import json
import re
import os

def convert_json_to_txt(file_path, new_file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        with open(new_file_path, 'w', encoding='utf-8') as f:
            for item in data:
                if isinstance(item, list):  # Check if it's a list
                    f.write(f"{str(item)}\n")  # Convert list to string before writing
                else:
                    f.write(f"{item}\n")

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def extend_file(to_be_extended, content):
    to_be_extended.extend(content)
    return to_be_extended

def make_file(file_path, file_name, new_content, type):
    for root, _, files in os.walk(file_path):
        if (file_name + type) not in files:
            print('Info1: File not found')
            with open((file_path + '/' + file_name + type), 'w', encoding='utf-8') as f:
                json.dump(new_content, f, indent=4)
        else:
            print('Info2: File found')
            with open((file_path + '/' + file_name + type), 'r+', encoding='utf-8') as f:
                to_be_extended = json.load(f)
                try:
                    if type == '.json':
                        data = extend_file(to_be_extended, new_content)
                        data = remove_duplicates(data)
                        f.seek(0)  # Move the pointer to the start of the file
                        f.truncate()  # Clear the file content
                        json.dump(data, f, indent=4)
                        # f.write(data)
                        
                except json.JSONDecodeError:
                    f.seek(0)  # Move to the start of the file to write new content
                    f.truncate()  # Clear the file content
                    json.dump(new_content, f, indent=4)


def remove_duplicates(data):
    unique_data = list({json.dumps(d, sort_keys=True): d for d in data}.values())
    return unique_data

def make_catalogue(market_name, product_name, product_path, content):
    content = json.loads(json.dumps(content))
    file_name = market_name + '_' + product_name
    make_file(product_path, file_name, content, '.json')

def extract_volume(text):
    pattern = r"(\d+\.?\d*)\s*(l|L|ml|ML)"
    
    match = re.search(pattern, text)
    
    if match:
        return f"{match.group(1)}{match.group(2).lower()}"
    return 0

def extract_item_count(text):
    match = re.search(r'(\d+)(?=개|P)', text)

    if match:
        return int(match.group(1)) if match else 1
    return 1

def contains_first_four(keyword, full_text):
    prefix = keyword[:4]  # Get the first four characters of the target string
    return prefix in full_text  # Check if it's in the text

def convert_to_ml(text):
    matches = re.findall(r"(\d+\.?\d*)(l|ml)", text)
    
    for match in matches:
        value, unit = match
        value = float(value)
        
        if unit == 'l':
            value *= 1000
        return value

def clean_and_convert_to_int(text):
    # Remove unwanted special characters (except dots and commas)
    cleaned_text = re.sub(r'[\-(){}\[\]_:;\'"!?\@#$%^&*+=|\\/]', "", text)
    
    # Find the first sequence of numbers, allowing for commas and dots
    match = re.search(r"\d+(?:[.,]\d+)*", cleaned_text)
    
    return int(match.group().replace(",", "")) if match else 1


    # return int(cleaned_text)