import json
import re

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)

def extract_volume(text):
    pattern = r"(\d+\.?\d*)\s*(l|L|ml|ML)"
    
    match = re.search(pattern, text)
    
    if match:
        return f"{match.group(1)}{match.group(2).lower()}"
    return None

def extract_item_count(text):
    match = re.search(r'(\d+)(?=개)', text)

    if match:
        return int(match.group(1)) if match else None
    return None

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
    # Define characters to remove
    cleaned_text = re.sub(r'[,.\-(){}\[\]_:;\'"!?\@#$%^&*+=|\\/]', "", text)    
    return int(cleaned_text)

# def handle_market_search(case):
#     if case == "01":
#         headerSearchKeyword

#         return "Apple"
#     elif case == "02":
#         return "Banana"
#     elif case == "03":
#         return "Cherry"
#     else:
#         return "Invalid choice"
