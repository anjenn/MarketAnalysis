import json

def read_json(file_path):
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)




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
