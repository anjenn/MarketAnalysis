html_path = "./Html"
import preprocess
# import predict
# import model
import utils

preprocess.process_files_in_repo(html_path)
utils.convert_json_to_txt("Ranks/해바라기씨유Reviews_rank.json", "Ranks/해바라기씨유Reviews_rank.txt")
utils.convert_json_to_txt("Ranks/해바라기씨유Unit_price_by_quantity.json", "Ranks/해바라기씨유Unit_price_by_quantity.txt")