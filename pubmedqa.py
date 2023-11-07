import json

class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        return data

    def get_items(self):
        for item_id, item_data in self.data.items():
            yield item_id, item_data

    def get_item_ids(self):
        return list(self.data.keys())

    def get_item(self, item_id):
        return self.data.get(item_id)

    def __iter__(self):
        return iter(self.data.items())

'''
# Usage example
file_path = 'ori_pqal.json'
dataset = Dataset(file_path)

# Iterate over the dataset items
for item_id, item_data in dataset.get_items():
    # Access item properties
    question = item_data['QUESTION']
    contexts = item_data['CONTEXTS']
    labels = item_data['LABELS']
    # ... (access other properties as needed)
    print(f"Item ID: {item_id}")
    print(f"Question: {question}")
    print(f"Contexts: {contexts}")
    print(f"Labels: {labels}")
    print()

# Access a specific item by ID
item_id = '21645374'
item = dataset.get_item(item_id)
if item:
    question = item['QUESTION']
    contexts = item['CONTEXTS']
    labels = item['LABELS']
    # ... (access other properties as needed)
    print(f"Item ID: {item_id}")
    print(f"Question: {question}")
    print(f"Contexts: {contexts}")
    print(f"Labels: {labels}")
'''