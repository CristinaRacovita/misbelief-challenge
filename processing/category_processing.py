def filter_categories(data, columns):
    return data[data["category"] in columns].reset_index()