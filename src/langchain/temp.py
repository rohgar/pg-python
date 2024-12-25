def create_doc_dicts(data, n=5):
    output = []
    for item in data[:n]:
        output.append({"doc": item})
    return output

data = ['foo', 'bar']

print(create_doc_dicts(data))