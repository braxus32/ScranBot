import json

obj = None
with open('key.json', 'r') as f:
    line = f.read()
    print(line)
    line = line.split('][')
    print(line)
    new_line = ""
    for item in line:
        new_line += item + '],['

    new_line = new_line[:-2]
    print('[' + new_line)
    obj = json.loads('[' + new_line)
    print(obj)

with open('key.json', 'w') as f:
    json.dump(obj, f)
