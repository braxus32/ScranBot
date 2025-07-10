import json

obj = None
# with open('key.json', 'r') as f:
#     line = f.read()
#     print(line)
#     line = line.split('][')
#     print(line)
#     new_line = ""
#     for item in line:
#         new_line += item + '],['
#
#     new_line = new_line[:-2]
#     print('[' + new_line)
#     obj = json.loads('[' + new_line)
#     print(obj)
#
# with open('key.json', 'w') as f:
#     json.dump(obj, f)

data_set = []
key_list = []

with open('data.json', 'r') as df, open('key.json', 'r') as kf:
    data_pairs = json.load(df)
    key_pairs = json.load(kf)
    for data_pair, key_pair in zip(data_pairs, key_pairs):
        for data, key in zip(data_pair, key_pair):
            if data not in data_set:
                data_set.append(data)
                key_list.append(key)

    print(len(data_pairs))
    print(len(key_pairs))

    print(data_set)
    print(len(data_set))
    print(key_list)
    print(len(key_list))

# with open('data.json', 'w') as df, open('key.json', 'w') as kf:
#     json.dump(data_set, df)
#     json.dump(key_list, kf)
#


