
import argparse
import json


def json2jsonl(path: str):
    data = json.load(open(path))
    writer = open(path.replace('.json', '.jsonl'), 'w')
    for idx, item in enumerate(data):
        conversations = item['conversations']
        if conversations[0]['from'] == 'system':
            item['conversations'] = item['conversations'][1:]
        item['id'] = idx
        writer.write(json.dumps(item, ensure_ascii=False) + '\n')

    writer.close()


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('path', type=str)

    args = argparse.parse_args()

    assert args.path.endswith('.json')

    path = args.path
    json2jsonl(path)

