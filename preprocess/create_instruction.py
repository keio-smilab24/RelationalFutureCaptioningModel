import json

path = "data/pfn-pic/labels/ja.ort.mecab.jsonl"
write_path = "data/pfn-pic/labels/ja.instructions.mecab.txt"

with open(write_path, 'w') as wf:
    with open(path) as f:
        for s_line in f:    # 1行ずつ
            line = json.loads(s_line)

            for j_line in line["objects"]:  # Objectsの中
                for instruction in j_line["instructions"]:
                    # 文章の表示
                    # print(instruction)
                    
                    wf.write(instruction)
                    wf.write('\n')
            
