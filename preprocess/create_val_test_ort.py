import json
import MeCab


path = "data/pfn-pic/labels/ja.ort.remove.jsonl"
output_path = "data/pfn-pic/labels/ja.test.mecab.txt"
# MeCab::Taggerクラスのインスタンスを作成（ここではデフォルト設定）
m = MeCab.Tagger('-Owakati')


with open(path) as f:
    with open(output_path, mode='w') as wf:
        for s_line in f:    # 1行ずつ
            line = json.loads(s_line)
            if line['split'] == 'test':
                for j_line in line["objects"]:  # Objectsの中
                    for instruction in j_line["instructions"]:
                        #文章の表示
                        # print(instruction)

                        # 前処理 (\nの削除)
                        sentence = instruction.replace('\n', '')
                        wf.write(m.parse(sentence))

output_path = "data/pfn-pic/labels/ja.val.mecab.txt" 
with open(path) as f:
    with open(output_path, mode='w') as wf:
        for s_line in f:    # 1行ずつ
            line = json.loads(s_line)
            if line['split'] == 'val':
                for j_line in line["objects"]:  # Objectsの中
                    for instruction in j_line["instructions"]:
                        #文章の表示
                        # print(instruction)

                        # 前処理 (\nの削除)
                        sentence = instruction.replace('\n', '')
                        wf.write(m.parse(sentence))

                    


            
