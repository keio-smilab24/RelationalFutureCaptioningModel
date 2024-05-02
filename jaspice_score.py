from metrics.JaSPICE.jaspice.api import JaSPICE
import json

INPUT_FILEPATH = "results/teisei/caption/translations_0_test.json"
OUTPUT_FILEPATH = "jaspice_scores/teisei_score.txt"

batch_size = 1
jaspice = JaSPICE(batch_size, server_mode = False)
sentence_list = []
gt_list = []
clip_list = []
with open(INPUT_FILEPATH, "r") as f:
    data = json.load(f)
    for result in data["results"].values():
        gts = []
        sentences = []
        clips = []
        if type(result) == list:
            for item in result:
                if item["sentence"] not in sentences:
                    sentences.append(item["sentence"])
                if item["clip_id"] not in clips:
                    clips.append(item["clip_id"])
                gts += (item["gt_sentence"])
            sentence_list.append(sentences)
            gt_list.append(gts)
            clip_list.append(clips)
        assert len(sentence_list) == len(gt_list)
result = []
for i in range(len(gt_list)):
    gts = dict()
    gt_listed = []
    for j in range(len(gt_list[i])):
        gt_listed.append(gt_list[i][j])
    gts[str(i)] = gt_listed
    caps = dict()

    for j in range(len(sentence_list[i])):
        caps[str(i)] = [sentence_list[i][j]]
    _, score = jaspice.compute_score(gts, caps)

    result.append([score, sentence_list[i], gt_list[i], clip_list[i]])
result.sort(key=lambda x: x[0], reverse=False)
with open(OUTPUT_FILEPATH, "w") as f:
    for i in range(len(result)):
        f.write(f"{result[i][0]}\n")
        f.write(f"clip_id: {result[i][3]}\n")
        f.write(f"Generated: {result[i][1]}\n")
        f.write(f"Ground truth: {result[i][2]}\n")
        f.write("\n")
