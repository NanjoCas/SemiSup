import json, random

INPUT_FILE = "train.jsonl"
LABELED_OUT = "labeled.jsonl"
UNLABELED_OUT = "unlabeled.jsonl"

# ---------------------------
# 配置：有标签比例
LABELED_RATIO = 0.10   # 10%
# ---------------------------

# 读取全部数据
data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

total = len(data)
print("Total samples:", total)

# 随机打乱
random.seed(42)
random.shuffle(data)

# 按比例切割
labeled_size = int(total * LABELED_RATIO)
labeled = data[:labeled_size]
unlabeled = data[labeled_size:]

print("Labeled:", len(labeled))
print("Unlabeled:", len(unlabeled))

# 写出 labeled.jsonl
with open(LABELED_OUT, "w", encoding="utf-8") as f:
    for item in labeled:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 写出 unlabeled.jsonl
with open(UNLABELED_OUT, "w", encoding="utf-8") as f:
    for item in unlabeled:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Files written:")
print(" -", LABELED_OUT)
print(" -", UNLABELED_OUT)
