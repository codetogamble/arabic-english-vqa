import subprocess
import json
from pathlib import Path

with open('pre_train.json', 'r') as f:
    data = json.load(f)


# Path("./pdf_images").mkdir(exist_ok=True)
# for x in data:
#     source_path = x["pdf_path"]
#     file_name = source_path.split("/")[-1].replace(".pdf", "")
#     # break
#     subprocess.run(f"pdftoppm {source_path} pdf_images/{file_name} -png", shell=True)


new_train_data = []
all_images = [x for x in Path("./pdf_images").rglob("*.png")]
for x in data:
    pdf_path = x["pdf_path"]
    file_name = pdf_path.split("/")[-1].replace(".pdf", "")
    images_for_pdf = [y for y in all_images if file_name in str(y)]
    print(images_for_pdf)
    for y in images_for_pdf:
        new_train_data.append({"image_path": str(y.name), "question": x["question"], "answer": x["answer"]})


with open("train.json", "w") as f:
    json.dump(new_train_data, f, indent=4, ensure_ascii=False)