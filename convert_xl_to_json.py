import json
with open("/home/shubham/Downloads/finetune_v1_imi-Sheet1.tsv") as f:
    data = [x.strip() for x in f]


def generate_combinations_of_questions(entity):
    # translate the question into arabic
    # ask LLM to rephrase the question
    return [entity, f"What is {entity}?"]

pre_train_data = []
for x in data:
    pdf_path = x.split("\t")[0]
    entity = x.split("\t")[1]
    answer = x.split("\t")[2]
    questions = generate_combinations_of_questions(entity)
    for q in questions:
        pre_train_data.append({"pdf_path": pdf_path, "question": q, "answer": answer})


with open("pre_train.json", "w") as f:
    json.dump(pre_train_data, f, indent=4, ensure_ascii=False)
