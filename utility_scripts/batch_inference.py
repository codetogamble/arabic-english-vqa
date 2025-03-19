import requests
import json

def send_image_questions(image_path, questions):
    url = "http://localhost:8000/qna"
    
    # Open and prepare the image file
    files = {
        "image": open(image_path, "rb")
    }
    
    # Prepare the form data with questions
    data = {
        "question": questions
    }
    
    # Send POST request
    try:
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None
    finally:
        files['image'].close()  # Make sure to close the file

# Example usage
if __name__ == "__main__":
    with open("/data/test.json") as f:
        data = json.load(f)

    image_path_questions = {} 
    for x in data[:5]:
        print(x)
        ques = x['question']
        image_path = x["image_name"]
        answer = x['answer']
        if image_path not in image_path_questions:
            image_path_questions[image_path] = []
        image_path_questions[image_path].append(ques)

    answers_map = {} 
    for k,v in image_path_questions.items():
        print("Image Path:", k)
        print("Questions:", v)
        result = send_image_questions(k, v)
        if result:
            print("Response:", result)
        answers_map[k] = result
    
    with open("test_answers.json","w") as f:
        json.dump(answers_map, f, indent=4)