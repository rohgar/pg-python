import fasttext
import os

# import nltk
# nltk.download('stopwords')

# Download stopwords from https://www.nltk.org/nltk_data/
# https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip
# and move to
# cp ~/Downloads/stopwords.zip /Users/rohit24/nltk_data/corpora/
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def filter_text(text: str):
    filtered_text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return filtered_text


MODEL_BIN_FILE='./model_cooking.bin'

if not os.path.exists(MODEL_BIN_FILE):
    model = fasttext.train_supervised(input="./data/cooking.train")
    model.save_model(MODEL_BIN_FILE)
    print("\n")

model = fasttext.load_model(MODEL_BIN_FILE)

if __name__ == '__main__':
    query="Which baking dish is best to bake a banana bread?"
    query = filter_text(query)
    print(f"predicting {query}")
    prediction_tuple = model.predict(query)
    print(f"result = {prediction_tuple}\n")

    query="Why not put knives in the dishwasher?"
    query = filter_text(query)
    print(f"predicting {query}")
    prediction_tuple = model.predict(query)
    print(f"result = {prediction_tuple}\n")

    print("Test the model using the validation dataset")
    print(model.test("./data/cooking.valid"))
    print("(total samples, P@total_samples, R@total_samples )")
    print("\n")

    print("Check the Precision@5 and Recall@5:")
    print(model.test("./data/cooking.valid", k=5))
    print("(total samples, P@5, R@5 )")
    print("\n")

