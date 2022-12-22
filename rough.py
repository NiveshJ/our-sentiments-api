from tqdm.notebook import tqdm
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

plt.style.use("ggplot")


app = FastAPI()


class senA(BaseModel):
    reviewerName: str
    reviewTitle: str
    reviewBody: str
    reviewStars: int


@app.get('/')
def index():
    return {'Index Here'}


@app.get('/json')
def data_json(file: json):
    return {'Json Here'}


@app.post('/Sentimentanalysis')
def robertaonly(jsondata):
    print(jsondata)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)


path = "./amazonReviews.json"
with open(path, 'r') as j:
    contents = json.loads(j.read())
df = pd.DataFrame(contents)

df = df.head(9)
print(df.shape)

ax = df['reviewStars'].value_counts().sort_index().plot(
    kind='bar', title='Count of reviews by stars', figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

example = df['reviewBody'][8]
print(example)


task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
print(scores_dict)


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['reviewBody']
        myid = row['reviewerName']
        overall = row['reviewStars']
        roberta_result = polarity_scores_roberta(text)
        res[myid, overall] = roberta_result
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(
    columns={'index': 'reviewerID', 'overall': 'rating'})
results_df.rename(columns={'level_0': 'reviewerID',
                  'level_1': 'overall'}, inplace=True)
print(results_df)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=results_df, x='overall', y='roberta_pos', ax=axs[0])
sns.barplot(data=results_df, x='overall', y='roberta_neu', ax=axs[1])
sns.barplot(data=results_df, x='overall', y='roberta_neg', ax=axs[2])
axs[0].set_title('Postivie')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
