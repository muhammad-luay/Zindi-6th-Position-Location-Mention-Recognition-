# %%
!pip install transformers

# %%
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


# %%
import json
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("rsuwaileh/IDRISI-LMR-EN-random-typebased")
model = AutoModelForTokenClassification.from_pretrained("rsuwaileh/IDRISI-LMR-EN-random-typebased")

# Create a pipeline
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer)

# Function to process a single text and extract entities
def process_text(text):
    entities = ner_pipe(text)
    # return [entity['word'] for entity in entities]
    return entities

# Initialize an empty list to store the rows of the output CSV
rows = []

# Read data from jsonl file
with open('/content/not_tested.jsonl', 'r') as f:
    for line in f:
        # Parse JSON
        tweet = json.loads(line)
        # Process text
        entities = process_text(tweet['text'])
        print(entities)
        # Append a row to the list
        rows.append({'tweet_id': tweet['tweet_id'], 'text': tweet['text'], 'entities_suweilah': entities})

# Convert the list of rows to a DataFrame
df = pd.DataFrame(rows)

# Write DataFrame to CSV
df.to_csv('suweilah_test.csv', index=False)


# %%
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# Function to process a single text and extract entities
def process_text(text):
    entities = ner_pipe(text)
    # return [entity['word'] for entity in entities]
    return entities

# Initialize an empty list to store the rows of the output CSV
rows = []

# Read data from jsonl file
with open('/content/train.jsonl', 'r') as f:
    for line in f:
        # Parse JSON
        tweet = json.loads(line)
        # Process text
        entities = process_text(tweet['text'])
        print(entities)
        # Append a row to the list
        rows.append({'tweet_id': tweet['tweet_id'], 'text': tweet['text'], 'entities': entities})

# Convert the list of rows to a DataFrame
df = pd.DataFrame(rows)

# Write DataFrame to CSV
df.to_csv('output3.csv', index=False)




