# %%
import ast
import pandas as pd
import jellyfish

import re

# %%
df = pd.read_csv('/Users/muhammadluay/Desktop/suweilah_test.csv')

# %%


# %%
# Function to remove unicode characters
def remove_unicode(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

# Apply the function to the 'text' column where 'is_match' is False
df['text'] = df['text'].apply(remove_unicode)


# %%
# Function to process each row and concatenate BILOU entities
def process_entities(row):
    entities = ast.literal_eval(row)
    if not entities:
        return []

    grouped_entities = []
    current_entity = []

    for entity in entities:
        tag = entity['entity'][0]  # Get the first character of the entity tag (B, I, L, U, O)

        if tag in ['B', 'U']:
            if current_entity:  # If there is an ongoing entity, save it and start a new one
                grouped_entities.append(current_entity)
            current_entity = [entity]  # Start a new entity group
        elif tag in ['I', 'L']:
            current_entity.append(entity)

        if tag in ['L', 'U']:  # If the entity ends, save it
            grouped_entities.append(current_entity)
            current_entity = []

    # Process grouped entities to concatenate words and get start/end offsets
    result_entities = []
    for group in grouped_entities:
        text = ' '.join([e['word'] for e in group])  # Concatenation
        start_offset = group[0]['start']
        end_offset = group[-1]['end']
        result_entities.append({'text': text, 'start_offset': start_offset, 'end_offset': end_offset})

    return result_entities

# Apply the function to each row in the dataframe
df['processed_entities'] = df['entities_suweilah'].apply(process_entities)

# Function to post-process entities and merge adjacent tokens
def merge_entities(row):
    entities = row
    if not entities:
        return []

    merged_entities = []
    current_entity = {'text': '', 'start_offset': entities[0]['start_offset'], 'end_offset': entities[0]['end_offset']}

    for i, entity in enumerate(entities):
        if i > 0 and entity['start_offset'] == entities[i - 1]['end_offset']:
            current_entity['text'] += entity['text'].replace('##', '')
            current_entity['end_offset'] = entity['end_offset']
        else:
            if current_entity['text']:
                merged_entities.append(current_entity)
            current_entity = {'text': entity['text'].replace('##', ''), 'start_offset': entity['start_offset'], 'end_offset': entity['end_offset']}

    merged_entities.append(current_entity)  # Append the last entity

    return merged_entities


# Apply the post-processing function
df['merged_entities'] = df['processed_entities'].apply(merge_entities)

# %%
df['entities_suweilah']

# %%
df['text']

# %%
df.columns

# %%
df['merged_entities'][4051]

# %%
df['suweilah_labels'] = df['merged_entities'].apply(lambda x: [item['text'] for item in x])

# %%
df['suweilah_labels'].tail(50)

# %%
malformed_df = pd.read_csv('/Users/muhammadluay/Desktop/Zindi/IDRISI-main/scripts/process/malformed_suweilah.csv')

# %%
malformed_df.head(50)

# %%
df['text']

# %%
corrected_labels = {}

# Loop through each malformed label in the df['suweilah_labels'] series
for index, labels in df['suweilah_labels'].items():
    # print(index)
    text = df['text'][index]  # Get the corresponding text
    corrected = []
    for label in labels:
        candidates = []  # To store potential correct labels
        for _, row in malformed_df.iterrows():
            malformed_label = row['malformed_label']
            correct_label = row['correct_label']
            # Calculate the Jaro similarity
            similarity = jellyfish.jaro_similarity(label.lower(), malformed_label.lower())
            if similarity > 0.8:
                candidates.append(correct_label)
        
        # Check the original text for the most suitable correct label
        if candidates:
            max_similarity = 0
            selected_label = ""
            for candidate in candidates:
                if candidate.lower() in text.lower() and len(candidate) > len(selected_label):
                    selected_label = candidate
            
            corrected.append(selected_label if selected_label else label)
        else:
            corrected.append(label)
    
    corrected_labels[index] = corrected

# Add the corrected labels to the dataframe
df['corrected_labels'] = pd.Series(corrected_labels)

# %%
df[['suweilah_labels','corrected_labels']].tail(50)

# %%
df['merged_entities'][4031]

# %%
df['text'][0]

# %%
# Function to process each row according to the provided conditions
def process_row(row):
    suweilah_labels = row['suweilah_labels']
    corrected_labels = row['corrected_labels']
    merged_entities = row['merged_entities']
    merged_entities2 = []

    # If the lengths of the suweilah_labels and corrected_labels lists are the same
    if len(suweilah_labels) == len(corrected_labels):
        for i in range(len(suweilah_labels)):
            entity = merged_entities[i]  # Get the corresponding entity
            
            # If the length of the individual labels is the same
            if len(suweilah_labels[i]) == len(corrected_labels[i]):
                merged_entities2.append(entity)
            else:
                # If the lengths are different, use the longer label and update the end_offset
                entity['text'] = corrected_labels[i]
                entity['end_offset'] = entity['start_offset'] + len(corrected_labels[i])
                merged_entities2.append(entity)

    return merged_entities2

# Apply the process_row function to each row in the DataFrame
df['merged_entities2'] = df.apply(process_row, axis=1)


# %%
def remove_overlapping_entities(entities):
    if not entities:
        return []
    
    non_overlapping_entities = []
    entities.sort(key=lambda x: x['end_offset'])  # Sort by end_offset to make the comparison easier
    
    previous_entity = None
    for entity in entities:
        if previous_entity:
            # If the end_offset is the same and the start_offset is larger, skip the entity
            if entity['end_offset'] == previous_entity['end_offset'] and entity['start_offset'] > previous_entity['start_offset']:
                continue
            
        non_overlapping_entities.append(entity)
        previous_entity = entity
    
    return non_overlapping_entities

# Applying the remove_overlapping_entities function
df['merged_entities2'] = df['merged_entities2'].apply(remove_overlapping_entities)


# %%
df['updated_labels2'] = df['merged_entities2'].apply(lambda x: [item['text'] for item in x])


# %%
# Function to check if a string has non-ASCII characters
def has_non_ascii(s):
    return not all(ord(c) < 128 for c in s)

# Function to replace '@' followed by any word with '@' followed by '^' of the same length as the word
def replace_at_word(text):
    return re.sub(r'@(\w+)', lambda x: '@' + '^' * len(x.group(1)), text)

df['text_cleaned'] = df['text'].apply(replace_at_word)

def refine_labels_and_entities(row):
    new_labels = []
    new_entities = []
    for label, entity in zip(row['updated_labels2'], row['merged_entities2']):
        # Extract the part of text_cleaned using start and end offsets
        text_part = row['text_cleaned'][entity['start_offset']:entity['end_offset']]
        
        # if not has_non_ascii(label) and '^' not in text_part and label != in ['S', 'the]':
        if not has_non_ascii(label) and '^' not in text_part and label not in ['S']:

            new_labels.append(label)
            new_entities.append(entity)
    row['updated_labels2'] = new_labels
    row['merged_entities2'] = new_entities
    return row

df = df.apply(refine_labels_and_entities, axis=1)

# %%
def check_is_match(row):
    is_match_list = []
    for entity in row['merged_entities2']:
        # Extract the part of text using start and end offsets
        text_part = row['text'][entity['start_offset']:entity['end_offset']]
        
        # Check if text_part matches the text value in merged_entities2 for that entity
        is_match_list.append(text_part == entity['text'])
    return is_match_list

df['is_match'] = df.apply(check_is_match, axis=1)

# %%
fp_df = pd.read_csv('/Users/muhammadluay/Library/Containers/com.microsoft.Excel/Data/Desktop/Zindi/IDRISI-main/scripts/EDA/fp_analysis.csv')

# %%
# Filter fp_df for rows where FPP <= 50
filtered_fp_df = fp_df[fp_df['False Positive Percentage'] <= 50]

def get_fp_pred_entity(text_cleaned):
    entities = []
    for label in filtered_fp_df['Label']:
        start_offset = text_cleaned.lower().find(label.lower())
        if start_offset != -1:
            end_offset = start_offset + len(label)
            entities.append({'text': text_cleaned[start_offset:end_offset], 'start_offset': start_offset, 'end_offset': end_offset})
    return entities

df['fp_pred_entity'] = df['text_cleaned'].apply(get_fp_pred_entity)

# %%
# Creating explicit ts and te columns in the DataFrame
for i in range(1, 18):  # Assuming up to 17 entities
    df[f'ts{i}'] = None
    df[f'te{i}'] = None

# Function to fill ts and te columns based on merged_entities2
def fill_ts_te(row):
    entities = row['merged_entities2']
    
    for i, entity in enumerate(entities, start=1):
        if i > 17:  # We're only considering up to 17 entities
            break
        
        # Assign start_offset and end_offset to ts and te columns respectively
        row[f'ts{i}'] = entity['start_offset']
        row[f'te{i}'] = entity['end_offset']
    
    return row

# Applying the fill_ts_te function to each row in the DataFrame
df = df.apply(fill_ts_te, axis=1)

# Print the DataFrame to check the updated ts and te columns
cols_to_display = ['merged_entities2'] + [f'ts{i}' for i in range(1, 18)] + [f'te{i}' for i in range(1, 18)]


# %%
df[['merged_entities2', 'ts1', 'te1', 'ts2', 'te2', 'ts3', 'te3', 'ts4', 'te4']]


# %%
df = df.fillna(0)

# %%
df[cols_to_display].head(50)

# %%
get_ids = '/Users/muhammadluay/Desktop/Zindi/IDRISI-main/scripts/all_test.jsonl'

# %%
ids_df = pd.read_json(get_ids, lines=True, dtype={'tweet_id': str})


# %%
ids_df

# %%
df['tweet_id'] = ids_df['tweet_id']

# %%
df

# %%
df[['tweet_id'] + cols_to_display]

# %%
# Holds the transformed data
transformed_data = []

# Iterating over each row in the DataFrame
for index, row in df.iterrows():
    tweet_id = row['tweet_id']

    for loc in range(1, 18):  # For each location from 1 to 17
        # Appending ts data
        start_id = f"ID_{tweet_id}_loc{loc}_start"
        start_value = row[f'ts{loc}']
        transformed_data.append([start_id, start_value])

        # Appending te data
        end_id = f"ID_{tweet_id}_loc{loc}_end"
        end_value = row[f'te{loc}']
        transformed_data.append([end_id, end_value])

# Creating a new DataFrame from the transformed data
transformed_df = pd.DataFrame(transformed_data, columns=['Tweet_ID', 'Value'])


# %%
transformed_df

# %%
transformed_df.to_csv('submit_suweilah17.csv', index=False)


