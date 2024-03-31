# %%
import pandas as pd
import ast
import json

# %%
df = pd.read_csv('/Users/muhammadluay/Desktop/Zindi/IDRISI-main/scripts/clean/check_detection.csv')

# %%
df

# %%
df['entities_suweilah'][0]

# %%
def merge_words(entities_str):
    entities = json.loads(entities_str.replace("'", '"'))
    processed_entities = []
    word_cache = ''
    entity_cache = None
    
    for entity in entities:
        word = entity['word']

        # Check if the current entity should be merged with the cached entity
        should_merge = (
            entity_cache is not None and 
            entity['start'] <= entity_cache['end'] + 2
        )
        
        # Including special characters into the previous entity
        if word in ['.', '/'] and entity_cache is not None:
            entity_cache['word'] += word
            entity_cache['end'] = entity['end']
            continue
        
        if should_merge or (word.startswith('##') and entity_cache is not None):
            word = word[2:] if word.startswith('##') else ' ' + word
            word_cache += word
            entity_cache['end'] = entity['end']
            entity_cache['word'] += word
        else:
            if entity_cache is not None:
                processed_entities.append(entity_cache)
            
            word_cache = word
            entity_cache = entity.copy()
    
    if entity_cache is not None:
        processed_entities.append(entity_cache)
    
    return processed_entities

# Applying the function to process entities and create the new 'processed_suweilah_entities' column
df['processed_suweilah_entities'] = df['entities_suweilah'].apply(merge_words)

# %%
df['processed_suweilah_entities'][521]


# %%
df['location_mentions'][16447]

# %%
df['processed_suweilah_labels'] = df['processed_suweilah_entities'].apply(lambda entities: [entity['word'] for entity in entities])


# %%
df[['processed_suweilah_labels','labels', 'entities_suweilah_labels']].head(50)

# %%
malformed_labels = []  # This list will store dictionaries of malformed and correct labels

for index, row in df.iterrows():
    suweilah_entities = row['processed_suweilah_entities']
    location_mentions = ast.literal_eval(row['location_mentions'])
    
    for suweilah in suweilah_entities:
        suweilah_start = suweilah['start']
        suweilah_end = suweilah['end']
        suweilah_word = suweilah['word']
        
        for location in location_mentions:
            location_start = location['start_offset']
            location_end = location['end_offset']
            location_text = location['text']
            
            if suweilah_start == location_start:
                if suweilah_end < location_end:
                    malformed_labels.append({
                        "malformed_label": suweilah_word,
                        "correct_label": location_text
                    })
                elif suweilah_end > location_end:
                    next_loc_index = location_mentions.index(location) + 1
                    if next_loc_index < len(location_mentions) and \
                        location_mentions[next_loc_index]['start_offset'] == location_end + 1:
                        correct_label = location_text + ' ' + location_mentions[next_loc_index]['text']
                        malformed_labels.append({
                            "malformed_label": suweilah_word,
                            "correct_label": correct_label
                        })

# %%
malformed_suweilah = pd.DataFrame(malformed_labels, columns=["malformed_label", "correct_label"])


# %%
malformed_suweilah.drop_duplicates()

# %%
malformed_suweilah.to_csv('malformed_suweilah.csv')


