# %%
import json
import pandas as pd
from collections import Counter

# Path to the JSONL file
file_path = '/Users/muhammadluay/Desktop/Zindi/IDRISI-main/scripts/all_jsonl.jsonl'

def extract_data_from_file(file_path):
    all_location_texts = set()
    lines_data = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            lines_data.append(data)
            location_mentions = data.get('location_mentions', [])
            for loc_mention in location_mentions:
                all_location_texts.add(loc_mention.get('text', '').lower())
    return all_location_texts, lines_data

all_location_texts, lines_data = extract_data_from_file(file_path)

false_positives_data = []
for data in lines_data:
    text = data.get('text').lower()
    location_mentions = data.get('location_mentions', [])
    specific_location_texts = {loc_mention.get('text', '').lower() for loc_mention in location_mentions}
    
    false_positives_for_line = [loc_text for loc_text in all_location_texts if loc_text in text and loc_text not in specific_location_texts]
    if false_positives_for_line:
        data['false_positives'] = false_positives_for_line
        false_positives_data.append(data)

false_positives_df = pd.DataFrame(false_positives_data)

# Counting false positives
false_positive_counts = Counter()
for fps in false_positives_df['false_positives']:
    fps_lower = [fp.lower() for fp in fps]
    false_positive_counts.update(fps_lower)

# Counting correct mentions
correct_counts = Counter()
for data in lines_data:
    location_mentions = data.get('location_mentions', [])
    for location in location_mentions:
        correct_counts[location['text'].lower()] += 1

# Creating DataFrame for analysis
fp_analysis = pd.DataFrame({
    'Label': list(correct_counts.keys()),
    'Correct Count': list(correct_counts.values())
})
fp_analysis['False Positive Count'] = fp_analysis['Label'].map(false_positive_counts).fillna(0).astype(int)
fp_analysis['Total Count'] = fp_analysis['Correct Count'] + fp_analysis['False Positive Count']
fp_analysis['False Positive Percentage'] = (fp_analysis['False Positive Count'] / fp_analysis['Total Count']) * 100

fp_analysis.to_csv('fp_analysis.csv', index=False)



