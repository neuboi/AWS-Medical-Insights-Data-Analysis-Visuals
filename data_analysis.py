"""

Medical Insights Data Analysis 
Date: 10-20-26

"""

from collections import Counter
from importlib.metadata import files
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from wordcloud import STOPWORDS

AWS_MI_ENTITY_CATEGORIES = [
        'BEHAVIORAL_ENVIRONMENTAL_SOCIAL',
        'ANATOMY', 
        'TEST_TREATMENT_PROCEDURE', 
        'MEDICAL_CONDITION', 
        'MEDICATION', 
        'PROTECTED_HEALTH_INFORMATION', 
        'TIME_EXPRESSION'
        ]

medical_insights_transcript_files = os.listdir('json_files')
word_files = os.listdir('word_list')

all_scores_in_dataset = []
all_categories_in_dataset = []
all_types_in_dataset = []

text_length_per_category = {}
text_per_category = {}
text_length_throughout_dataset = [] 

unique_categories_in_dataset = set()
unique_types_in_dataset = set()

score_distribution_per_category = {}
score_distribution_per_types = {}

types_count_per_transcript = []
categories_count_per_transcript = []

words_to_omit = {}

def create_custom_stopwords():
    stopwords = STOPWORDS

    wordList = {
            'um', 'uh', 'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'from', 'has',
            'have', 'he', 'her', 'his', 'i', 'in', 'is', 'it', 'its', 'me', 'my', 'not',
            'of', 'on', 'or', 'our', 'she', 'so', 'that', 'the', 'their', 'them', 'they',
            'this', 'to', 'was', 'we', 'were', 'what', 'with', 'you', 'your',
        }

    for i in range(0, len(word_files)):
        with open(f'word_list/{word_files[i]}', 'r') as file:
            for line in file:
                word = line.strip()
                if word: stopwords.add(word.lower())
    
    for word in list(wordList):
        stopwords.add(word.lower())

    return stopwords

def get_entity_frequency():
    words_to_omit = create_custom_stopwords()
    entity_frequency = {}

    for cat in AWS_MI_ENTITY_CATEGORIES:
        entity_frequency[cat] = Counter()

    for i in range(0, len(medical_insights_transcript_files)):
        with open(f'json_files/{medical_insights_transcript_files[i]}', 'r') as file:
            data = json.load(file)
            entities = data['Entities']
            for entity in entities:
                if entity['Text'].lower() not in words_to_omit: 
                    entity_frequency[entity['Category']][entity['Text']] += 1

    return entity_frequency

def get_statistics(window=0, score_range=None):

    clear_statistics()

    for i in range(0, len(medical_insights_transcript_files)):

        types_count_per_transcript.append(set()) 
        categories_count_per_transcript.append(set())

        with open(f'json_files/{medical_insights_transcript_files[i]}', 'r') as file:
            data = json.load(file)
            entities = data['Entities']
            entity_set = get_entities_in_sliding_char_window(entities, window, score_range)

            for j in entity_set:
                if j['Category'] != 'PROTECTED_HEALTH_INFORMATION': # Exclude PHI
                    all_scores_in_dataset.append(j['Score'])

                    all_categories_in_dataset.append(j['Category'])
                    unique_categories_in_dataset.add(j['Category'])
                    categories_count_per_transcript[i].add(j['Category'])

                    if j['Category'] not in score_distribution_per_category: score_distribution_per_category[j['Category']] = [];
                    score_distribution_per_category[j['Category']].append(j['Score'])
                    
                    all_types_in_dataset.append(j['Type'])
                    unique_types_in_dataset.add(j['Type'])  # Add type to the set
                    types_count_per_transcript[i].add(j['Type'])

                    if j['Type'] not in score_distribution_per_types: score_distribution_per_types[j['Type']] = [];
                    score_distribution_per_types[j['Type']].append(j['Score'])

                    text_length_throughout_dataset.append(len(j['Text']))  # Store the length of the text
                    
                    if j['Category'] not in text_length_per_category: text_length_per_category[j['Category']] = [];
                    text_length_per_category[j['Category']].append(len(j['Text']))

                    if j['Category'] not in text_per_category: text_per_category[j['Category']] = [];
                    text_per_category[j['Category']].append(j['Text'])


    return all_categories_in_dataset, unique_categories_in_dataset, score_distribution_per_category, text_per_category

def clear_statistics():
    global all_scores_in_dataset, all_categories_in_dataset, all_types_in_dataset, \
    text_length_per_category, text_length_throughout_dataset, unique_categories_in_dataset, \
    unique_types_in_dataset, score_distribution_per_category, score_distribution_per_types, \
    types_count_per_transcript, text_per_category, categories_count_per_transcript

    all_scores_in_dataset = []
    all_categories_in_dataset = []
    all_types_in_dataset = []

    text_length_per_category = {}
    text_per_category = {}
    text_length_throughout_dataset = [] 

    unique_categories_in_dataset = set()
    unique_types_in_dataset = set()

    score_distribution_per_category = {}
    score_distribution_per_types = {}

    types_count_per_transcript = []
    categories_count_per_transcript = []

def get_entities_in_sliding_char_window(entities, character_window, score_threshold):
    entity_lookup = {}
    entity_candidates = []

    for entity in entities:

        last_entity = entity_lookup[entity['Category']][entity['Text']] if \
            entity['Category'] in entity_lookup and \
                entity['Text'] in entity_lookup[entity['Category']] \
                    else None
        
        if last_entity is None:
            if entity['Category'] not in entity_lookup: 
                entity_lookup[entity['Category']] = {}
            entity_lookup[entity['Category']][entity['Text']] = entity
            entity_candidates.append(entity)
            continue

        else:
            if is_entity_in_char_window(entity, last_entity, character_window):
                entity_lookup[entity['Category']][entity['Text']] = entity
                continue
            else:
                entity_lookup[entity['Category']][entity['Text']] = entity
                entity_candidates.append(entity)

    if score_threshold is not None: entity_candidates = [d for d in entity_candidates if d['Score'] >= score_threshold]

    return entity_candidates

def is_entity_in_char_window(new_entity, last_entity, window):
    return new_entity['PlainBeginOffset'] - last_entity['PlainBeginOffset'] < window

if __name__ == '__main__':
    get_statistics()