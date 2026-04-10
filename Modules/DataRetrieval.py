import pandas as pd
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataRetrieval:
    def __init__(self, metadata_path):
        self.df = pd.read_csv(metadata_path, sep='\t')

    def parse_and_set_defaults(self, structured_data):
        defaults = {
            'accent': 'USA',
            'language': 'EN',
            'age': '[20, 40]',
            'gender': 'M'
        }
        specified = {}
        # Age: keep exact or range as is, default to range if missing or 'unspecified'
        age_val = structured_data.get('age', None)
        if age_val and str(age_val).lower() != 'unspecified':
            specified['age'] = True
            structured_data['age'] = age_val
        else:
            specified['age'] = False
            # print("[set_default] SET AGE AS DEFAULT")
            structured_data['age'] = defaults['age']
        # Other fields
        for key, value in defaults.items():
            if key == 'age':
                continue
            if key not in structured_data or not structured_data[key] or str(structured_data[key]).lower() == 'unspecified':
                structured_data[key] = value
                # print(f"[set_default] SET {key} AS DEFAULT {value}")
                specified[key] = False
            else:
                specified[key] = True
        return structured_data, specified


    def similarity_search(self, df, target_text, top_n=5):
        # print(f"[similarity_search] Starting similarity search for: '{target_text}'")
        if 'transcript' not in df.columns or df.empty:
            print("[similarity_search] No 'transcript' column or empty DataFrame. Returning top_n rows.")
            return df.head(top_n)
        texts = df['transcript'].fillna("").astype(str).tolist()
        # print(f"[similarity_search] Number of candidate texts: {len(texts)}")
        vectorizer = TfidfVectorizer().fit(texts + [target_text])
        tfidf_matrix = vectorizer.transform(texts + [target_text])
        # print("[similarity_search] TF-IDF matrix computed.")
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        top_indices = similarities.argsort()[::-1][:top_n]
        return df.iloc[top_indices]


    def parse_age_value(self, age_val):
        # Handles string, tuple, list, or int
        if isinstance(age_val, list) and len(age_val) == 2:
            return age_val
        if isinstance(age_val, int):
            return age_val
        if isinstance(age_val, str):
            s = age_val.strip()
            # Range string: [40, 50]
            if s.startswith('[') and s.endswith(']'):
                try:
                    age_range = [int(x) for x in s[1:-1].split(',')]
                    return age_range
                except Exception:
                    pass
            # Int string
            try:
                return int(s)
            except Exception:
                print("[parse_age_value]: Invalid format")
                pass
        return age_val

    def filter_by_age(self, query, age_val, min_speakers=10):
        age_parsed = self.parse_age_value(age_val)
        # print(f"[find_relevant] Filtering by age: {age_parsed}")
        try:
            exact_age = None
            age_range = None
            # List/range: [start, end]
            if isinstance(age_parsed, list) and len(age_parsed) == 2:
                exact_age = None
                age_range = age_parsed
            # Exact age int
            elif isinstance(age_parsed, int):
                exact_age = age_parsed
                age_range = None
            else:
                print(f"Parsed data is type [{type(age_parsed)}]. Type is invalid.")
                return query

            n_speakers = 0

            # Try using exact age first
            if exact_age is not None:
                mask_exact = (query['age'] == exact_age)
                n_exact = mask_exact.sum()
                n_speakers = query[mask_exact]['speaker'].nunique()
                # print(f"[filter_by_age] Candidates after exact age ({exact_age}) filter: {n_exact}, speakers: {n_speakers}")
                # If exact age speakers are enough, use them directly
                if n_speakers > min_speakers:
                    return query[mask_exact]
                
            # Use age range
            else:
                mask_range = (query['age'] >= age_range[0]) & (query['age'] <= age_range[1])
                n_range = mask_range.sum()
                n_speakers = query[mask_range]['speaker'].nunique()
                # print(f"[filter_by_age] Candidates after age range ({age_range}) filter: {n_range}, speakers: {n_speakers}")

                if n_speakers > min_speakers:
                    return query[mask_range]

            # Find min_speakers closest to the age range (use midpoint)
            if exact_age:
                mid_age = exact_age
            else:
                mid_age = (age_range[0] + age_range[1]) // 2

            # Find min_speakers closest to the exact age, then sort them from closest to furthest.
            closest = query.copy()
            closest['age_diff'] = (closest['age'] - mid_age).abs() # calculate age difference and abs
            closest_sorted = closest.sort_values('age_diff') # sort from closest to furthest
            unique_speakers = closest_sorted['speaker'].unique() # find unique closest speakers
            selected_speakers = unique_speakers[:min_speakers] # select top N speakers
            selected = closest_sorted[closest_sorted['speaker'].isin(selected_speakers)]
            # if exact_age:
            #     print(f"[filter_by_age] Selected {len(selected_speakers)} closest speakers to age {exact_age}, total rows: {len(selected)}, final age range: [{selected['age'].min()}, {selected['age'].max()}]")
            # else:
            #     print(f"[filter_by_age] Selected {len(selected_speakers)} closest speakers to age range {age_range[0]}-{age_range[1]}, total rows: {len(selected)}, final age range: [{selected['age'].min()}, {selected['age'].max()}]")

            return selected.sort_values('age_diff')

                
        except Exception as e:
            print(f"[find_relevant] Error in age filtering: {e}")
            return query

    def find_relevant(self, structured_data, top_n=10):
        structured_data, specified = self.parse_and_set_defaults(structured_data)
        query = self.df
        # print(f"[find_relevant] Initial candidates: {len(query)}")
        # Priority: gender > accent > age > language
        # priority = ['gender', 'accent', 'age', 'language']

        priority = ['gender', 'accent', 'age']

        unspecified = []
        # Filter first based on non-defaults
        for key in priority:
            if not specified.get(key, False):
                # print(f"[find_relevant] Skipping filter for '{key}' (not specified)")
                unspecified.append(key)
                continue  # Ignore filter if not specified
            prev_query = query.copy()
            # print(f"[find_relevant] Applying filter for '{key}' with value '{structured_data[key]}'")
            if key == 'age':
                query = self.filter_by_age(query, structured_data['age'])
            else:
                query = query[query[key].astype(str).str.lower() == str(structured_data[key]).lower()]
                # print(f"[find_relevant] Candidates after '{key}' filter: {len(query)}")
            # If results drop below 5, revert to previous query
            if len(query) < 5:
                # print(f"[find_relevant] Reverting '{key}' filter, candidates dropped below 5.")
                query = prev_query
        
        # Filter based on defaults
        for key in unspecified:
            if key == 'age':
                query = self.filter_by_age(query, structured_data['age'])
            else:
                query = query[query[key].astype(str).str.lower() == str(structured_data[key]).lower()]
                # print(f"[find_relevant] Candidates after DEFAULT '{key}' filter: {len(query)}")
            # If results drop below 5, revert to previous query
            if len(query) < top_n:
                # print(f"[find_relevant] Reverting '{key}' filter, candidates dropped below 5.")
                query = prev_query


        # Similarity search on text if provided
        if 'text' in structured_data and structured_data['text']:
            # print(f"[find_relevant] Performing similarity search for text: '{structured_data['text']}'")
            query = self.similarity_search(query, structured_data['text'], top_n=top_n)
        else:
            # print(f"[find_relevant] No text provided, returning top {top_n} candidates.")
            query = query.head(top_n)
        # print(f"[find_relevant] Final candidates: {len(query)}")

        # [NEW] SORT BY MOS IF THERE IS SG/MY ACCENTS
        shortlisted_accents = query['accent'].str.upper().unique()
        if any(acc in ['SG', 'MY'] for acc in shortlisted_accents):
            # Sort by 'mos' descending, then take top_n
            query = query.sort_values(by='mos', ascending=False).head(top_n)

        return query
