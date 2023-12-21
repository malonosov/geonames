from sqlalchemy import create_engine, ARRAY, Float
from sqlalchemy.engine.url import URL
from sentence_transformers import SentenceTransformer
from scipy.spatial import cKDTree
from transliterate import translit
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

class GeonamesSimilarityFinder:
    def __init__(self, database_config, model_name):
        self.engine = create_engine(URL(**database_config))
        self.model = SentenceTransformer(model_name)
        self.load_data()
        self.build_tree()

    def load_data(self):
        query = "SELECT * FROM final_geonames"
        self.final_data_str = pd.read_sql_query(query, con=self.engine)

    def build_tree(self):
        embeddings_new = self.final_data_str['embedding'].tolist()
        self.city_tree = cKDTree(embeddings_new)

    def get_most_similar(self, city_name):
        translit_city_name = translit(city_name, 'ru', reversed=True)
        query_embedding = self.model.encode(translit_city_name)

        distances, indices = self.city_tree.query(query_embedding, k=5)

        similar_cities_df = self.final_data_str.iloc[indices]
        similar_cities = similar_cities_df[['geonameid', 'name', 'alternatenames', 'region', 'country_']]
        similar_cities['cosine_similarity'] = 1 - distances / 2
        similar_cities = similar_cities.sort_values(by='cosine_similarity', ascending=False)

        result = {
            'input_city': city_name,
            'similar_cities': similar_cities.to_dict(orient='records'),
        }

        return result

        

    
