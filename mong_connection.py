import pandas as pd
import numpy as np
import pymongo
from sdv.tabular import CTGAN
import bson

class MongoConnection:
    def __init__(self, client, database_name, collection_name):
        self.client = client
        self.database_name = database_name
        self.collection_name = collection_name

    def establish_mongo_connection(self):
        db = self.client[self.database_name]
        collection = db[self.collection_name]
        cursor = collection.find().limit(100)
        df = pd.DataFrame(list(cursor))
        return db, df

    @staticmethod
    def is_list(column):
        return isinstance(column, list)

    @staticmethod
    def is_dict(column):
        return isinstance(column, dict)

    @staticmethod
    def split_list(row):
        return pd.Series(row)

    @staticmethod
    def combine_list_columns(row, columns, list_mapping):
        combined_values = []
        matching_columns = list_mapping[columns]
        combined_values.extend([value for value in row[matching_columns] if not pd.isna(value)])
        return combined_values

    @staticmethod    
    def combine_dict_columns(row, columns, dict_mapping):
        combined_dict = {}
        matching_columns = dict_mapping[columns]
        for col in matching_columns:
            value = row[col]
            original_col = col.replace(f"{columns}_",'')
            if pd.isna([value]).any():
                combined_dict[original_col] = None
            else:
                combined_dict[original_col] = value
        return combined_dict

    def clean_df(self, df):
        if '_id' in df.columns:
            df.drop("_id", axis=1, inplace=True)

        list_columns = []
        dict_columns = []
        mapping = {}
        list_mapping = {}
        while True:
            for column in df.columns:
                is_list_column = df[column].apply(self.is_list)
                if any(is_list_column):
                    list_columns.append(column)

                is_dict_column = df[column].apply(self.is_dict)
                if any(is_dict_column):
                    dict_columns.append(column)

            if not list_columns and not dict_columns:
                break

            
            for column in list_columns:
                max_columns = df[column].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
                column_names = [f"{column}_{i+1}" for i in range(max_columns)]
                mapping[column] = column_names
                list_mapping[column] = column_names
                # Apply the function to split the column
                df_split = df[column].apply(self.split_list)
                df_split.columns = column_names

                # Concatenate the new columns with the original DataFrame
                df = pd.concat([df, df_split], axis=1)
                # Drop the original column
                df = df.drop(column, axis=1)

            
            for column in dict_columns:
                df_temp = df[column].apply(pd.Series)
                int_keys = [key for key in df_temp.columns if isinstance(key, int)]
                if int_keys:
                    df_temp = df_temp.drop(int_keys, axis=1)
                prefix = f"{column}_"
                df_temp.columns = [f"{prefix}{col}" for col in df_temp.columns]
                mapping[column] = df_temp.columns

                df = pd.concat([df, df_temp], axis=1)
                df = df.drop(column, axis=1)

            list_columns = []
            dict_columns = []
        #Handling Decimal128 columns
        for col in df.columns:
            if isinstance(df[col].dropna().iloc[0],bson.decimal128.Decimal128):
                print(col)
                df[col] = df[col].astype(str).astype(float)

        return df, mapping, list_mapping

    def generate_synthetic_data(self,cleaned_df):
        model = CTGAN()
        model.fit(cleaned_df[:100])
        synth_data = model.sample(1000)
        return synth_data

    @staticmethod
    def clean_synthetic_data(synthetic_data, mapping, list_mapping):
        datetime_columns = synthetic_data.select_dtypes(include='datetime64').columns
        synthetic_data[datetime_columns] = synthetic_data[datetime_columns].astype(object).where(synthetic_data[datetime_columns].notnull(), None)
        for columns in reversed(mapping.keys()):
            if columns in list_mapping:
                synthetic_data[columns] = synthetic_data.apply(lambda row: 
                                                               MongoConnection.combine_list_columns(row, columns, mapping), axis=1)
                synthetic_data = synthetic_data.drop(columns=mapping[columns])
            else:
                synthetic_data[columns] = synthetic_data.apply(lambda row: 
                                                               MongoConnection.combine_dict_columns(row, columns, mapping), axis=1)
                synthetic_data = synthetic_data.drop(columns=mapping[columns])
        return synthetic_data

    def push_data_to_new_table(self, synth_data,db):
        new_collection_name = f"{self.collection_name}_test"
        new_collection = db[new_collection_name]
        data_to_insert = synth_data.to_dict(orient='records')
        new_collection.insert_many(data_to_insert)
        self.client.close()


# Example Usage:
# client = pymongo.MongoClient("uri")
# connection = MongoConnection(client, "sample_mflix", "movies")
# db, df = connection.establish_mongo_connection()
# cleaned_df, list_mapping, dict_mapping = connection.clean_df(df)
# synthetic_data = connection.generate_synthetic_data(cleaned_df)
# cleaned_synthetic_data = connection.clean_synthetic_data(synthetic_data, list_mapping, dict_mapping)
# print(cleaned_synthetic_data.columns)
# connection.push_data_to_new_table(cleaned_synthetic_data,db)
