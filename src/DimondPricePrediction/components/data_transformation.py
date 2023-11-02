import pandas as pd
import os
import sys
import numpy as np

from dataclasses import dataclass
from src.DimondPricePrediction.exception import customexception
from src.DimondPricePrediction.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.DimondPricePrediction.utils.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()
    def get_data_transformation(self):
        try:
            logging.info("Data Transformation initiated")

            # Define which columns are categorical and which are numerical
            categorical_columns= ['cut', 'color', 'clarity']
            numerical_columns= ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the ranking of ordinal categories
            cut_category= ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            clarity_category= ['I1','SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
            color_category= ['J', 'I', 'H', 'G', 'F', 'E', 'D']

            logging.info("Pipeline initiated")

            # Numerical Pipeline
            num_pipeline= Pipeline(
                steps= [
                ('imputer', SimpleImputer(strategy= 'median')),
                ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline= Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy= 'most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories= [cut_category,color_category,clarity_category])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor= ColumnTransformer([
                ('num_pipeline',num_pipeline, numerical_columns),
                ('cat_pipeline',cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Error occured in initiate_data_transformation")
            raise customexception(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Reading train and test data completed")
            logging.info(f'Train Dataframe Head: \n {train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n {test_df.head().to_string()}')

            preprocessing_obj= self.get_data_transformation()

            target_column_name= 'price'
            drop_columns= [target_column_name, 'price']

            input_feature_train_df=train_df.drop(columns= drop_columns,axis= 1)
            target_feature_train_df= train_df[target_column_name]
            input_feature_test_df= test_df.drop(columns=drop_columns,axis= 1)
            target_feature_test_df= test_df[target_column_name]

            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            logging.info ("Applying preprocessing on training and testing datasets")

            train_arr= np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )

            logging.info("preprocessor pickle file saved")
            return (
                train_arr,
                test_arr
            )

        except Exception as e:
            logging.info("Error occured in initiate_data_transformation")
            raise customexception(e,sys)