#!/usr/bin/env python
"""
Ranking Model for Job Matching Recommendation System
Second stage of the recommendation pipeline: ranks candidate-job pairs
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm
import joblib
import random

class FeatureExtractor:
    """Feature extractor for candidate-job pairs"""
    
    def __init__(self):
        self.job_seekers = None
        self.job_listings = None
        self.retrieval_results = None
    
    def load_data(self, job_seekers_file="data/raw/job_seekers.json", 
                 job_listings_file="data/raw/job_listings.json",
                 retrieval_results_file="data/retrieval_results.json"):
        """Load raw data and retrieval results"""
        # Load job seekers
        with open(job_seekers_file, "r") as f:
            self.job_seekers = {js["id"]: js for js in json.load(f)}
        
        # Load job listings
        with open(job_listings_file, "r") as f:
            self.job_listings = {jl["id"]: jl for jl in json.load(f)}
        
        # Load retrieval results
        with open(retrieval_results_file, "r") as f:
            self.retrieval_results = json.load(f)
        
        print(f"Loaded {len(self.job_seekers)} job seekers, {len(self.job_listings)} job listings, "
              f"and {len(self.retrieval_results)} retrieval results")
    
    def extract_features(self):
        """Extract features for each candidate-job pair"""
        features = []
        
        print("Extracting features for candidate-job pairs...")
        for result in tqdm(self.retrieval_results):
            job_id = result["job_id"]
            candidate_id = result["candidate_id"]
            
            # Get job and candidate data
            job = self.job_listings.get(job_id)
            candidate = self.job_seekers.get(candidate_id)
            
            if not job or not candidate:
                continue  # Skip if job or candidate not found
            
            # Base features from retrieval
            feature_dict = {
                "job_id": job_id,
                "candidate_id": candidate_id,
                "retrieval_rank": result["rank"],
                "similarity_score": result["similarity_score"],
            }
            
            # Add skill match features
            required_skills = set(job["required_skills"])
            preferred_skills = set(job.get("preferred_skills", []))
            candidate_skills = set(candidate["skills"])
            
            feature_dict["required_skills_match_ratio"] = len(required_skills.intersection(candidate_skills)) / len(required_skills) if required_skills else 0
            feature_dict["preferred_skills_match_ratio"] = len(preferred_skills.intersection(candidate_skills)) / len(preferred_skills) if preferred_skills else 0
            feature_dict["total_skills_match_ratio"] = len((required_skills.union(preferred_skills)).intersection(candidate_skills)) / len(required_skills.union(preferred_skills)) if required_skills.union(preferred_skills) else 0
            
            # Experience match
            job_min_experience = job.get("min_experience_months", 0)
            candidate_experience = candidate.get("total_experience_months", 0)
            feature_dict["experience_match"] = 1 if candidate_experience >= job_min_experience else candidate_experience / job_min_experience if job_min_experience else 1
            feature_dict["experience_ratio"] = candidate_experience / job_min_experience if job_min_experience else 1
            
            # Education match
            education_levels = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
            job_education = job.get("education_requirement", "High School")
            candidate_education = candidate.get("education", {}).get("degree", "High School")
            
            job_education_level = education_levels.get(job_education, 1)
            candidate_education_level = education_levels.get(candidate_education, 1)
            
            feature_dict["education_match"] = 1 if candidate_education_level >= job_education_level else 0
            feature_dict["education_level_diff"] = candidate_education_level - job_education_level
            
            # Location match
            job_location = job.get("location", "")
            candidate_location = candidate.get("location", "")
            candidate_preferred_locations = candidate.get("preferred_locations", [])
            
            feature_dict["location_exact_match"] = 1 if job_location == candidate_location else 0
            feature_dict["location_preferred_match"] = 1 if job_location in candidate_preferred_locations else 0
            
            # Remote work match
            job_remote = job.get("remote", False)
            candidate_remote = candidate.get("open_to_remote", False)
            
            feature_dict["remote_match"] = 1 if (job_remote and candidate_remote) or not job_remote else 0
            
            # Title match
            job_title = job.get("title", "")
            candidate_preferred_titles = candidate.get("preferred_job_titles", [])
            
            feature_dict["title_preferred_match"] = 1 if job_title in candidate_preferred_titles else 0
            
            # Salary match
            job_salary_min = job.get("salary_range", {}).get("min", 0)
            job_salary_max = job.get("salary_range", {}).get("max", 0)
            candidate_salary_expectation = candidate.get("salary_expectation", 0)
            
            if job_salary_min and job_salary_max and candidate_salary_expectation:
                if candidate_salary_expectation >= job_salary_min and candidate_salary_expectation <= job_salary_max:
                    feature_dict["salary_match"] = 1
                else:
                    # Calculate ratio of how far outside the range
                    if candidate_salary_expectation < job_salary_min:
                        feature_dict["salary_match"] = candidate_salary_expectation / job_salary_min
                    else:
                        feature_dict["salary_match"] = job_salary_max / candidate_salary_expectation
            else:
                feature_dict["salary_match"] = 0.5  # Neutral if information missing
            
            features.append(feature_dict)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        print(f"Extracted features for {len(features_df)} candidate-job pairs")
        
        return features_df

class RankingModel:
    """Ranking model that scores and orders candidate-job pairs"""
    
    def __init__(self, model_type="xgboost"):
        """
        Initialize ranking model
        
        Args:
            model_type (str): Type of model to use ('xgboost', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
    
    def generate_synthetic_labels(self, features_df, positive_ratio=0.1):
        """
        Generate synthetic labels for training
        In a real system, these would be collected from user feedback
        
        Args:
            features_df (DataFrame): Feature DataFrame
            positive_ratio (float): Ratio of positive examples
        
        Returns:
            DataFrame: Features with added label column
        """
        # Copy dataframe to avoid modifying the original
        labeled_df = features_df.copy()
        
        # Generate synthetic labels based on feature values
        scores = (
            labeled_df["similarity_score"] * 0.3 +
            labeled_df["required_skills_match_ratio"] * 0.3 +
            labeled_df["experience_match"] * 0.2 +
            labeled_df["education_match"] * 0.1 +
            labeled_df["location_exact_match"] * 0.1 +
            labeled_df["remote_match"] * 0.05 +
            labeled_df["title_preferred_match"] * 0.05 +
            labeled_df["salary_match"] * 0.1
        )
        
        # Add some random noise
        scores += np.random.normal(0, 0.1, size=len(scores))
        
        # Normalize scores to [0, 1]
        min_score = scores.min()
        max_score = scores.max()
        scores = (scores - min_score) / (max_score - min_score)
        
        # Convert to binary labels based on threshold to get desired positive ratio
        threshold = scores.quantile(1 - positive_ratio)
        labeled_df["label"] = (scores >= threshold).astype(int)
        
        # Add relevance score (can be used for NDCG)
        labeled_df["relevance"] = scores
        
        print(f"Generated synthetic labels with {labeled_df['label'].sum()} positive examples "
              f"({100 * labeled_df['label'].mean():.2f}% positive)")
        
        return labeled_df
    
    def prepare_data(self, features_df, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            features_df (DataFrame): Feature DataFrame
            test_size (float): Proportion of data to use for testing
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, relevance_train, relevance_test)
        """
        # Drop ID columns
        self.feature_columns = [col for col in features_df.columns 
                              if col not in ['job_id', 'candidate_id', 'label', 'relevance']]
        
        # Split data
        train_df, test_df = train_test_split(features_df, test_size=test_size, random_state=42)
        
        X_train = train_df[self.feature_columns].values
        y_train = train_df['label'].values
        relevance_train = train_df['relevance'].values
        
        X_test = test_df[self.feature_columns].values
        y_test = test_df['label'].values
        relevance_test = test_df['relevance'].values
        
        return X_train, X_test, y_train, y_test, relevance_train, relevance_test, train_df, test_df
    
    def train(self, X_train, y_train):
        """
        Train ranking model
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
        """
        if self.model_type == "xgboost":
            # Train XGBoost model
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        elif self.model_type == "linear":
            # Train logistic regression model
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1.0, random_state=42)
            self.model.fit(X_train, y_train)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(self, X):
        """
        Predict relevance scores
        
        Args:
            X (ndarray): Features
        
        Returns:
            ndarray: Predicted scores
        """
        if self.model_type == "xgboost":
            # Get probability of positive class
            return self.model.predict_proba(X)[:, 1]
        
        elif self.model_type == "linear":
            return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test, relevance_test, test_df):
        """
        Evaluate model performance
        
        Args:
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
            relevance_test (ndarray): Test relevance scores
            test_df (DataFrame): Test DataFrame with job and candidate IDs
        
        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
        
        # Predict scores
        y_pred_proba = self.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate basic metrics
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "average_precision": average_precision_score(y_test, y_pred_proba)
        }
        
        # Calculate NDCG for each job
        job_ids = test_df["job_id"].unique()
        ndcg_values = []
        
        for job_id in job_ids:
            job_mask = test_df["job_id"] == job_id
            if sum(job_mask) > 1:  # Need at least 2 items for ranking
                job_relevance = relevance_test[job_mask]
                job_pred_proba = y_pred_proba[job_mask]
                
                # Calculate NDCG@5 if possible, otherwise use all available items
                k = min(5, len(job_relevance))
                ndcg_value = self.ndcg_at_k(job_relevance, job_pred_proba, k)
                ndcg_values.append(ndcg_value)
        
        if ndcg_values:
            metrics["ndcg@5"] = np.mean(ndcg_values)
        
        print(f"Evaluation metrics: {metrics}")
        return metrics
    
    @staticmethod
    def ndcg_at_k(relevance, scores, k):
        """
        Calculate NDCG@k for a single query
        
        Args:
            relevance (ndarray): True relevance scores
            scores (ndarray): Predicted scores
            k (int): Number of items to consider
        
        Returns:
            float: NDCG@k value
        """
        # Sort by predicted scores
        pred_indices = np.argsort(scores)[::-1][:k]
        pred_relevance = relevance[pred_indices]
        
        # Sort by true relevance
        ideal_indices = np.argsort(relevance)[::-1][:k]
        ideal_relevance = relevance[ideal_indices]
        
        # Calculate DCG
        dcg = np.sum((2 ** pred_relevance - 1) / np.log2(np.arange(2, len(pred_relevance) + 2)))
        idcg = np.sum((2 ** ideal_relevance - 1) / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def save_model(self, model_file="models/ranking/ranking_model.joblib"):
        """Save model to file"""
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type
        }
        joblib.dump(model_data, model_file)
        print(f"Model saved to {model_file}")
    
    def load_model(self, model_file="models/ranking/ranking_model.joblib"):
        """Load model from file"""
        model_data = joblib.load(model_file)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.model_type = model_data["model_type"]
        print(f"Model loaded from {model_file}")
    
    def rank_candidates_for_jobs(self, features_df, output_file=None):
        """
        Rank candidates for each job based on model predictions
        
        Args:
            features_df (DataFrame): Feature DataFrame
            output_file (str): Output file path
        
        Returns:
            dict: Rankings by job
        """
        # Ensure we have trained model
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get features
        X = features_df[self.feature_columns].values
        
        # Predict scores
        scores = self.predict(X)
        
        # Add scores to dataframe
        features_df["predicted_score"] = scores
        
        # Group by job and rank candidates
        rankings = {}
        
        for job_id in features_df["job_id"].unique():
            job_df = features_df[features_df["job_id"] == job_id]
            job_df = job_df.sort_values("predicted_score", ascending=False)
            
            rankings[job_id] = [
                {
                    "job_id": job_id,
                    "candidate_id": row["candidate_id"],
                    "predicted_score": row["predicted_score"],
                    "rank": i + 1
                }
                for i, (_, row) in enumerate(job_df.iterrows())
            ]
        
        # Flatten rankings for output
        all_rankings = []
        for job_rankings in rankings.values():
            all_rankings.extend(job_rankings)
        
        # Save results if output file provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(all_rankings, f, indent=2)
            print(f"Rankings saved to {output_file}")
        
        return rankings

def run_ranking_pipeline():
    """Run full ranking pipeline"""
    # 1. Extract features
    extractor = FeatureExtractor()
    extractor.load_data()
    features_df = extractor.extract_features()
    
    # 2. Generate synthetic labels (in a real system, these would come from user feedback)
    labeled_df = RankingModel().generate_synthetic_labels(features_df)
    
    # 3. Train and evaluate model
    ranker = RankingModel(model_type="xgboost")
    X_train, X_test, y_train, y_test, relevance_train, relevance_test, train_df, test_df = ranker.prepare_data(labeled_df)
    ranker.train(X_train, y_train)
    metrics = ranker.evaluate(X_test, y_test, relevance_test, test_df)
    
    # 4. Save model
    ranker.save_model()
    
    # 5. Rank all candidates for all jobs
    rankings = ranker.rank_candidates_for_jobs(features_df, output_file="data/ranking_results.json")
    
    return metrics, rankings

if __name__ == "__main__":
    run_ranking_pipeline() 