#This module handles matching target values with stored ads and

import pandas as pd
from typing import Dict, Optional, List
import os
import random


class AdRecommendationEngine:
    
    def __init__(self, ads_database_path: str = "Classification model dataset.csv"):
        self.ads_database_path = ads_database_path
        self.ads_df = None
        self.is_loaded = False
        
    def load_ads_database(self) -> bool:
        try:
            if not os.path.exists(self.ads_database_path):
                print(f"Error: Ads database not found at {self.ads_database_path}")
                return False
            
            print(f"Loading ads database from {self.ads_database_path}...")
            
            self.ads_df = pd.read_csv(self.ads_database_path)
            
            required_cols = ['pid', 'ad_title', 'target_age_group', 
                           'target_gender', 'target_mood', 'target_weather']
            
            missing_cols = [col for col in required_cols if col not in self.ads_df.columns]
            if missing_cols:
                print(f"Error: Missing required columns: {missing_cols}")
                return False
            
            self.is_loaded = True
            print(f"Loaded {len(self.ads_df)} ads from database")
            
            return True
            
        except Exception as e:
            print(f"Error loading ads database: {str(e)}")
            return False
    
    def calculate_match_score(self, target: Dict[str, str], ad: pd.Series) -> int:
        score = 0
        
        if str(target.get('target_age_group', '')).lower() == str(ad['target_age_group']).lower():
            score += 1
        if str(target.get('target_gender', '')).lower() == str(ad['target_gender']).lower():
            score += 1
        if str(target.get('target_mood', '')).lower() == str(ad['target_mood']).lower():
            score += 1
        if str(target.get('target_weather', '')).lower() == str(ad['target_weather']).lower():
            score += 1
        
        # Add ad_type matching if both target has it and ad has it
        if 'target_ad_type' in target and target.get('target_ad_type'):
            if 'ad_type' in ad.index and pd.notna(ad['ad_type']):
                if str(target['target_ad_type']).lower() == str(ad['ad_type']).lower():
                    score += 1
            
        return score
    
    def find_best_ad(self, target_values: Dict[str, str], debug: bool = False) -> Optional[Dict]:
        if not self.is_loaded:
            print("Ads database not loaded. Call load_ads_database() first.")
            return None
        
        try:
            if debug:
                print(f"\nDEBUG: Target values received:")
                for key, value in target_values.items():
                    print(f"  {key}: '{value}' (type: {type(value).__name__})")
            
            self.ads_df['match_score'] = self.ads_df.apply(
                lambda row: self.calculate_match_score(target_values, row),
                axis=1
            )
            
            max_score = self.ads_df['match_score'].max()
            best_matches = self.ads_df[self.ads_df['match_score'] == max_score]
            
            if debug:
                print(f"\nDEBUG: Max score found: {max_score}/4")
                print(f"  Number of ads with max score: {len(best_matches)}")
                if len(best_matches) <= 5:
                    print(f"\n  Top matches:")
                    for idx, row in best_matches.iterrows():
                        print(f"    {row['ad_title'][:50]}... | {row['target_age_group']}/{row['target_gender']}/{row['target_mood']}/{row['target_weather']}")
            
            best_ad = best_matches.sample(n=1).iloc[0]
            
            # Determine max possible score based on whether ad_type was requested
            max_score = 5 if ('target_ad_type' in target_values and target_values.get('target_ad_type')) else 4
            
            result = {
                'pid': best_ad['pid'],
                'ad_title': best_ad['ad_title'],
                'target_age_group': best_ad['target_age_group'],
                'target_gender': best_ad['target_gender'],
                'target_mood': best_ad['target_mood'],
                'target_weather': best_ad['target_weather'],
                'match_score': int(best_ad['match_score']),
                'max_possible_score': max_score
            }
            
            # Add ad_type if available in the dataset
            if 'ad_type' in best_ad.index and pd.notna(best_ad['ad_type']):
                result['ad_type'] = best_ad['ad_type']
            
            return result
            
        except Exception as e:
            print(f"Error finding best ad: {str(e)}")
            return None
    
    def find_top_n_ads(self, target_values: Dict[str, str], n: int = 5) -> List[Dict]:
        if not self.is_loaded:
            print("Ads database not loaded. Call load_ads_database() first.")
            return []
        
        try:
            self.ads_df['match_score'] = self.ads_df.apply(
                lambda row: self.calculate_match_score(target_values, row),
                axis=1
            )
            
            top_ads = self.ads_df.nlargest(n, 'match_score')
            
            # Determine max possible score based on whether ad_type was requested
            max_score = 5 if ('target_ad_type' in target_values and target_values.get('target_ad_type')) else 4
            
            results = []
            for _, ad in top_ads.iterrows():
                result = {
                    'pid': ad['pid'],
                    'ad_title': ad['ad_title'],
                    'target_age_group': ad['target_age_group'],
                    'target_gender': ad['target_gender'],
                    'target_mood': ad['target_mood'],
                    'target_weather': ad['target_weather'],
                    'match_score': int(ad['match_score']),
                    'max_possible_score': max_score
                }
                
                # Add ad_type if available
                if 'ad_type' in ad.index and pd.notna(ad['ad_type']):
                    result['ad_type'] = ad['ad_type']
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error finding top ads: {str(e)}")
            return []
    
    def get_database_stats(self) -> Optional[Dict]:
        if not self.is_loaded:
            return None
        
        stats = {
            'total_ads': len(self.ads_df),
            'unique_ads': self.ads_df['ad_title'].nunique(),
            'age_groups': self.ads_df['target_age_group'].value_counts().to_dict(),
            'genders': self.ads_df['target_gender'].value_counts().to_dict(),
            'moods': self.ads_df['target_mood'].value_counts().to_dict(),
            'weather': self.ads_df['target_weather'].value_counts().to_dict()
        }
        
        # Add ad_type stats if column exists
        if 'ad_type' in self.ads_df.columns:
            stats['ad_types'] = self.ads_df['ad_type'].value_counts().to_dict()
        
        return stats


if __name__ == "__main__":
    engine = AdRecommendationEngine()
    
    if engine.load_ads_database():
        target = {
            'target_age_group': '18-39',
            'target_gender': 'Female',
            'target_mood': 'neutral',
            'target_weather': 'sunny'
        }
        
        print("\n" + "="*60)
        print("Testing Ad Recommendation")
        print("="*60)
        print("\nTarget Values:")
        for key, value in target.items():
            print(f"  {key}: {value}")
        
        print("\n🔍 Finding best matching ad...")
        best_ad = engine.find_best_ad(target)
        
        if best_ad:
            print(f"\nBest Match Found!")
            print(f"  Ad Title: {best_ad['ad_title']}")
            print(f"  Product ID: {best_ad['pid']}")
            print(f"  Match Score: {best_ad['match_score']}/{best_ad['max_possible_score']}")
            print(f"\n  Categories:")
            print(f"  - Age Group: {best_ad['target_age_group']}")
            print(f"  - Gender: {best_ad['target_gender']}")
            print(f"  - Mood: {best_ad['target_mood']}")
            print(f"  - Weather: {best_ad['target_weather']}")
        
        print("\n" + "="*60)
        print("Database Statistics")
        print("="*60)
        stats = engine.get_database_stats()
        if stats:
            print(f"\nTotal Ads: {stats['total_ads']}")
            print(f"Unique Ad Titles: {stats['unique_ads']}")
