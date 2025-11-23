"""
Ad Recommendation Engine
=========================
This module handles matching target values with stored ads and
recommending the best fit ad based on similarity.
"""

import pandas as pd
from typing import Dict, Optional, List
import os


class AdRecommendationEngine:
    """
    Engine for matching target values with available ads and recommending best fit.
    """
    
    def __init__(self, ads_database_path: str = "Classification model dataset.csv"):
        """
        Initialize the recommendation engine.
        
        Args:
            ads_database_path: Path to the CSV file containing all available ads
        """
        self.ads_database_path = ads_database_path
        self.ads_df = None
        self.is_loaded = False
        
    def load_ads_database(self) -> bool:
        """
        Load the ads database from CSV.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.ads_database_path):
                print(f"❌ Error: Ads database not found at {self.ads_database_path}")
                return False
            
            print(f"📦 Loading ads database from {self.ads_database_path}...")
            
            # Read CSV with proper headers
            self.ads_df = pd.read_csv(self.ads_database_path)
            
            # Verify required columns
            required_cols = ['pid', 'ad_title', 'target_age_group', 
                           'target_gender', 'target_mood', 'target_weather']
            
            missing_cols = [col for col in required_cols if col not in self.ads_df.columns]
            if missing_cols:
                print(f"❌ Error: Missing required columns: {missing_cols}")
                return False
            
            self.is_loaded = True
            print(f"✅ Loaded {len(self.ads_df)} ads from database")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading ads database: {str(e)}")
            return False
    
    def calculate_match_score(self, target: Dict[str, str], ad: pd.Series) -> int:
        """
        Calculate how well an ad matches the target values.
        
        Args:
            target: Dictionary with target values
            ad: Pandas Series representing an ad
            
        Returns:
            int: Match score (0-4, higher is better)
        """
        score = 0
        
        # Check each category
        if target.get('target_age_group') == ad['target_age_group']:
            score += 1
        if target.get('target_gender') == ad['target_gender']:
            score += 1
        if target.get('target_mood') == ad['target_mood']:
            score += 1
        if target.get('target_weather') == ad['target_weather']:
            score += 1
            
        return score
    
    def find_best_ad(self, target_values: Dict[str, str]) -> Optional[Dict]:
        """
        Find the best matching ad for given target values.
        
        Args:
            target_values: Dict with keys: target_age_group, target_gender, 
                          target_mood, target_weather
            
        Returns:
            Dict with ad information or None if no ads found
        """
        if not self.is_loaded:
            print("❌ Ads database not loaded. Call load_ads_database() first.")
            return None
        
        try:
            # Calculate match scores for all ads
            self.ads_df['match_score'] = self.ads_df.apply(
                lambda row: self.calculate_match_score(target_values, row),
                axis=1
            )
            
            # Find ads with highest score
            max_score = self.ads_df['match_score'].max()
            best_matches = self.ads_df[self.ads_df['match_score'] == max_score]
            
            # Select first best match (you could randomize or use other criteria)
            best_ad = best_matches.iloc[0]
            
            return {
                'pid': best_ad['pid'],
                'ad_title': best_ad['ad_title'],
                'target_age_group': best_ad['target_age_group'],
                'target_gender': best_ad['target_gender'],
                'target_mood': best_ad['target_mood'],
                'target_weather': best_ad['target_weather'],
                'match_score': int(best_ad['match_score']),
                'max_possible_score': 4
            }
            
        except Exception as e:
            print(f"❌ Error finding best ad: {str(e)}")
            return None
    
    def find_top_n_ads(self, target_values: Dict[str, str], n: int = 5) -> List[Dict]:
        """
        Find top N matching ads for given target values.
        
        Args:
            target_values: Dict with target categories
            n: Number of top ads to return
            
        Returns:
            List of ad dictionaries
        """
        if not self.is_loaded:
            print("❌ Ads database not loaded. Call load_ads_database() first.")
            return []
        
        try:
            # Calculate match scores
            self.ads_df['match_score'] = self.ads_df.apply(
                lambda row: self.calculate_match_score(target_values, row),
                axis=1
            )
            
            # Sort by score and get top N
            top_ads = self.ads_df.nlargest(n, 'match_score')
            
            results = []
            for _, ad in top_ads.iterrows():
                results.append({
                    'pid': ad['pid'],
                    'ad_title': ad['ad_title'],
                    'target_age_group': ad['target_age_group'],
                    'target_gender': ad['target_gender'],
                    'target_mood': ad['target_mood'],
                    'target_weather': ad['target_weather'],
                    'match_score': int(ad['match_score']),
                    'max_possible_score': 4
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Error finding top ads: {str(e)}")
            return []
    
    def get_database_stats(self) -> Optional[Dict]:
        """
        Get statistics about the ads database.
        
        Returns:
            Dict with database statistics or None if not loaded
        """
        if not self.is_loaded:
            return None
        
        return {
            'total_ads': len(self.ads_df),
            'unique_ads': self.ads_df['ad_title'].nunique(),
            'age_groups': self.ads_df['target_age_group'].value_counts().to_dict(),
            'genders': self.ads_df['target_gender'].value_counts().to_dict(),
            'moods': self.ads_df['target_mood'].value_counts().to_dict(),
            'weather': self.ads_df['target_weather'].value_counts().to_dict()
        }


# Example usage
if __name__ == "__main__":
    engine = AdRecommendationEngine()
    
    if engine.load_ads_database():
        # Test with sample target values
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
            print(f"\n✅ Best Match Found!")
            print(f"   Ad Title: {best_ad['ad_title']}")
            print(f"   Product ID: {best_ad['pid']}")
            print(f"   Match Score: {best_ad['match_score']}/{best_ad['max_possible_score']}")
            print(f"\n   Categories:")
            print(f"   - Age Group: {best_ad['target_age_group']}")
            print(f"   - Gender: {best_ad['target_gender']}")
            print(f"   - Mood: {best_ad['target_mood']}")
            print(f"   - Weather: {best_ad['target_weather']}")
        
        # Show database stats
        print("\n" + "="*60)
        print("Database Statistics")
        print("="*60)
        stats = engine.get_database_stats()
        if stats:
            print(f"\nTotal Ads: {stats['total_ads']}")
            print(f"Unique Ad Titles: {stats['unique_ads']}")
