"""
Dataset Extension Script: Combining Bad Jailbreak Prompts with Good Prompts
===========================================================================

This script:
1. Loads the bad jailbreak prompts dataset (28K samples, label=1)
2. Loads the good prompts dataset (20K samples)
3. Engineers features for good prompts to match the bad prompts structure
4. Assigns label=0 to good prompts (non-jailbreak)
5. Combines both datasets into an extended dataset
6. Saves the extended dataset with ~48K total samples

The extended dataset will have balanced classes for better model training.
"""

import pandas as pd
import numpy as np
import os
import re
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoodPromptsFeatureEngineering:
    """
    Feature engineering for good (non-jailbreak) prompts to match
    the structure of bad jailbreak prompts dataset.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline"""
        self.feature_names = []
        
    def extract_text_features(self, df, text_column='prompt'):
        """
        Extract text complexity features from prompts
        
        Args:
            df: DataFrame with prompt text
            text_column: Name of the column containing prompt text
            
        Returns:
            DataFrame with added text features
        """
        logger.info("Extracting text complexity features for good prompts...")
        
        # Query length (character count)
        df['query_length'] = df[text_column].fillna('').apply(len)
        
        # Token count (word count)
        df['token_count'] = df[text_column].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        # Question count
        df['question_count'] = df[text_column].fillna('').apply(
            lambda x: str(x).count('?')
        )
        
        # Meaningful word count (approximate)
        df['meaningful_word_count'] = df['token_count']  # Same as token count for good prompts
        
        logger.info("Text features extracted: query_length, token_count, question_count, meaningful_word_count")
        return df
    
    def extract_heuristic_features(self, df, text_column='prompt'):
        """
        Extract prompt-level heuristic features
        
        Args:
            df: DataFrame with prompt text
            text_column: Name of the column containing prompt text
            
        Returns:
            DataFrame with added heuristic features
        """
        logger.info("Extracting prompt-level heuristics for good prompts...")
        
        # Code block detection - good prompts may have code-related requests
        df['has_code_blocks'] = df[text_column].fillna('').apply(
            lambda x: 1 if ('```' in str(x) or 'code' in str(x).lower() or 'json' in str(x).lower() or 'csv' in str(x).lower()) else 0
        )
        
        # Roleplay markers - good prompts often have "act as" for legitimate purposes
        roleplay_patterns = r'(act as|identify|describe|create|write|analyze)'
        df['has_roleplay_markers'] = df[text_column].fillna('').apply(
            lambda x: 1 if re.search(roleplay_patterns, str(x).lower()) else 0
        )
        
        # Instruction markers - good prompts DON'T have jailbreak instructions
        instruction_patterns = r'(ignore|override|bypass|disregard|forget|do not follow)'
        df['has_instruction_markers'] = df[text_column].fillna('').apply(
            lambda x: 1 if re.search(instruction_patterns, str(x).lower()) else 0
        )
        
        # Prompt chaining - good prompts may have steps
        chaining_patterns = r'(\d+\.|step \d+|first.*then|after that|next,)'
        df['has_prompt_chaining'] = df[text_column].fillna('').apply(
            lambda x: 1 if re.search(chaining_patterns, str(x).lower()) else 0
        )
        
        logger.info("Heuristic features extracted: 4 binary indicators")
        return df
    
    def create_toxicity_features(self, df):
        """
        Create toxicity features (all zeros for good prompts)
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with toxicity features
        """
        logger.info("Creating toxicity features (zeros for good prompts)...")
        
        # All toxicity metrics should be 0 for good prompts
        df['toxicity'] = 0.0
        df['severe_toxicity'] = 0.0
        df['obscene'] = 0.0
        df['threat'] = 0.0
        df['insult'] = 0.0
        df['identity_attack'] = 0.0
        df['harm_score'] = 0.0
        
        logger.info("Toxicity features created (all zeros)")
        return df
    
    def create_policy_encodings(self, df):
        """
        Create policy category encodings for good prompts
        Map categories to appropriate policies or create 'Benign' policy
        
        Args:
            df: DataFrame with 'category' column
            
        Returns:
            DataFrame with policy column
        """
        logger.info("Creating policy mappings for good prompts...")
        
        # Map good prompt categories to benign policy categories
        # These are legitimate use cases, not policy violations
        category_to_policy = {
            'Culinary/Food': 'Benign_Creative',
            'Coding': 'Benign_Technical',
            'Creative Writing': 'Benign_Creative',
            'Nature/Biology': 'Benign_Educational',
            'Education': 'Benign_Educational',
            'Fashion/Style': 'Benign_Creative',
            'Home/DIY': 'Benign_Technical',
            'Travel/Landmarks': 'Benign_Creative',
            'OCR/Extraction': 'Benign_Technical',
            'Data Analysis': 'Benign_Technical'
        }
        
        df['policy'] = df['category'].map(category_to_policy).fillna('Benign_General')
        
        # Get unique policies from good prompts
        unique_policies = df['policy'].unique()
        logger.info(f"Found {len(unique_policies)} unique policy categories for good prompts: {unique_policies}")
        
        return df
    
    def create_metadata_features(self, df, text_column='prompt'):
        """
        Create metadata features to match bad prompts structure
        
        Args:
            df: DataFrame
            text_column: Name of the column containing prompt text
            
        Returns:
            DataFrame with metadata features
        """
        logger.info("Creating metadata features for good prompts...")
        
        # Map prompt to jailbreak_query column
        df['jailbreak_query'] = df[text_column]
        
        # Create redteam_query (same as prompt for good prompts)
        df['redteam_query'] = df[text_column]
        
        # Format is 'Benign' for good prompts
        df['format'] = 'Benign'
        
        # Image path mapping (using image_context as description)
        df['image_path'] = 'good_prompts/image_' + df['id'].astype(str) + '.png'
        
        # Source
        df['from'] = 'Good_Prompts_Dataset'
        
        # Binary flags
        df['selected_mini'] = False
        df['transfer_from_llm'] = False
        
        # Image descriptions
        df['image_description_x'] = df['image_context']
        df['image_description_y'] = df['image_context']
        df['visible_text_x'] = ''
        df['visible_text_y'] = '[]'
        df['img_path'] = df['image_path']
        
        logger.info("Metadata features created")
        return df
    
    def create_image_features(self, df):
        """
        Create placeholder image features for good prompts
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with image features
        """
        logger.info("Creating image features for good prompts...")
        
        # Use image_context length as proxy for image complexity
        df['image_desc_length'] = df['image_context'].fillna('').apply(len)
        
        # Good prompts reference images, so has_visible_text = 0 (no text manipulation)
        df['has_visible_text'] = 0
        
        logger.info("Image features created")
        return df
    
    def engineer_all_features(self, df, text_column='prompt'):
        """
        Main feature engineering pipeline for good prompts
        
        Args:
            df: Raw good prompts dataframe
            text_column: Name of the column containing prompt text
            
        Returns:
            DataFrame with engineered features matching bad prompts structure
        """
        logger.info("Starting feature engineering pipeline for good prompts...")
        
        # Extract all feature types
        df = self.extract_text_features(df, text_column)
        df = self.extract_heuristic_features(df, text_column)
        df = self.create_toxicity_features(df)
        df = self.create_policy_encodings(df)
        df = self.create_metadata_features(df, text_column)
        df = self.create_image_features(df)
        
        logger.info("Feature engineering complete for good prompts")
        return df


def load_bad_prompts(path='dataset/df_28K_completed_features_1116.csv'):
    """
    Load bad jailbreak prompts dataset
    
    Args:
        path: Path to bad prompts CSV
        
    Returns:
        DataFrame with bad prompts
    """
    logger.info(f"Loading bad jailbreak prompts from {path}...")
    df_bad = pd.read_csv(path)
    
    # Add label
    df_bad['jailbroken'] = 1  # Bad prompts are jailbreak attempts
    df_bad['label'] = 'bad_jailbreak'
    
    logger.info(f"Loaded {len(df_bad)} bad jailbreak prompts")
    logger.info(f"Columns: {df_bad.columns.tolist()}")
    
    return df_bad


def load_good_prompts(path='dataset/multimodal_prompts_20k.csv'):
    """
    Load good prompts dataset
    
    Args:
        path: Path to good prompts CSV
        
    Returns:
        DataFrame with good prompts
    """
    logger.info(f"Loading good prompts from {path}...")
    df_good = pd.read_csv(path)
    
    # Add label
    df_good['jailbroken'] = 0  # Good prompts are NOT jailbreak attempts
    df_good['label'] = 'good_prompt'
    
    logger.info(f"Loaded {len(df_good)} good prompts")
    logger.info(f"Columns: {df_good.columns.tolist()}")
    
    return df_good


def align_columns(df_bad, df_good):
    """
    Align columns between bad and good prompts datasets
    Ensure both have the same columns in the same order
    
    Args:
        df_bad: Bad prompts dataframe
        df_good: Good prompts dataframe with engineered features
        
    Returns:
        Tuple of (df_bad, df_good) with aligned columns
    """
    logger.info("Aligning columns between datasets...")
    
    # Get column sets
    bad_cols = set(df_bad.columns)
    good_cols = set(df_good.columns)
    
    # Find differences
    only_in_bad = bad_cols - good_cols
    only_in_good = good_cols - bad_cols
    
    logger.info(f"Columns only in bad prompts: {only_in_bad}")
    logger.info(f"Columns only in good prompts: {only_in_good}")
    
    # Add missing columns to good prompts with default values
    for col in only_in_bad:
        if col not in df_good.columns:
            # Determine appropriate default value
            if df_bad[col].dtype in [np.float64, np.int64]:
                df_good[col] = 0
            else:
                df_good[col] = ''
            logger.info(f"Added column '{col}' to good prompts with default value")
    
    # Add missing columns to bad prompts with default values
    for col in only_in_good:
        if col not in df_bad.columns:
            # Determine appropriate default value
            if df_good[col].dtype in [np.float64, np.int64]:
                df_bad[col] = 0
            else:
                df_bad[col] = ''
            logger.info(f"Added column '{col}' to bad prompts with default value")
    
    # Ensure same column order
    common_cols = sorted(list(bad_cols.union(good_cols)))
    df_bad = df_bad[common_cols]
    df_good = df_good[common_cols]
    
    logger.info(f"Columns aligned: {len(common_cols)} total columns")
    
    return df_bad, df_good


def combine_datasets(df_bad, df_good):
    """
    Combine bad and good prompts into extended dataset
    
    Args:
        df_bad: Bad prompts dataframe
        df_good: Good prompts dataframe
        
    Returns:
        Combined dataframe
    """
    logger.info("Combining bad and good prompts...")
    
    # Concatenate
    df_extended = pd.concat([df_bad, df_good], ignore_index=True)
    
    # Reset index
    df_extended['id'] = range(len(df_extended))
    
    # Shuffle the dataset
    df_extended = df_extended.sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"Combined dataset size: {len(df_extended)} samples")
    logger.info(f"Bad prompts (label=1): {(df_extended['jailbroken'] == 1).sum()} samples")
    logger.info(f"Good prompts (label=0): {(df_extended['jailbroken'] == 0).sum()} samples")
    logger.info(f"Class balance: {df_extended['jailbroken'].value_counts(normalize=True).to_dict()}")
    
    return df_extended


def save_extended_dataset(df_extended, output_path='dataset/extended_dataset_with_good_prompts.csv'):
    """
    Save the extended dataset
    
    Args:
        df_extended: Combined dataset
        output_path: Output file path
    """
    logger.info(f"Saving extended dataset to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df_extended.to_csv(output_path, index=False)
    
    logger.info(f"Extended dataset saved successfully!")
    logger.info(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


def generate_summary_report(df_extended, output_path='dataset/extension_summary_report.txt'):
    """
    Generate a summary report of the dataset extension
    
    Args:
        df_extended: Extended dataset
        output_path: Output file path for report
    """
    logger.info("Generating summary report...")
    
    report = []
    report.append("="*80)
    report.append("DATASET EXTENSION SUMMARY REPORT")
    report.append("="*80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Dataset sizes
    report.append("DATASET SIZES")
    report.append("-"*80)
    report.append(f"Total samples: {len(df_extended)}")
    report.append(f"Bad jailbreak prompts (label=1): {(df_extended['jailbroken'] == 1).sum()}")
    report.append(f"Good prompts (label=0): {(df_extended['jailbroken'] == 0).sum()}")
    report.append("")
    
    # Class balance
    report.append("CLASS DISTRIBUTION")
    report.append("-"*80)
    class_dist = df_extended['jailbroken'].value_counts(normalize=True)
    report.append(f"Jailbreak prompts (1): {class_dist.get(1, 0)*100:.2f}%")
    report.append(f"Good prompts (0): {class_dist.get(0, 0)*100:.2f}%")
    report.append("")
    
    # Policy distribution
    report.append("POLICY DISTRIBUTION")
    report.append("-"*80)
    policy_dist = df_extended['policy'].value_counts()
    report.append(f"Total unique policies: {len(policy_dist)}")
    report.append("\nTop 10 policies:")
    for policy, count in policy_dist.head(10).items():
        report.append(f"  {policy}: {count} ({count/len(df_extended)*100:.2f}%)")
    report.append("")
    
    # Feature statistics
    report.append("FEATURE STATISTICS")
    report.append("-"*80)
    report.append(f"Total features: {len(df_extended.columns)}")
    
    # Text features
    if 'query_length' in df_extended.columns:
        report.append(f"\nQuery length (mean): {df_extended['query_length'].mean():.2f}")
        report.append(f"Query length by class:")
        report.append(f"  Bad prompts: {df_extended[df_extended['jailbroken']==1]['query_length'].mean():.2f}")
        report.append(f"  Good prompts: {df_extended[df_extended['jailbroken']==0]['query_length'].mean():.2f}")
    
    if 'token_count' in df_extended.columns:
        report.append(f"\nToken count (mean): {df_extended['token_count'].mean():.2f}")
        report.append(f"Token count by class:")
        report.append(f"  Bad prompts: {df_extended[df_extended['jailbroken']==1]['token_count'].mean():.2f}")
        report.append(f"  Good prompts: {df_extended[df_extended['jailbroken']==0]['token_count'].mean():.2f}")
    
    # Heuristic features
    report.append("\nHEURISTIC FEATURES (% of samples with feature)")
    report.append("-"*80)
    heuristic_features = ['has_code_blocks', 'has_roleplay_markers', 'has_instruction_markers', 'has_prompt_chaining']
    for feat in heuristic_features:
        if feat in df_extended.columns:
            overall_pct = df_extended[feat].mean() * 100
            bad_pct = df_extended[df_extended['jailbroken']==1][feat].mean() * 100
            good_pct = df_extended[df_extended['jailbroken']==0][feat].mean() * 100
            report.append(f"{feat}:")
            report.append(f"  Overall: {overall_pct:.2f}%")
            report.append(f"  Bad prompts: {bad_pct:.2f}%")
            report.append(f"  Good prompts: {good_pct:.2f}%")
    
    # Toxicity features
    report.append("\nTOXICITY FEATURES (mean scores)")
    report.append("-"*80)
    toxicity_features = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'harm_score']
    for feat in toxicity_features:
        if feat in df_extended.columns:
            overall_mean = df_extended[feat].mean()
            bad_mean = df_extended[df_extended['jailbroken']==1][feat].mean()
            good_mean = df_extended[df_extended['jailbroken']==0][feat].mean()
            report.append(f"{feat}:")
            report.append(f"  Overall: {overall_mean:.6f}")
            report.append(f"  Bad prompts: {bad_mean:.6f}")
            report.append(f"  Good prompts: {good_mean:.6f}")
    
    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Summary report saved to {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def main():
    """
    Main execution function for dataset extension
    """
    logger.info("="*80)
    logger.info("DATASET EXTENSION: COMBINING BAD AND GOOD PROMPTS")
    logger.info("="*80)
    
    # Step 1: Load bad prompts
    df_bad = load_bad_prompts()
    
    # Step 2: Load good prompts
    df_good = load_good_prompts()
    
    # Step 3: Engineer features for good prompts
    feature_engineer = GoodPromptsFeatureEngineering()
    df_good = feature_engineer.engineer_all_features(df_good, text_column='prompt')
    
    # Step 4: Align columns
    df_bad, df_good = align_columns(df_bad, df_good)
    
    # Step 5: Combine datasets
    df_extended = combine_datasets(df_bad, df_good)
    
    # Step 6: Save extended dataset
    save_extended_dataset(df_extended)
    
    # Step 7: Generate summary report
    generate_summary_report(df_extended)
    
    logger.info("="*80)
    logger.info("DATASET EXTENSION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Extended dataset saved to: dataset/extended_dataset_with_good_prompts.csv")
    logger.info(f"Summary report saved to: dataset/extension_summary_report.txt")
    logger.info(f"Total samples: {len(df_extended)}")
    logger.info(f"Ready for model training!")


if __name__ == '__main__':
    main()