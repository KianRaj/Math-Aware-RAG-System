import pandas as pd
import numpy as np

def aggregate_saral_results(input_file="/media/nas_mount/research3/aman_kr/SARAL_TASK/SARAL_THEME3_FINAL/evaluation/reports/saral_evaluation_data.csv"):
    df = pd.read_csv(input_file)
    rating_cols = [
        "Audience_Appropriateness_Rating", 
        "Factuality_Rating", 
        "Helpfulness_Rating"
        ]
        
    for col in rating_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

        #Group by Paper_ID to see averages per paper
        #aggregating both the automatic metrics and human ratings
        numeric_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64']]
        
        summary_df = df.groupby("Paper_ID")[numeric_cols].mean().reset_index()
        
        # Calculate an "Overall Quality Score" (Weighted Average Example)
    summary_df["FINAL_COMPOSITE_SCORE"] = (
        (summary_df["Semantic_Overlap_Factuality_Proxy_OpenAI"] * 100 * 0.4) + 
        (summary_df["BERTScore_F1"] * 100 * 0.3) + 
        (summary_df["Helpfulness_Rating"] / 5 * 100 * 0.3)
        ).round(2)

        # Displaying Report
    print("\n" + "="*80)
    print("SARAL EVALUATION REPORT: AGGREGATED RESULTS")
    print("="*80)
        
        # round to 2 decimal
    display_cols = ["Paper_ID", "Citation_Coverage", "ROUGE_rougeL_F1", "BERTScore_F1", 
                    "Audience_Appropriateness_Rating", "Factuality_Rating", 
                    "Helpfulness_Rating", "FINAL_COMPOSITE_SCORE"]
        
    valid_cols = [c for c in display_cols if c in summary_df.columns]
        
    print(summary_df[valid_cols].to_string(index=False))
    print("-" * 80)
        
    system_score = summary_df["FINAL_COMPOSITE_SCORE"].mean()
    print(f"\nSYSTEM-WIDE AVERAGE SCORE: {system_score:.2f} / 100")
        
    summary_df.to_csv("saral_final_report.csv", index=False)


if __name__ == "__main__":
    aggregate_saral_results()