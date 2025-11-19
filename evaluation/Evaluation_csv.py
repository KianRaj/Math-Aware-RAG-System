import pandas as pd

data = [
    # --- P1: VDM ---
    {
        "Paper_ID": "P1_VDM",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7304,
        "Citation_Coverage": 0.3333,
        "ROUGE_rougeL_F1": 0.186,
        "BERTScore_F1": 0.8091,
        "Rater_ID": "Rater_1", "Audience_Appropriateness_Rating": 5, "Factuality_Rating": 5, "Helpfulness_Rating": 4
    },
    {
        "Paper_ID": "P1_VDM",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7304,
        "Citation_Coverage": 0.3333,
        "ROUGE_rougeL_F1": 0.186,
        "BERTScore_F1": 0.8091,
        "Rater_ID": "Rater_2", "Audience_Appropriateness_Rating": 4, "Factuality_Rating": 5, "Helpfulness_Rating": 5
    },
    {
        "Paper_ID": "P1_VDM",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7304,
        "Citation_Coverage": 0.3333,
        "ROUGE_rougeL_F1": 0.186,
        "BERTScore_F1": 0.8091,
        "Rater_ID": "Rater_3", "Audience_Appropriateness_Rating": 5, "Factuality_Rating": 4, "Helpfulness_Rating": 4
    },

    # --- P2: MFP3D ---
    {
        "Paper_ID": "P2_MFP3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.5961,
        "Citation_Coverage": 0.1429,
        "ROUGE_rougeL_F1": 0.1719,
        "BERTScore_F1": 0.8482,
        "Rater_ID": "Rater_1", "Audience_Appropriateness_Rating": 3, "Factuality_Rating": 4, "Helpfulness_Rating": 3
    },
    {
        "Paper_ID": "P2_MFP3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.5961,
        "Citation_Coverage": 0.1429,
        "ROUGE_rougeL_F1": 0.1719,
        "BERTScore_F1": 0.8482,
        "Rater_ID": "Rater_2", "Audience_Appropriateness_Rating": 3, "Factuality_Rating": 3, "Helpfulness_Rating": 4
    },
    {
        "Paper_ID": "P2_MFP3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.5961,
        "Citation_Coverage": 0.1429,
        "ROUGE_rougeL_F1": 0.1719,
        "BERTScore_F1": 0.8482,
        "Rater_ID": "Rater_3", "Audience_Appropriateness_Rating": 4, "Factuality_Rating": 4, "Helpfulness_Rating": 3
    },

    # --- P3: MetaFood3D ---
    {
        "Paper_ID": "P3_MetaFood3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7618,
        "Citation_Coverage": 0.1667,
        "ROUGE_rougeL_F1": 0.1962,
        "BERTScore_F1": 0.8299,
        "Rater_ID": "Rater_1", "Audience_Appropriateness_Rating": 5, "Factuality_Rating": 5, "Helpfulness_Rating": 5
    },
    {
        "Paper_ID": "P3_MetaFood3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7618,
        "Citation_Coverage": 0.1667,
        "ROUGE_rougeL_F1": 0.1962,
        "BERTScore_F1": 0.8299,
        "Rater_ID": "Rater_2", "Audience_Appropriateness_Rating": 5, "Factuality_Rating": 5, "Helpfulness_Rating": 5
    },
    {
        "Paper_ID": "P3_MetaFood3D",
        "Semantic_Overlap_Factuality_Proxy_OpenAI": 0.7618,
        "Citation_Coverage": 0.1667,
        "ROUGE_rougeL_F1": 0.1962,
        "BERTScore_F1": 0.8299,
        "Rater_ID": "Rater_3", "Audience_Appropriateness_Rating": 4, "Factuality_Rating": 5, "Helpfulness_Rating": 4
    },
]

df = pd.DataFrame(data)

output_path = "/media/nas_mount/research3/aman_kr/SARAL_TASK/SARAL_THEME3_FINAL/evaluation/reports/saral_evaluation_data.csv"
df.to_csv(output_path, index=False)

print(f"Successfully recreated clean CSV at: {output_path}")