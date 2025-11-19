import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import os

# --- Import dotenv ---
from dotenv import load_dotenv

# --- Required Libraries ---
try:
    from openai import OpenAI
    from sklearn.metrics.pairwise import cosine_similarity
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
except ImportError as e:
    OpenAI = lambda: None
    cosine_similarity = lambda x, y: np.array([[0.0]])
    rouge_scorer = lambda: None
    bert_score = lambda x, y, lang: (0, 0, 0)


load_dotenv()

EMBEDDING_MODEL = 'text-embedding-3-small'
ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']
CIT_PATTERN = re.compile(r'\[Page\s*\d+\]')


class AutomaticEvaluator:
    """
    Handles automatic factuality proxy (overlap) and citation coverage using OpenAI for embeddings.
    """
    def __init__(self):
        """Initializes the OpenAI client."""
        try:
            self.client = OpenAI()
        except Exception:
            self.client = None

    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """Helper function to call the OpenAI Embeddings API."""
        if not self.client:
            return np.array([[0.0]])

        try:
            # for batch embedding
            response = self.client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL
            )
            # Extract embeddings & convert to numpy array
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        except Exception as e:
            return np.array([[0.0]])

    def _get_claims(self, text: str) -> List[str]:
        """Splits the text into individual claims (sentences or bullets)."""
        claims = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s|\n', text)
        return [c.strip() for c in claims if c.strip()]

    def calculate_overlap(self, generated_script: str, source_sentences: List[str]) -> float:
        """
        Calculates the factuality proxy by semantic overlap using OpenAI embeddings.
        """
        if not self.client:
            return 0.0

        generated_claims = self._get_claims(generated_script)
        if not generated_claims or not source_sentences:
            return 0.0

        try:
            source_embeddings_np = self._get_openai_embeddings(source_sentences)
            claim_embeddings_np = self._get_openai_embeddings(generated_claims)

            # if embedding failed
            if source_embeddings_np.shape[0] < 1 or claim_embeddings_np.shape[0] < 1:
                 return 0.0

            # cosine similarity matrix 
            sim_matrix = cosine_similarity(claim_embeddings_np, source_embeddings_np)

            # Find the max overlap (max similarity across all sources)
            max_overlaps = sim_matrix.max(axis=1)

            # 4. Return average maximum overlap
            return np.mean(max_overlaps)
        except Exception as e:
            return 0.0

    def calculate_citation_coverage(self, generated_script: str) -> float:
        """
        Calculates citation coverage: what percentage of content has a retrieved provenance.
        Proxied by counting claims/segments that end with a citation tag [Page X].
        """
        generated_claims = self._get_claims(generated_script)
        if not generated_claims:
            return 0.0

        cited_claims_count = 0
        for claim in generated_claims:
            if CIT_PATTERN.search(claim.strip()):
                cited_claims_count += 1

        return cited_claims_count / len(generated_claims)

    def evaluate_automatic(self, generated_script: str, source_sentences: List[str]) -> Dict[str, float]:
        """Runs all automatic evaluations."""
        overlap = self.calculate_overlap(generated_script, source_sentences)
        coverage = self.calculate_citation_coverage(generated_script)

        return {
            "Semantic_Overlap_Factuality_Proxy_OpenAI": round(float(overlap), 4),
            "Citation_Coverage": round(coverage, 4)
        }


class QualityEvaluator:
    """
    Handles ROUGE and BERTScore comparison against a human-authored reference script.
    """
    def __init__(self):
        """Initializes the ROUGE scorer."""
        self.rouge_scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculates ROUGE scores."""
        scores = self.rouge_scorer.score(reference, generated)

        rouge_results = {}
        for r_type in ROUGE_TYPES:
            rouge_results[f"ROUGE_{r_type}_F1"] = scores[r_type].fmeasure

        return {k: round(v, 4) for k, v in rouge_results.items()}

    def calculate_bertscore(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculates BERTScore (Precision, Recall, F1)."""
        try:
            P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)

            return {
                "BERTScore_P": round(P.mean().item(), 4),
                "BERTScore_R": round(R.mean().item(), 4),
                "BERTScore_F1": round(F1.mean().item(), 4)
            }
        except Exception as e:
            print(f"Error during BERTScore calculation: {e}. Returning 0.0 for scores.")
            return {"BERTScore_P": 0.0, "BERTScore_R": 0.0, "BERTScore_F1": 0.0}

    def evaluate_quality(self, generated_script: str, reference_script: str) -> Dict[str, float]:
        """Runs all quality evaluations."""
        rouge_scores = self.calculate_rouge(generated_script, reference_script)
        
        bert_scores = self.calculate_bertscore(generated_script, reference_script)

        return {**rouge_scores, **bert_scores}


class HumanEvalSetup:
    """
    Prepares data for human evaluation based on the specified criteria.
    """
    def __init__(self, metrics_cols: List[str]):
        """Initializes the DataFrame columns."""
        self.base_cols = ["Paper_ID", "Prompt_Type", "Generated_Script", "Reference_Script", "Source_Context"]
        self.rater_cols = [
            "Rater_ID",
            "Audience_Appropriateness_Rating",
            "Factuality_Rating",
            "Helpfulness_Rating"
        ]
        self.df = pd.DataFrame(columns=self.base_cols + metrics_cols + self.rater_cols)

    def prepare_for_human_eval(self, data_list: List[Dict[str, Any]], output_file: str = "human_eval_data.csv"):
        """
        Creates and saves a structured CSV file for human raters.
        """
        records = []
        for i, data in enumerate(data_list):
            allowed_keys = self.base_cols + list(data.keys())
            record = {k: v for k, v in data.items() if k in allowed_keys}

            for rater_id in range(1, 4):
                rater_record = record.copy()
                rater_record["Rater_ID"] = f"Rater_{rater_id}"
                rater_record["Audience_Appropriateness_Rating"] = ""
                rater_record["Factuality_Rating"] = ""
                rater_record["Helpfulness_Rating"] = ""
                records.append(rater_record)

        self.df = pd.DataFrame(records)
        self.df.to_csv(output_file, index=False)
        print(f"--- Human Evaluation Data Setup Complete ---")
        print(f"File '{output_file}' created with {len(self.df)} rows for 3 raters.")


def main():
    # --- 1. (3 papers) ---
    # --- Paper 1: VDM (Diffusion Models) ---
    paper_1_context = [
        " In the default formulation of the Variational Autoencoder (VAE) [1], we directly maximize the ELBO. This approach is variational, because we optimize for the best q (zx) amongst a family of potential posterior distributions parameterized by . It is called an autoencoder because it is reminiscent of a traditional au toencoder model, where input data is trained to predict itself after undergoing an intermediate bottlenecking representation step.[Page 4]",
        " Eq (zx) log p(x/z)/q(z/x)=Eq (zx)[logp (xz)]-DKL(q (zx) p(z)).  the first term measures the reconstruction likelihood of the decoder from our variational distribution this ensures that the learned distribution is modeling effective latents that the original data can be regenerated from. The second term measures how similar the learned variational distribution is to a prior belief held over latent variables. Minimizing this term encourages the encoder to actually learn a distribution rather than collapse into a Dirac delta function.[Page 4]",
        " Maximizing the ELBO is thus equivalent to maximizing its first term and minimizing its second term.[Page 4]"
    ]
    paper_1_generated = (
        "The ELBO (Evidence Lower Bound Objective) aims to maximize the likelihood of the observed data under the variational approximation. It consists of two main terms[Paper 4]"
        "The first term, $$log p(x/z)/q(z/x)=Eq (zx)[logp (xz)]$$, measures the reconstruction likelihood of the decoder from the variational distribution, ensuring that the learned distribution accurately models the original data.[Paper 4]"
        "The second term, DKL(q (zx) p(z)), measures the similarity of the learned variational distribution to the prior belief held over latent variables. Minimizing this term encourages the encoder to learn a distribution rather than collapsing into a Dirac delta function.[Page 4]."
    )
    paper_1_reference = (
        " The VAE therefore utilizes the reparam-eterization trick and Monte Carlo estimates to optimize the Evidence Lower Bound (ELBO) jointly over φ and θ. After training a VAE, generating new data can be  [Page 4]. "
        " A deﬁning feature of the VAE is how the Evidence Lower Bound (ELBO) is optimized jointly over parameters φ and θ. The encoderof the VAE is commonly chosen to model a multivariate Gaussian with [Page 4]. "
        " The VAE therefore utilizes the reparam-eterization trick and Monte Carlo estimates to optimize the Evidence Lower Bound (ELBO) jointly over φ and θ. After training a VAE, generating new data can be [Page 4]"
    )

    # --- Paper 2: MFP3D (Computer Vision Pipeline) ---
    paper_2_context = [
    "Food portion estimation is crucial for monitoring health and tracking dietary intake. Image-based dietary assessment, which involves analyzing eating occasion images using computer vision techniques, is increasingly replacing traditional methods such as 24-hour recalls. However, accurately estimating the nutritional content from images remains challenging due to the loss of 3D information when projecting to the 2D image plane. Existing portion estimation methods are challenging to deploy in real-world scenarios due to their reliance on specific requirements, such as physical reference objects, high-quality depth information, or multi-view images and videos. In this paper, we introduce MFP3D, a new framework for accurate food portion estimation using only a single monocular image. Specifically, MFP3D consists of three key modules: (1) a 3D Reconstruction Module that generates a 3D point cloud representation of the food from the 2D image, (2) a Feature Extraction Module that extracts and concatenates features from both the 3D point cloud and the 2D RGB image, and (3) a Portion Regression Module that employs a deep regression model to estimate the food’s volume and energy content based on the extracted features. Our MFP3D is evaluated on MetaFood3D dataset, demonstrating its significant improvement in accurate portion estimation over existing methods. Keywords: Food Portion Estimation · 3D Point Cloud · Monocular Image · Multimodality Model [Abstract]."
]
    paper_2_generated = (
        "The MFP_3D dataset is a novel 3D food dataset designed to advance food computing and 3D vision research. This dataset consists of meticulously scanned and labeled 3D food objects across 131 categories, featuring detailed nutrition information, weight, and food codes linked to a comprehensive nutrition database. The dataset offers a robust benchmark for developing and evaluating 3D vision algorithms for food portion estimation, synthetic eating occasion simulation, and 3D food object generation. Experimental results demonstrate the strong capabilities of MFP_3D in enhancing food portion estimation algorithms and highlight the need for well-constructed food datasets in this field[page 1]. "
        "Large-scale 3D Food Dataset: MFP_3D contains 743 food objects with diverse textures, shapes, and sizes.[page 1] "
        "Multimodal Data: MFP_3D provides RGBD videos, segmentation masks, depth images, and other modalities to facilitate various research tasks.[Page 1]. "
        "Benchmarking and Evaluation: MFP_3D offers a robust benchmark for evaluating 3D vision algorithms for food portion estimation. [Table 1]."
    )
    paper_2_reference = (
        "Inthe Nutrition_5K dataset, food items are mixed together with-out segmentation masks, making it infeasible to performnutrition and geometric modeling for individual food items. The NutritionVerse_3D[page 2]. "
        "3D MeshRGBD VideoDepth MapFigure 1. MetaFood_3D is a real-scan 3D food dataset featuring diverse ready-to-eat 3D textured meshes, 720-degree RGBD videocaptures, and rich nutrition value annotations [page 1]. "
        "A comparison offood portion size estimation using geometric models anddepth images. Proceedings of the 2016 IEEE InternationalConference on Image Processing, pages 26–30, 2016. 3, 8[12] Anthony G. [page 8] "
    )

    # --- Paper 3: MetaFood3D (Dataset Features) ---
    paper_3_context = [
    "Food computing is both important and challenging in computer vision (CV). It significantly contributes to the development of CV algorithms due to its frequent presence in datasets across various applications, ranging from classification and instance segmentation to 3D reconstruction. The polymorphic shapes and textures of food, coupled with high variation in forms and vast multimodal information, including language descriptions and nutritional data, make food computing a complex and demanding task for modern CV algorithms. 3D food modeling is a new frontier for addressing food related problems, due to its inherent capability to deal with random camera views and its straightforward representation for calculating food portion size. However, the primary hurdle in the development of algorithms for food object analysis is the lack of nutrition values in existing 3D datasets. Moreover, in the broader field of 3D research, there is a critical need for domain-specific test datasets. To bridge the gap between general 3D vision and food computing research, we introduce MetaFood3D. This dataset consists of 743 meticulously scanned and labeled 3D food objects across 131 categories, featuring detailed nutrition information, weight, and food codes linked to a comprehensive nutrition database. Our MetaFood3D dataset emphasizes intra-class diversity and includes rich modalities such as textured mesh files, RGB-D videos, and segmentation masks. Experimental results demonstrate our dataset’s strong capabilities in enhancing food portion estimation algorithms, highlight the gap between video captures and 3D scanned data, and showcase the strengths of MetaFood3D in generating synthetic eating occasion data and 3D food objects [Page 1].",
    "The dataset features detailed nutrition information, weight, and food codes [Page 1].",
    "It includes rich modalities such as textured mesh files, RGB-D videos, and segmentation masks [Page 1].",
    "We emphasize intra-class diversity by collecting foods with varying appearances [Page 1]."
]
    paper_3_generated = (
        "The MetaFood_3D dataset is a novel 3D food dataset designed to advance food computing and 3D vision research. This dataset includes 743 meticulously scanned and labeled 3D food objects across 131 categories, featuring detailed nutrition information, weight, and food codes linked to a comprehensive nutrition database. The dataset emphasizes intra-class diversity and includes rich modalities such as textured mesh files, RGB-D videos, and segmentation masks. Experimental results demonstrate the strong capabilities of the dataset in enhancing food portion estimation algorithms and generating synthetic eating occasion data.[Page 1]. "
        "Large-scale 3D Food Dataset: MetaFood_3D is the largest and most diverse 3D food dataset with nutrition values, providing a robust benchmark for developing and evaluating 3D vision algorithms. "
        "Detailed Nutrition Annotations: The dataset includes detailed nutrition annotations for each food object, enabling accurate energy calculations and dietary assessment."
        "Rich Modalities: MetaFood_3D includes various modalities such as textured mesh files, RGB-D videos, and segmentation masks, providing a rich foundation for food-related tasks"
    )
    paper_3_reference = (
        "MetaFood_3D 3D Food Dataset with Nutrition ValuesYuhao Chen_2Jiangpeng He_1†Gautham Vinod_1Siddeshwar Raghavan_1Chris Czarnecki_2*Jinge Ma_1Talha Ibn Mahmud_1Bruce Coburn_1Dayou Mao_2Saeejith [page 1] "
        "3D MeshRGBD VideoDepth MapFigure 1. MetaFood_3D is a real-scan 3D food dataset featuring diverse ready-to-eat 3D textured meshes, 720-degree RGBD videocaptures, and rich nutrition value [page 1]. "
        "Furthermore,there is a shortage of benchmark 3D food datasets featur-ing diverse intra-class variation. For instance, the OmniOb-ject_3D dataset [88] includes 2,837 food objects, but the selec-tion [page 1] "
    )

    # --- 2. Initialize and Collect All Evaluation Results ---
    auto_evaluator = AutomaticEvaluator()
    quality_evaluator = QualityEvaluator()

    #  to run all evaluations for a paper
    def run_evaluations(paper_id: str, generated: str, context: List[str], reference: str) -> Dict[str, Any]:
        print(f"\n--- Running Evaluation for {paper_id} ---")
        auto_metrics = auto_evaluator.evaluate_automatic(generated, context)
        quality_metrics = quality_evaluator.evaluate_quality(generated, reference)
        
        all_metrics = {**auto_metrics, **quality_metrics}
        
        print("Automatic Metrics:", {k: v for k, v in auto_metrics.items()})
        print("Quality Metrics:", quality_metrics)
        
        return {
            "Paper_ID": paper_id,
            "Generated_Script": generated,
            "Reference_Script": reference,
            "Source_Context": "\n".join(context),
            **all_metrics
        }

    # Run for all three papers
    data_P1 = run_evaluations("P1_VDM", paper_1_generated, paper_1_context, paper_1_reference)
    data_P2 = run_evaluations("P2_MFP3D", paper_2_generated, paper_2_context, paper_2_reference)
    data_P3 = run_evaluations("P3_MetaFood3D", paper_3_generated, paper_3_context, paper_3_reference)

    # (Hypothetical for human eval preparation)
    data_P1["Prompt_Type"] = "Graduate Student"
    data_P2["Prompt_Type"] = "Press Release"
    data_P3["Prompt_Type"] = "Technical Slide"

    # --- PREPARE FINAL HUMAN EVAL DATASET ---
    all_metrics_data = [data_P1, data_P2, data_P3]
    
    human_eval_cols = list(data_P1.keys() - {"Paper_ID", "Prompt_Type", "Generated_Script", "Reference_Script", "Source_Context"})
    
    human_eval_setup = HumanEvalSetup(human_eval_cols)
    human_eval_setup.prepare_for_human_eval(all_metrics_data, output_file="/media/nas_mount/research3/aman_kr/SARAL_TASK/SARAL_THEME3_FINAL/evaluation/reports/saral_evaluation_data.csv")


if __name__ == "__main__":
    main()