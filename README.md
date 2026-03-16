# AI_ML_Learning_mentor

AI-powered assistant that helps users discover datasets, learn machine learning concepts, and build AI/ML projects step-by-step.

What a smart AI ML Project Mentor should do:
search → understand intent → retrieve candidates → evaluate suitability → explain recommendation → support follow-up Q&A → expand chosen dataset with analytics

Phase 1: Understand the user’s real intent
When user types something like:
- xgboost
- image classification
- beginner NLP project
- time series forecasting using sales data
the agent should first convert that into a structured intent.
Build an intent object


Example: search -> 'xgboost'
The word xgboost alone does not mean a dataset topic.
It may mean:
- a dataset suitable for learning XGBoost
- a dataset created using XGBoost experiments
- a dataset whose description mentions xgboost
- a project idea where XGBoost is a good model choice

So the agent must interpret the query semantically, not literally.

Phase 2: Search all candidate datasets from sources
Separate retrieval from recommendation.

Phase 3: Normalize and enrich metadata
Different sources return different fields. Normalize them into a common schema

Phase 4: Rank datasets for suitability, not keyword only -> this defines the metrics(feature engineering)
For each retrieved dataset, compute a recommendation score based on multiple dimensions. Use weighted ranking.

Phase 5: Show multiple candidates, then recommend one
Example response:
We found 4 matching datasets.
- Cinema-Tickets
  Best beginner-friendly choice for learning XGBoost on structured tabular data.

- California-Environmental-Conditions-Dataset
  Good for regression-style experimentation and feature importance analysis.

- Gender-Recognition-by-Voice
  Interesting but slightly less direct for raw XGBoost learning unless features are already extracted.

- mlr_xgboost_rng
  Matches the keyword exactly, but it is more of an XGBoost-related benchmark dataset than the best beginner project dataset.

Phase 6: Convince the user why the recommendation is correct
This is where your AI mentor becomes genuinely useful.
For each recommendation, provide:
1. Why it matches the topic
2. Why it is suitable for their skill level
3. Why it is better than the other candidates
4. What they will learn from it

Example explanation:
I recommend Cinema-Tickets over mlr_xgboost_rng for a beginner learning XGBoost because it is easier to frame as a real supervised tabular ML problem. XGBoost shines most on structured tabular data, and this dataset is likely easier to explain, preprocess, and model than a benchmark-style dataset whose main relevance is the keyword itself. It gives you a more practical learning experience: feature engineering, prediction target selection, model training, evaluation, and interpretation.

Phase 7: Allow follow-up questioning over the selected dataset
Once user selects a dataset, shift into a dataset mentor mode.
The agent should answer questions like:
- What is this dataset about?
- What is the target column?
- Is this classification or regression?
- Is XGBoost the best model here?
- What preprocessing is needed?
- What metrics should I use?
- What project can I build with this?
- Is this dataset too hard for me?

Store a selected dataset state:
{
  "selected_dataset_id": "openml_43388",
  "selected_dataset_name": "Cinema-Tickets",
  "selected_source": "OpenML"
}
Then all future questions use this context.

Phase 8: Expand the selected dataset automatically
When the user selects a dataset, the agent should fetch and analyze it.

- Expansion pipeline
- Download metadata
- Load sample rows
- Detect target candidates
- Compute summary analytics
- Generate beginner explanation
- Suggest project ideas
- Suggest baseline ML workflow

Phase 9: General analytics the mentor should showcase
When dataset is selected, run automatic EDA.

Example narrative:
This dataset has 143k rows and 14 columns, which is large enough to learn from but still manageable. Most columns are structured/tabular, making it a strong fit for XGBoost. Missing values are low, which is good for a beginner workflow. The target appears moderately imbalanced, so evaluation using F1 or ROC-AUC may be better than plain accuracy.