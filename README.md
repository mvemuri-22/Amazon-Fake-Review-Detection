# Amazon Fake Review Detection
This project focuses on detecting fake reviews on platforms like Amazon, which can mislead consumers, distort product rankings, and erode trust in online shopping. Manual moderation is impractical due to the millions of reviews submitted daily. Traditional rule-based methods often lack adaptability across various product types and writing styles.



## Team
Ben Brown 

Kurt Fischer 

Alex Lee 

Manas Vemuri 

## Problem Statement
The proliferation of fake reviews poses a significant challenge to the integrity of online shopping platforms. Our goal is to develop a robust system to identify these deceptive reviews.

## Data Description
This project utilizes the Fake Reviews Dataset from Kaggle, specifically curated for binary classification of Amazon-style product reviews.

Total Samples: 40,526 reviews 
20,294 real reviews (label = 0) 
20,232 fake reviews (label = 1), including 94 synthetically added samples 
Each review record contains:
Category (e.g., Home_and_Kitchen) 
Rating (1.0 to 5.0 stars) 
Text (review content) 
Label (0 = real, 1 = fake) 
Assumptions and Hypotheses
Linguistic Distinctions: We hypothesize that fake reviews will exhibit different linguistic patterns compared to real human reviews. 
Certain words or phrases may be more frequent in fake reviews.
Fake reviews may overuse formal language, repeat phrases unnaturally, and be less likely to have spelling errors.
Review Length Variations: We assumed observable differences in the typical length of fake reviews compared to real ones. However, our EDA showed fake reviews tend to be longer than real reviews.

Lack of Specificity: We hypothesize that fake reviews will generally lack personalized details, specific product usage examples, or unique insights found in real customer reviews.
## Exploratory Data Analysis (EDA) - Data Findings
The dataset has a roughly equal distribution of real (20,294) and fake (20,232) reviews.
There are no noticeable trends in the distribution of fake vs. real reviews across different star ratings.
Contrary to initial hypothesis, fake reviews actually trend to be longer than real human reviews.
The average review length is approximately 32 tokens.
31,903 out of 40,526 reviews are shorter than 100 tokens, leading to the decision to limit review length to 100 tokens to avoid excessive zero-padding for shorter reviews.
Feature Engineering and Preprocessing
Stop Word Removal: Common words (e.g., "the", "at", "and") were removed to focus on more meaningful terms.
Stemming: Words were reduced to their root form (e.g., "ran", "runs", "running" to "run") to reduce vocabulary and complexity.
Tokenization: Words were tokenized into smaller units and mapped to unique integers.
Padding: Input sequences were padded with zeros to ensure uniform length for neural network input.
## Models and Results
| Model Type           | Best Accuracy |
|-----------------------|---------------|
| XGBoost              | 85%           |
| LSTM                 | 88%           |
| GRU                  | 91.3%         |
| 1-D CNN              | 90.5%         |
| Transformer Encoder  | 91.3%         |
| RoBERTa              | 95%           |

### Baseline Model: XGBoost with Word Vectorizer
Methodology: Used TF-IDF word vectorizer to create a sparse matrix of words, with weights for frequency. Random search with cross-validation was used to tune hyperparameters and fit the XGBoost model.
Results: 85% accuracy, 89% recall of fake reviews.
Long Short Term Memory (LSTM)
Motivation: LSTMs can capture word order and context, and may learn style and tone differences.
Model Details: Sequential LSTM with an embedding layer, two dropout layers for regularization, and two dense layers for binary classification. Early stopping was implemented to prevent overfitting.
Results: 88% accuracy.
### Gated Recurrent Unit (GRU)
Motivation: GRUs are effective for sequential text data, capturing long-term contexts efficiently.
Model Architectures & Findings:
Bidirectional GRU: Enhanced context understanding by processing text in both directions, leading to notable accuracy gains.
Dense Layers: Additional dense layers improved test accuracy.
Hyperparameter Tuning: Tuned parameters such as number of GRU units, embedding dimensions, dropout, optimizers, and batch sizes to achieve optimal test accuracy.
Early Stopping: Used early stopping on the validation set to prevent overfitting, with most models converging around 5-6 epochs.
Results: Achieved up to 91.34% validation accuracy with a bidirectional GRU, 128 GRU units, one ReLU dense layer, and RMSprop optimizer.
### CNN for Text
Motivation: Convolutional layers excel at capturing local patterns regardless of positioning. A 1-D CNN was used for sequence input.

Model Experimentation: Tuned number of filters and kernel size. Experimented with pooling types (global max pooling vs. max pooling first) and regularization (20% dropout or no dropout). All models used the same embedding, Dense, and output layers, with Adam optimizer and early stopping.
Takeaway: The best CNN model used 128 filters, a kernel size of 7, max pooling then global max pooling, and no dropout.
Results: Achieved 90.48% test accuracy, outperforming the baseline and LSTM, but not the GRU, suggesting a limitation in handling longer contextual dependencies.
### Encoder-Only Transformer
Motivation: Transformer models are highly effective for textual data, and an encoder-only model was used for text input and probability output.
Experimentation: Stacking multiple encoders and changing MLP width yielded minimal improvements.
Findings: Transformers learned data quickly, reaching maximum validation accuracy after 1-2 epochs. Early stopping was used to prevent overfitting.
Results: Achieved up to 91.32% validation accuracy with 4 encoder layers, 8 attention heads, and a 128 intermediate dimension.
### Fine-Tuning RoBERTa Model
What is RoBERTa?: An expansion of BERT that modifies key hyperparameters and uses significantly more training data, employing dynamic masking to prevent overfitting to static masking patterns.

Challenges: Scaled up RAM usage, requiring Colab GPU and reduced batch sizes. Can overfit on imbalanced datasets. Considered a "black-box" model.
Results: Achieved 95% accuracy, with a classification report showing 0.99 precision and 0.91 recall for "Real" reviews, and 0.91 precision and 0.99 recall for "Fake" reviews.

Our Best Model
If frequent retraining on new data is necessary (due to the rapid improvement of AI text generators), the GRU model would be chosen as it had the highest accuracy among the models built from scratch. Otherwise, RoBERTa would be selected due to its superior performance. Our models trained from scratch struggled with newer AI-generated text, while RoBERTa performed well.

When we make predictions on new data our models built from scratch do not perform well. But the fine-tuned RoBERTa does well. This shows us that we need some newer training data to improve predictive power

## Future Work
Categorize Fake Review Types (Topic Modeling): Analyze fake reviews to identify distinct sub-types and their distribution, including:
Generic, Short Reviews 
Keyword-Stuffed Reviews 
Feature-Bombing Reviews 
Integrate Auxiliary Data: Incorporate product rating and item category to detect inconsistencies between review text and product metadata. This could reveal if a review's content doesn't match its star rating or category.

Explore Advanced Models: Investigate large pre-trained language models (e.g., BERT, GPT) and ensemble methods to enhance detection accuracy and robustness against evolving AI-generated text.
GitHub Repository
https://github.com/mvemuri-22/Amazon-Fake-Review-Detection
