# Section A – Concept Application

## S1. NLP techniques for the customer-support chatbot
Two important techniques are **text preprocessing/tokenisation** and **intent classification**.

1. **Text preprocessing and tokenisation:** The chatbot should clean and split messages into useful tokens so that phrases such as “cancel my order” and “food arrived cold” can be processed consistently. Without preprocessing, punctuation, casing and unnecessary words can create noisy features and reduce matching/classification accuracy.
2. **Intent classification:** The chatbot needs to identify the user's purpose, such as order tracking, cancellation, or food-quality complaint, and route the request to the correct workflow. Without intent classification, the chatbot may recognise words but send the customer to the wrong action, reducing routing accuracy.

## S2. Text preprocessing pipeline
A suitable order is:

1. **Lowercase conversion** – makes “Food” and “food” the same feature.
2. **Punctuation/special-character handling** – removes unnecessary symbols that can create noisy features.
3. **Whitespace cleanup** – removes repeated spaces and gives a consistent string.
4. **Tokenisation** – splits the text into individual words/tokens.
5. **Stopword removal** – removes very common words such as “the”, “is”, and “a”.
6. **Stemming or lemmatisation** – reduces related word forms such as “delivered” and “delivery” to a more useful common representation where appropriate.
7. The processed tokens can then be joined back into a string and passed to a vectoriser/model.

Stopword removal can reduce the number of uninformative features. If it is skipped, frequent filler words may receive attention in the feature space, increasing noise and potentially making classification less efficient.

## S3. Bag of Words vs TF-IDF
**Bag of Words (BoW)** represents a document using word counts. It is simple, but common words can receive high importance simply because they occur frequently.

**TF-IDF** considers both how often a term occurs in a document and how common that term is across the whole corpus. For menu-search ranking, TF-IDF is useful because a word that is frequent in one menu description but uncommon across the whole corpus can receive a higher relevance weight.

For this ranking task, I would choose **TF-IDF** because it generally gives more useful relevance weights than raw word counts.

One limitation remains: TF-IDF does not truly understand word meaning or context. For example, semantically similar words may be treated as different terms.

## S4. Naive Bayes assumption
Naive Bayes assumes that the input features are **conditionally independent given the class**.

A realistic violation would be a support ticket containing strongly related phrases such as “late delivery” and “driver arrived late”. The occurrence of these words is not actually independent; they are related to the same delivery event. If the independence assumption is violated, the model can misestimate class probabilities and this may reduce classification accuracy.

## S5. Metrics for imbalanced sentiment data
Two metrics I would prioritise are:

1. **Precision** – shows how many reviews predicted as a class are actually that class.
2. **Recall** – shows how many actual reviews of a class are successfully detected.

F1-score is also useful because it combines precision and recall.

For the 80% Positive / 15% Neutral / 5% Negative distribution, one possible technique is **class weighting**, so minority classes receive greater importance during training. Another possible approach is resampling, but the choice should be evaluated on the held-out test set.

## S6. Attention in Transformers
The attention mechanism allows a transformer to consider relationships between tokens across a sequence and give different importance to relevant tokens. This is particularly useful for long text because traditional RNNs process sequences step-by-step and can struggle to preserve useful information across long distances.

A production challenge is **high inference latency and memory/compute usage**, especially with several hundred requests per minute. A common mitigation is **batching requests**, and other production approaches include using a smaller/distilled model, caching, quantisation, or GPU inference.
