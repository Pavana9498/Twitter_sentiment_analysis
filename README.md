Performed sentiment analysis over a corpus of tweets during the U.S. 2012 Re-Election about the candidates Barack Obama and Mitt Romney. The sentiment analysis had several tasks and stages in itself.
How to run the code: python3 test_model.py

Tasks:
1. Cleaning: The text corpus that has been provided to us had many
tweets that belonged to class 2 I.e they were of mixed opinions about the candidates which were deemed to be unnecessary for the sentiment classifier that is being built. Thus these rows were removed.
2. Pre-processing: The text corpus had irrelevant information such as hash tags, URLs, numbers. The tweets were pre-processed by removing all the irrelevant information. Other pre-processing such as tokenisation, n-grams, tf-idf vectorisation were also performed on the text corpus.
3. Training / Testing: The pre-processed detests were then combined into a single text corpus, for an approach we describe as “Joint training.” The naive Bayes model is then trained and tested on the unseen dataset provided.
Techniques tried:
1. Logistic regression: The parameter grid is generated and the
parameters of the model are then optimised by cross-validated grid- search over the parameters grid. The model is trained on 100 fold stratified cross validation, to obtain 100 different models.
2. Naive Bayes model: The n-grams from the pre-processed tweets of length 1 and 2 are generated and provided as input. The features are then extracted by vectorising the n-grams text using tf-idf vectorisation. The model with the given features is trained on 100 fold stratified cross validation, to obtain 100 different models. The model is optimised using cross validation.

Results:
The naive Bayes model performed the best, getting a test-time score of 60% beating the score obtained by logistic regression which was 58.7% on the combined test datasets. When tested separately on the two test datasets provided, the naives Bayes model was still the best performer,

 getting a test-time score of 61% on Obama’s dataset and 59.1% on Romney’s dataset.
Conclusion:
The results suggest that improvements can be made by using few more machine learning techniques like ensemble learning or soft voting classifier or Deep learning algorithms like LSTM RNN etc. We can also improve this further using better feature selection for training the models.
