# NER-Entity-Recognition
# Named Entity Recoognition

### Word Embeddings:
  For the given baseline model, Glove25 word embeddings are used as input features. However, we can combine different types of input features such as POS tags, NER tags, length of word, word embeddings, TFIDF etc. For our experiments we experimented with an array of features and their combinations. The different features were concatenated together and then passed to the models. The main features we experimented with and their implementation details are as follows:
1. Glove Embeddings
2. Fasttext Embeddings

### Part-Of-Speech Tags (POS)
  The Part-of-Speech tags are associated with each word that define the usage and function of a word in the given sentence and categorizes them as nouns, verbs, adjectives, adverbs etc. A particular word can have more than one POS tags associated with it based on its usage in the sentence. For instance, the word ‘run’ can be a ‘noun’ or a ‘verb’ depending on its usage in a sentence. POS tags make the word-based features strong as when a speech of tag is combined the word
feature, it helps in preserving the context, thereby strengthening the features. For our problem, hence, it was important to include the POS tags as features. After tokenization, POS tags were assigned to each word of the dataset using Stanford’s average perceptron tagger. Later the one-hot encoded feature vectors were generated for each word keeping in mind that more than one tags can be associated with a word. Each vector was of the size 43 (equal to the no. of unique tags).

### Named Entity Recognition (NER)
  The process of detecting the named entities such as person names, location names, company names etc. from the text is called as NER.[https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-naturallanguage-processing-codes-in-python/] 
The main idea to use pre-trained NER was to give direction to the model to predict the tags. The given dataset consists of the IO tags and is a part of CONLL-2003 dataset. We first perform the Noun-Phrasing on our dataset by using NLTK RegexParser on top of our POS tag results and then NP-chunking to get the IOB tags. Similar to the POS tags, one word can have more than one IOB tags as well. Hence, IOB tags were converted to features in the same way.

### TFIDF
  It aims to convert the text documents into weighted vector models on the basis of occurrence of words in the documents without taking considering the exact ordering, by computing the relative importance of a word in a document.

### Proposed Model Architecture
We will tackle the NER Tag prediction problem as a sequence classification problem. However, rather than using standard deep learning architecture, we have chosen a encoder-decoder based architecture. The encoder will help us to extract a contextual sentence information which can be utilized by a decoder to classify each NER tag.
1. Context Encoder :
A simple bi-GRU unit that runts through training dataset and stores context information on a sentence level.
  1.1. Input:
    * Embedding Dimension Size - dynamice depnding on features selected
    * Hidden Layer Size - parameterised value
  1.2. Return:
    * Output and Hidden layer encodings

2. Attention Decoder :
An bi-GRU unit (bi-directional / Layer switch available) + Linear layer that takes in encoded context information from step 1, calculates attention, and classifies NER tag based on Conditional Random Field scoring. The Decoder works in following steps
  2.1. RNN-based encodings:
    * Input : Embedding matrix
  2.1. Attention Calculation:
    * Input 1 : Encoded hidden states for overall sentence (contextual information)
    * Input 2 : bi-Gru output from step 1
    * Returns : Concatenated product based on attention method selected (Dot or Scaled dot)
  2.3. Linear Layer:
    * Input : Attention concatenated layer
    * Returns : Flattened output into tag space
  2.4. CRF Scoring :
    * Input 1 : Flattened tag space
    * Input 2 : Target NER tags
    * Returns : Predicted sequence based on Viterbi scoring
