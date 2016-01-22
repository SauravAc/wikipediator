# wikipediator
Wikipediator - It fetches the Wikipedia article that is most relevant to a given block of text.
It uses an n-gram model to extract noun phrases that are likely to be the candidate topic of the input text, and chooses the Wikipedia article with the best tf-idf weighted cosine similarity to the input text.
