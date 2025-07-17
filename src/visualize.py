from wordcloud import WordCloud
import matplotlib.pyplot as plt

def make_wordcloud(problem_text: str, stopwords: set = None):
    """
    Generate and display a word cloud from the given word problem text.
    """
    if stopwords is None:
        stopwords = {"the", "and", "to", "a", "of", "in", "on", "for", "is", "with", "each"}
    
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords)
    wc.generate(problem_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of LP Problem")
    plt.show()
