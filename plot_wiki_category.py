import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

import json

np.random.seed(19680801)

def plot_wiki_category(m, category_limit=300):
    with open('data/evaluations/wiki_category_plot/category_instances.json', "r") as fin:
        category_instances = json.load(fin)

    # cat2colors = {cat:  np.random.rand(1)[0] for i, cat in enumerate(category_instances.keys())}
    cat2colors = {cat:  'C{}'.format(i) for i, cat in enumerate(category_instances.keys())}

    vectors = []
    terms = []
    colors = []
    for category, instances in category_instances.items():
        np.random.shuffle(instances)

        cat_count = 0
        for t in instances:
            try:
                vec = m[t]
                vectors.append(vec)
                terms.append(t)
                colors.append(cat2colors[category])
                cat_count+=1
                if cat_count >= category_limit:
                    break
                    
            except KeyError as e:
                pass


    #apply tsne 
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000, random_state=23, verbose=0)
    Yt = tsne_model.fit_transform(vectors)

    mpl.style.use('ggplot')

    plt.figure(figsize=(100,100))
    # plt.title("tSNE Plot")

    # fig, ax = plt.subplots()
    plt.scatter(Yt[:, 0], Yt[:, 1], s = 0, c=colors)

    for i, t in enumerate(terms):
    #     kwarg = {'color':colors[i]}
        plt.annotate(t, (Yt[i][0],Yt[i][1]), fontsize = 8, color=colors[i])
        
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)    
    plt.show()


def main():
    from word2vec import W2vWordVectorizer
    w2v_wv = W2vWordVectorizer(100, algorithm='cbow', min_count=0, window_size=5)
    w2v_wv.load_model('/home/chien/Desktop/NTU/Lab/word-vectors-comparison/exp_results/wiki_30mt_try2')

    plot_wiki_category(w2v_wv, 300)

if __name__ == '__main__':
    main()