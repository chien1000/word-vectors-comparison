import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from math import floor, ceil

import json
import os 

np.random.seed(19680801)

def plot_wiki_category(m, category_limit=300, save_path=None):
    with open('data/evaluations/wiki_category_plot/category_instances.json', "r") as fin:
        category_instances = json.load(fin)

    # cat2colors = {cat:  np.random.rand(1)[0] for i, cat in enumerate(category_instances.keys())}
    selected_colors = ['rosybrown', 'mediumblue', 'goldenrod','forestgreen','darkorchid','crimson']
    cat2colors = {cat:selected_colors[i] for i, cat in enumerate(category_instances.keys())}
    
    vectors = []
    terms = []
    colors = []
    cat_cum = {}
    last_cat_count=0
    for category, instances in category_instances.items():
        # np.random.shuffle(instances)

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
        cat_cum[category] = last_cat_count + cat_count
        last_cat_count += cat_count

    #apply tsne 
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000, random_state=23, verbose=0)
    Yt = tsne_model.fit_transform(vectors)

    mpl.style.use('seaborn-dark-palette')

    plt.figure(figsize=(10,10), dpi=100)
    # plt.title("tSNE Plot")

    # fig, ax = plt.subplots()
    # plt.scatter(Yt[:, 0], Yt[:, 1], s = 0, c=colors)
    start = 0
    for cat, cum_ind in sorted(list(cat_cum.items()), key=lambda x: x[1]):
        plt.scatter(Yt[start:cum_ind, 0], Yt[start:cum_ind, 1], s = 2, c=colors[start:cum_ind], label=cat)
        start = cum_ind

    for i, t in enumerate(terms):
    #     kwarg = {'color':colors[i]}
        plt.annotate(t, (Yt[i][0],Yt[i][1]), fontsize = 10, color=colors[i])
    
    sort_coor = np.sort(Yt[:, 0])
    xmin, xmax = floor(sort_coor[5]-1), ceil(sort_coor[-5]+1) #avoid extreme value
    sort_coor = np.sort(Yt[:, 1])
    ymin, ymax = floor(sort_coor[5]-1), ceil(sort_coor[-5]+1)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)   

    plt.legend(loc='lower right', ncol=3, markerscale=5)

    if save_path:
        save_file = os.path.join(save_path, 'wiki_plot_{}.png'.format(m.get_mid()))
        plt.savefig(save_file, dpi=1000, bbox_inches='tight')

    # plt.show()

def main():
    from word2vec import W2vWordVectorizer
    w2v_wv = W2vWordVectorizer(100, algorithm='cbow', min_count=0, window_size=5)
    w2v_wv.load_model('/home/chien/Desktop/NTU/Lab/word-vectors-comparison/exp_results/wiki_30mt_try2')

    plot_wiki_category(w2v_wv, 100,  './figs')

if __name__ == '__main__':
    main()
