import umap
from sklearn.manifold import TSNE
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import fire

#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ntu_one_shot_classes = [
    "drink water",
    "throw",
    "tear up paper",
    "take off glasses",
    "reach into pocket",
    "pointing to something with finger",
    "wipe face",
    "falling",
    "feeling warm",
    "hugging other person",
    "put on headphone",
    "hush (quite)",
    "staple book",
    "sniff (smell)",
    "apply cream on face",
    "open a box",
    "arm circles",
    "yawn",
    "grab other person’s stuff",
    "take a photo of other person"]

ntu_one_shot_classes_ids = range(1,21)
#ntu_one_shot_classes = range(1,21)

def plot_embedding(filename, target_filename, mode="umap", legend=False, size=10, color_map="utdmhad"):
    """
    Plots an embedding from a dumped pickel file

    :filename: filename to the pickle file
    :target_filename: output file, different extention fives different format
    :mode: could be either `umap` or `tsne`, depending on your preference
    :legend: if 'True' plots a legend
    """
    print(len(ntu_one_shot_classes))
    with open(filename, "rb") as f:
        a = pickle.load(f)
    num_samples = len(a[0])

    if mode == "tsne":
        x_reduced = TSNE(n_components=2).fit_transform(a[0])
    if mode == "umap":
        x_reduced = umap.UMAP(random_state=42).fit_transform(a[0])
    y = a[1][:num_samples].flatten()
    print(x_reduced.shape, y.shape)


    #plt.figure(figsize=(, 5))
    plt.rcParams['legend.fontsize'] = 'xx-small'
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    if color_map == "ntu":
        colors = plt.cm.get_cmap("tab20b").colors
    else:
        colors = plt.cm.get_cmap("Set1").colors
    #colors = plt.cm.get_cmap("hsv", 20).colors
    for i, c, label in zip(ntu_one_shot_classes_ids, colors, ntu_one_shot_classes):
        #print(i, c, label)
        markerx = "o" if i % 2 else "x"
        #size = size
        print(label, len(x_reduced[y==i-1]))
        plt.scatter(x_reduced[y == i-1, 0][:], x_reduced[y == i-1, 1][:],
                    label=label, c=c, s=size, marker=markerx, alpha=1.0)
        #plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1], label=label, s=5, marker=markerx, alpha=0.5)
    if legend:
        plt.legend(loc='center right', bbox_to_anchor=(1.3, 0.5))
    plt.savefig(target_filename, bbox_inches='tight')
    plt.show()


    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("Set1", 20),alpha=1.0)
    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("rainbow", 27),alpha=0.5)
    #scatter = plt.scatter(x_reduced[:num_samples, 0], x_reduced[:num_samples, 1], c=y.flatten(), cmap=plt.cm.get_cmap("tab20b", len(ntu_one_shot_classes)),alpha=1.0, s=1)
    #plt.legend(handles=scatter.legend_elements()[0], labels=ntu_one_shot_classes)
    #plt.colorbar(ticks=ntu_one_shot_classes_ids)
    #plt.show()
    #plt.savefig(target_filename, bbox_inches='tight')

# Side helpers for paper writing
def get_tex_colors_for_matplotlib_colors():
     for i in zip(ntu_one_shot_classes, plt.cm.get_cmap("tab20b").colors): 
        s = "\definecolor{%s}{rgb}{%2f, %2f, %2f}"%((i[0].replace(" ","_")).replace("’",""), i[1][0], i[1][1], i[1][2]) 
        print(s)

def get_colorized_labels_colors_for_matplotlib_colors():
     for i in zip(ntu_one_shot_classes, plt.cm.get_cmap("tab20b").colors): 
        s = "\\textcolor{%s}{%s},"%((i[0].replace(" ","_")).replace("’",""), i[0]) 
        print(s)

def get_colorized_labels_colors_for_matplotlib_colors_squared():
     for i in zip(ntu_one_shot_classes, plt.cm.get_cmap("tab20b").colors): 
        s = "%s \\textcolor{%s}{\\bullet},"%(i[0], (i[0].replace(" ","_")).replace("’","")) 
        print(s)

if __name__ == '__main__':
    fire.Fire(plot_embedding)
