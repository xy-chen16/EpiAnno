from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
def sort_by_classes(X, y, classes):
    if classes is None:
        classes = np.unique(y)
    index = []
    #print(classes)
    #print(y)
    for c in classes:
        #print(np.where(y==c))
        ind = np.where(y==c)[0]
        #print(ind)
        index.append(ind)
    index = np.concatenate(index)
    #print(index)
    X = X[:,index]
    y = y[index]
    return X, y, classes, index
def plot_heatmap(X, y, classes=None, y_pred=None, row_labels=None, colormap=None, row_cluster=False,
                 cax_title='', xlabel='', ylabel='', yticklabels='', legend_font=10, 
                 show_legend=True, show_cax=True, tick_color='black', ncol=3,
                 bbox_to_anchor=(0.5, 1.3), position=(0.8, 0.78, .1, .04), return_grid=False,
                 save=None, **kw):
    """
    plot hidden code heatmap with labels

    Params:
        X: fxn array, n is sample number, f is feature
        y: a array of labels for n elements or a list of array
    """

    import matplotlib.patches as mpatches  # add legend
    # if classes is not None:
    X, y, classes, index = sort_by_classes(X, y, classes)
    # else:
        # classes = np.unique(y)

    if y_pred is not None:
        y_pred = y_pred[index]
        classes = list(classes) + list(np.unique(y_pred)) 
        if colormap is None:
            colormap = plt.cm.tab20
            colors = {c:colormap(i) for i, c in enumerate(classes)}
        else:
            colors = {c:colormap[i] for i, c in enumerate(classes)}
        col_colors = []
        col_colors.append([colors[c] for c in y])
        col_colors.append([colors[c] for c in y_pred])
    else:
        if colormap is None:
            colormap = plt.cm.tab20
            colors = {c:colormap(i) for i, c in enumerate(classes)}
        else:
            colors = {c:colormap[i] for i, c in enumerate(classes)}
        col_colors = [ colors[c] for c in y ]
        
    legend_TN = [mpatches.Patch(color=color, label=c) for c, color in colors.items()]

    if row_labels is not None:
        row_colors = [ colors[c] for c in row_labels ]
        kw.update({'row_colors':row_colors})

    kw.update({'col_colors':col_colors})

    cbar_kws={"orientation": "horizontal"}
    grid = sns.clustermap(X, yticklabels=True, 
            col_cluster=False,
            row_cluster=row_cluster,
            cbar_kws=cbar_kws, **kw)
    if show_cax:
        grid.cax.set_position(position)
        grid.cax.tick_params(length=1, labelsize=4, rotation=0)
        grid.cax.set_title(cax_title, fontsize=6, y=0.35)

    if show_legend:
        grid.ax_heatmap.legend(loc='upper center', 
                               bbox_to_anchor=bbox_to_anchor, 
                               handles=legend_TN, 
                               fontsize=legend_font, 
                               frameon=False, 
                               ncol=ncol)
        grid.ax_col_colors.tick_params(labelsize=6, length=0, labelcolor='orange')
 
    if (row_cluster==True) and (yticklabels is not ''):
        yticklabels = yticklabels[grid.dendrogram_row.reordered_ind]

    grid.ax_heatmap.set_xlabel(xlabel)
    grid.ax_heatmap.set_ylabel(ylabel, fontsize=8)
    grid.ax_heatmap.set_xticklabels('')
    grid.ax_heatmap.set_yticklabels(yticklabels, color=tick_color)
    grid.ax_heatmap.yaxis.set_label_position('left')
    grid.ax_heatmap.tick_params(axis='x', length=0)
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0, rotation=0, labelleft=True, labelright=False)
    grid.ax_row_dendrogram.set_visible(False)
    grid.cax.set_visible(show_cax)
    grid.row_color_labels = classes

    if save:
        plt.savefig(save, format='pdf', bbox_inches='tight')
    else:
        plt.show()
    if return_grid:
        return grid
