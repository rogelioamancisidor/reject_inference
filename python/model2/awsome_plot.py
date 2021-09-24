# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 23:06:52 2020

@author: rma050
"""
def scatter_hist(x,y,x2,y2,pi,pi2,path):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.ticker import FuncFormatter
    matplotlib.use('Agg')

    # definitions for the axes
    left, width = 0.28, 0.25
    bottom, height = 0.12, 0.65
    spacing = 0.017


    rect_scatter  = [left                   , bottom, width, height]
    rect_scatter2 = [left+width+spacing-0.02     , bottom, width-0.05, height]
    rect_histx1   = [left - 0.18 -spacing   , bottom, 0.17  , height]
    rect_histx2   = [left+width*2+spacing*2 -0.07, bottom, 0.17  , height]

    # start with a rectangular Figure
    plt.figure(figsize=(20, 5))

    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=False, labelleft=True, labelbottom=True)
    ax_scatter.grid(True)
    ax_scatter.set_title('Latent space for y = 0')

    ax_scatter2 = plt.axes(rect_scatter2)
    ax_scatter2.tick_params(direction='in', top=True, right=True, labelleft=False, labelbottom=True)
    ax_scatter2.grid(True)
    ax_scatter2.set_title('Latent space for y = 1')

    # makes box for hist to the right
    ax_histx1 = plt.axes(rect_histx1)
    ax_histx1.tick_params(direction='in', labelleft=False, labelbottom=False, bottom = False)
    ax_histx2 = plt.axes(rect_histx2)
    ax_histx2.tick_params(direction='in', labelright=False,labelbottom=False,bottom=False, right=False)

    # the scatter plot:
    pcm=ax_scatter.scatter(x, y, c=pi,s=4)

    cbar = plt.colorbar(pcm,ax=ax_scatter)
    cbar.ax.set_yticklabels(["{:.0%}".format(i) for i in cbar.get_ticks()]) 
    ax_scatter2.scatter(x2, y2, c=pi2,s=4)

    # now determine nice limits by hand:
    binwidth = 0.25
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth
    ax_scatter.set_xlim((-lim, lim))
    ax_scatter.set_ylim((-lim, lim))
    ax_scatter2.set_xlim((-lim, lim))
    ax_scatter2.set_ylim((-lim, lim))
    ax_scatter.grid(True)
    ax_scatter2.grid(True)

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx1.hist(pi, orientation='horizontal',histtype='stepfilled')
    ax_histx1.invert_xaxis()
    ax_histx1.spines['left'].set_visible(False)
    ax_histx1.spines['bottom'].set_visible(False)
    ax_histx1.spines['top'].set_visible(False)
    ax_histx1.yaxis.set_ticks_position('right')
    ax_histx1.tick_params(axis='y', labelrotation=90)
    ax_histx1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    ax_histx2.hist(pi2, orientation='horizontal')
    ax_histx2.spines['right'].set_visible(False)
    ax_histx2.spines['bottom'].set_visible(False)
    ax_histx2.spines['top'].set_visible(False)
    ax_histx2.yaxis.set_ticks_position('left')
    ax_histx2.tick_params(axis='y', labelrotation=-90)
    ax_histx2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
    plt.savefig(path)

