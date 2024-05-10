# -*- coding: utf-8 -*-clear


"""
-------------------------------------------------------------------------------

    Authors:    Parshan Pakiman  | https://parshanpakiman.github.io/homepage/
                Selva Nadarajah  | https://selvan.people.uic.edu/
                         
    Licensing Information: The MIT License
-------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

rc_fonts = {
    "font.family": "serif",
    "font.size": 10,
    "text.usetex": True,
}
mpl.rcParams.update(rc_fonts)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def depict_error_bar(data,ax,pos,color,delta):

        parts = ax.violinplot(dataset= data,
                      positions = [pos],
                      vert = True,
                      showextrema = False,
                      widths=.3)
        
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(.2)
            
        ax.vlines(x=[pos],ymin=[min(data)],ymax=[max(data)],colors=[color],linewidths=1 )
        ax.hlines(y=[min(data)],xmin=[pos -delta/2],xmax=[pos+delta/2],colors=[color],linewidths=1 )
        ax.hlines(y=[max(data)],xmin=[pos -delta/2],xmax=[pos+delta/2],colors=[color],linewidths=1 )
        
        ax.scatter(x=pos,y=np.mean(data),color=color,s=100,marker='.')
        

def plot():
    
    
    fig, axs    = plt.subplots(1,2,figsize = (9,3),sharex= True,sharey = False)
    axs         = axs.flatten()
    seed_list   = [111,222,333,444,555,666,777,888,999,1010]
    b_values    = [0,1,2,3,4,5]
    handles = [mpatches.Rectangle((0,0),.1,.1,color='blue',alpha=.8,label='Upper bound'),
               mpatches.Rectangle((0,0),.1,5,color='red',alpha=.8, label='Lower bound'),
               mpatches.Rectangle((0,0),.1,5,color='green',alpha=.2,label='Mean optimality gap $\%$')
        ]
    
    for instance_itr,instance in enumerate(['19','20']):
        
        path        = '../Output/PIC/instance_' + instance
        file_name   = lambda seed: '/PIC_fourier_SGFALP_uniform_non_adaptive' + \
                                    '_inner_update_0_Batch_100_seed_' + str(seed)+'.csv'
                                    
    
        mean_ub     = []
        mean_lb     = []
        mean_gap    = []
        max_ub      = []
        
        for b in b_values:
            ub_list     = []
            lb_list     = []
            gap_list    = []
            
            for seed in seed_list:
                data    = pd.read_csv(path + file_name(seed))   
                ub      = list(data['policy cost mean']) 
                lb      = list(data['best_lower_bound'])
                
                gap_list.append(100*(ub[b] - lb[b]) / lb[b])
                
                ub      = np.log(ub) 
                lb      = np.log(lb) 
                
                ub_list.append(ub[b])
                lb_list.append(lb[b])
                
                
            max_ub.append(max(ub_list))
                
        
            depict_error_bar(ub_list,axs[instance_itr],b,'blue',.2)
            depict_error_bar(lb_list,axs[instance_itr],b,'red',.2)
    
            
            mean_ub.append(     np.mean(ub_list)  ) 
            mean_lb.append(     np.mean(lb_list)  )  
            mean_gap.append(    np.mean(gap_list))

        label_candidates = np.round(np.exp(mean_lb+max_ub+ mean_ub))
    
    
        for b in b_values:
    
            axs[instance_itr].text(
                    b,
                    max_ub[b]+.15,
                    r'' + str(int(np.round(mean_gap[b])))+'$\%$',
                    fontsize=9,
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(color='green',alpha=.1, pad=1))
            
        
        
        chosen_label_candidates = []
        for _ in label_candidates:
            if len(chosen_label_candidates)==0:
                chosen_label_candidates.append(int(_))
            else:
                dist = min([abs(x-_) for x in chosen_label_candidates])
                if dist>400 and not(int(_) ==20144):
                    chosen_label_candidates.append(int(_))
            
        
    
        axs[instance_itr].set_yticks(ticks = np.log(chosen_label_candidates),
                                     labels=chosen_label_candidates
                                     ,fontsize=10)
        
        axs[instance_itr].plot(b_values, mean_ub, color='blue', lw=1)
        axs[instance_itr].plot(b_values, mean_lb, color='red', lw=1)
        axs[instance_itr].set_xlim([-.5,5.5])
        axs[instance_itr].set_ylim(np.log([700,80000]))
        axs[instance_itr].grid(axis='y',which='major',alpha=.6,zorder=0)
        axs[instance_itr].set_xlabel(r'Iteration $q$',fontsize=11)
        if instance_itr== 0:
            axs[instance_itr].set_ylabel(r'Upper and lower bound values',fontsize=11)
 
        
        
        axs[instance_itr].legend(handles = handles,
                      ncol    = 1,
                       fontsize = 9,
                      )
        
    plt.tight_layout()
    
    
    



    plt.subplots_adjust(bottom=0.15,right=.99,wspace=.18)
    
    plt.savefig('SG_FALP_lb_ub_gap.pdf', dpi=300)

plot()


