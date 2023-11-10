import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass

from .classes import SoleExp

def search_sole(cfg):
    sexp = SoleExp(cfg)
    sexp.run()
    df = sexp.save_csv()
    return df

@dataclass
class SubLog:
    best_fitness : float
    avg_fitness : float
@dataclass
class TrialLogA:
    trial : int
    time : float
    best_fitness : float
    avg_fitness : float
    gen100 : SubLog
    gen300 : SubLog
    gen500 : SubLog
    gen1000 : SubLog

# function to search by trial and kept in dataframe without saving.
def search_trail(cfg):
    num_trial = cfg.num_trial
    res = []
    for i in range(num_trial):
        log = TrialLogA(0,0,0,0,None,None,None,None)
        starttime = time.time()
        df = search_sole(cfg=cfg)
        endtime = time.time()
        log.trial = i + 1
        log.time = endtime - starttime
        log.best_fitness = df['best_fitness'].iloc[-1]
        log.avg_fitness = df['avg_fitness'].iloc[-1]
        log.gen100 = SubLog(
            df['best_fitness'][100-1],
            df['avg_fitness'][100-1]
        )
        log.gen300 = SubLog(
            df['best_fitness'][300-1],
            df['avg_fitness'][300-1]
        )
        log.gen500 = SubLog(
            df['best_fitness'][500-1],
            df['avg_fitness'][500-1]
        )
        log.gen1000 = SubLog(
            df['best_fitness'][1000-1],
            df['avg_fitness'][1000-1]
        )
        res.append(
            (log.trial,
            log.time,
            log.best_fitness,
            log.avg_fitness,
            log.gen100.best_fitness, 
            log.gen100.avg_fitness,
            log.gen300.best_fitness, 
            log.gen300.avg_fitness,
            log.gen500.best_fitness, 
            log.gen500.avg_fitness,
            log.gen1000.best_fitness, 
            log.gen1000.avg_fitness,
            ))
    
    # create dataframe
    columns = pd.MultiIndex.from_tuples(
    [
        ('Trial', ''), 
        ('Time', ''),
        ('Best_fitness', ''),
        ('Avg_fitness', ''),
        ('Gen100', 'Best'), 
        ('Gen100', 'Avg'),
        ('Gen300', 'Best'), 
        ('Gen300', 'Avg'),
        ('Gen500', 'Best'), 
        ('Gen500', 'Avg'),
        ('Gen1000', 'Best'), 
        ('Gen1000', 'Avg'),
        ])
    return  pd.DataFrame(res,columns=columns)

def gens_plot(df):
    # fig , axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
    sns.lineplot(data=df,x='gens',y='best_fitness',label='best_fitness')
    sns.lineplot(data=df,x='gens',y='avg_fitness',label='avg_fitness')
    plt.show()


def trial_plot(df):
    fig , axs = plt.subplots(2,1,figsize=(12,10))
    sns.lineplot(x=df.Trial,y=df.Gen100.Best,label='100th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen500.Best,label='500th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Best,label='1000th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen100.Avg,label='100th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen500.Avg,label='500th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Avg,label='1000th',linestyle='-.',ax=axs[1])
    # sns.lineplot(x=np.arange(len(avg100)),y=avg100,label='avg100')
    # plt.title('Brazil trials')
    axs[0].axhline(y=max(df.Best_fitness),color='red')
    axs[1].axhline(y=max(df.Avg_fitness),color='red')
    axs[0].set_title('Best_finess')
    axs[0].text(10,max(df.Best_fitness) , f'max {max(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].text(10,max(df.Avg_fitness) , f'max {max(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].set_title('Avg_finess')
    plt.xlabel('trial')
    plt.ylabel('fitness_value')
    plt.show()