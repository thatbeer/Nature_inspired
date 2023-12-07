import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dataclasses import dataclass

from .classes import SoleExp

# utilize function to run the SoleExp without using yaml file
# the function return dataframe of 1 trial of genetic algorithm experiment.
def search_sole(cfg):
    sexp = SoleExp(cfg)
    sexp.run()
    df = sexp.save_csv()
    return df # dataframe of the generation from 1 to max-generation

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
# used to collect data for experiment analysis
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
        # collect the 100th genration information
        log.gen100 = SubLog(
            df['best_fitness'][100-1],
            df['avg_fitness'][100-1]
        )
        # collect the 300th genration information
        log.gen300 = SubLog(
            df['best_fitness'][300-1],
            df['avg_fitness'][300-1]
        )
        # collect the 500th genration information
        log.gen500 = SubLog(
            df['best_fitness'][500-1],
            df['avg_fitness'][500-1]
        )
        # collect the 1000th genration information
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

# function to search by trial and kept in dataframe without saving
# specific to generate for 3000 max generations
# used to collect data for experiment analysis
def search_trail3k(cfg):
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
        log.gen3000 = SubLog(
            df['best_fitness'][3000-1],
            df['avg_fitness'][3000-1]
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
            log.gen3000.best_fitness, 
            log.gen3000.avg_fitness,
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
        ('Gen3000', 'Best'), 
        ('Gen3000', 'Avg'),
        ])
    return  pd.DataFrame(res,columns=columns)

# specific to generate for 5000 max generations
# used to collect data for experiment analysis
def search_trail5k(cfg):
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
        log.gen3000 = SubLog(
            df['best_fitness'][3000-1],
            df['avg_fitness'][3000-1]
        )
        log.gen5000 = SubLog(
            df['best_fitness'][5000-1],
            df['avg_fitness'][5000-1]
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
            log.gen3000.best_fitness, 
            log.gen3000.avg_fitness,
            log.gen5000.best_fitness, 
            log.gen5000.avg_fitness,
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
        ('Gen3000', 'Best'), 
        ('Gen3000', 'Avg'),
        ('Gen5000', 'Best'), 
        ('Gen5000', 'Avg'),
        ])
    return  pd.DataFrame(res,columns=columns)

# visualize the single trial of genetic algorithm to observe the pattern of fitness value vs iterations
def gens_plot(df):
    # fig , axs = plt.subplots(2,1,sharex=True,figsize=(10,6))
    sns.lineplot(data=df,x='gens',y='best_fitness',label='best_fitness')
    sns.lineplot(data=df,x='gens',y='avg_fitness',label='avg_fitness')
    plt.text(df.gens.iloc[-1]*0.9,df.best_fitness.iloc[-1]*1.05,f'v:{df.best_fitness.iloc[-1]}')
    plt.show()

# visualize the average best fintess value and the average of average fitness value from each trial
# to understand the quality of the model and its consistency.
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

    axs[0].axhline(y=np.mean(df.Best_fitness),color='blue',linestyle='-.')
    axs[1].axhline(y=np.mean(df.Avg_fitness),color='blue',linestyle='-.')
    
    axs[0].set_title('Best_finess')
    axs[0].text(df.Trial.iloc[-1]*0.5,max(df.Best_fitness) , f'max {max(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].text(df.Trial.iloc[-1]*0.5,max(df.Avg_fitness) , f'max {max(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    axs[0].text(df.Trial.iloc[-1]*0.8,np.mean(df.Best_fitness) , f'mean {np.mean(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].text(df.Trial.iloc[-1]*0.8,np.mean(df.Avg_fitness) , f'mean {np.mean(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].set_title('Avg_finess')
    plt.xlabel('trial')
    plt.ylabel('fitness_value')
    plt.show()
    
# visualize the average best fintess value and the average of average fitness value from each trial
# to understand the quality of the model and its consistency with specific 3000 max generation
def trial3k_plot(df):
    fig , axs = plt.subplots(2,1,figsize=(12,10))
    sns.lineplot(x=df.Trial,y=df.Gen100.Best,label='100th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen500.Best,label='500th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Best,label='1000th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen3000.Best,label='3000th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen100.Avg,label='100th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen500.Avg,label='500th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Avg,label='1000th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen3000.Avg,label='3000th',linestyle='-.',ax=axs[1])
    # sns.lineplot(x=np.arange(len(avg100)),y=avg100,label='avg100')
    # plt.title('Brazil trials')
    axs[0].axhline(y=max(df.Best_fitness),color='red')
    axs[1].axhline(y=max(df.Avg_fitness),color='red')

    axs[0].axhline(y=np.mean(df.Best_fitness),color='blue',linestyle='-.')
    axs[1].axhline(y=np.mean(df.Avg_fitness),color='blue',linestyle='-.')
    
    axs[0].set_title('Best_finess')
    # axs[0].text(df.Trial.iloc[-1]*0.5,max(df.Best_fitness)*1.1 , f'max {max(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    # axs[1].text(df.Trial.iloc[-1]*0.5,max(df.Avg_fitness)*1.1 , f'max {max(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    # axs[0].text(df.Trial.iloc[-1]*0.8,np.mean(df.Best_fitness)*0.9 , f'mean {np.mean(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    # axs[1].text(df.Trial.iloc[-1]*0.8,np.mean(df.Avg_fitness)*0.9 , f'mean {np.mean(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    # axs[1].set_title('Avg_finess')
    plt.xlabel('trial')
    plt.ylabel('fitness_value')
    plt.show()

# visualize the average best fintess value and the average of average fitness value from each trial
# to understand the quality of the model and its consistency with specific 5000 max generation
def trial5k_plot(df):
    fig , axs = plt.subplots(2,1,figsize=(12,10))
    sns.lineplot(x=df.Trial,y=df.Gen100.Best,label='100th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen500.Best,label='500th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Best,label='1000th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen3000.Best,label='3000th',linestyle='--',ax=axs[0])
    sns.lineplot(x=df.Trial,y=df.Gen100.Avg,label='100th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen500.Avg,label='500th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen1000.Avg,label='1000th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen3000.Avg,label='3000th',linestyle='-.',ax=axs[1])
    sns.lineplot(x=df.Trial,y=df.Gen5000.Avg,label='5000th',linestyle='-.',ax=axs[1])
    # sns.lineplot(x=np.arange(len(avg100)),y=avg100,label='avg100')
    # plt.title('Brazil trials')
    axs[0].axhline(y=max(df.Best_fitness),color='red')
    axs[1].axhline(y=max(df.Avg_fitness),color='red')

    axs[0].axhline(y=np.mean(df.Best_fitness),color='blue',linestyle='-.')
    axs[1].axhline(y=np.mean(df.Avg_fitness),color='blue',linestyle='-.')
    
    axs[0].set_title('Best_finess')
    axs[0].text(df.Trial.iloc[-1]*0.5,max(df.Best_fitness)*1.1 , f'max {max(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].text(df.Trial.iloc[-1]*0.5,max(df.Avg_fitness)*1.1 , f'max {max(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    axs[0].text(df.Trial.iloc[-1]*0.8,np.mean(df.Best_fitness)*0.9 , f'mean {np.mean(df.Best_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].text(df.Trial.iloc[-1]*0.8,np.mean(df.Avg_fitness)*0.9 , f'mean {np.mean(df.Avg_fitness)}', color='blue', fontsize=12, ha='center')
    axs[1].set_title('Avg_finess')
    plt.xlabel('trial')
    plt.ylabel('fitness_value')
    plt.show()