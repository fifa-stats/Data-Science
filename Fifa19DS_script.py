#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[2]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[3]:


df = pd.read_csv('data.csv')


# ## Grouping Players Dataset by Teams

# In[4]:


df_teams = df


# ### Dropping NaNs in `Club` and `Position` Features

# In[5]:


df_teams = df_teams.drop('Unnamed: 0', axis=1)


# In[6]:


df_teams = df_teams.dropna(subset=['Club', 'Position'], axis=0)


# ### Goal Keeper rows: Replacing NaNs with 0s in `Position` Column

# In[7]:


# Raplacing NaNs with 0s for Goal Keeper rows
df_teams.iloc[:,27:53] = df_teams.iloc[:,27:53].fillna(value=0)


# ### Dropping `Joined` and Replacing NaNs in `Release Clause` and `Loaned From`

# In[8]:


# Dropping 'Joined' column
df_teams = df_teams.drop('Joined', axis=1)


# In[9]:


# Replacing NaNs in 'Release Clause' and 'Loaned From' features
df_teams['Release Clause'] = df_teams['Release Clause'].fillna(0)
df_teams['Loaned From'] = df_teams['Loaned From'].fillna('Not Loaned')


# ### Adding `Field Position` Feature

# In[10]:


defense = ['CB', 'RB', 'LB', 'RWB', 'LWB', 'RCB', 'LCB']
midfield = ['RW', 'LW', 'RM', 'LM', 'CM', 'CDM', 'CAM', 'RCM', 'LCM', 'LAM', 'RAM', 'RDM', 'LDM']
attack = ['ST', 'CF', 'RF', 'LF', 'RS', 'LS']
goalkeeper = ['GK']


# In[11]:


# function to create Field Position for each player
def field(row):
    if row['Position'] in defense:
        val = 'Defense'
    elif row['Position'] in midfield:
        val = 'Midfield'
    elif row['Position'] in attack:
        val = 'Attack'
    else:
        val = 'GK'
    return val


# In[12]:


df_teams['Field Position'] = df_teams.apply(field, axis=1)


# ### Editing values in `Value` and `Wage` columns

# In[13]:


def change_value(row):
    if (row['Value'][-1]=='K'):
        return int(pd.to_numeric(row['Value'][1:-1])*1000)
    elif (row['Value'][-1]=='M'):
        return int(pd.to_numeric(row['Value'][1:-1])*1000000)
    elif (row['Value'][-1]=='0'):
        return 0


# In[14]:


df_teams['Value'] = df_teams.apply(change_value, axis=1)


# In[15]:


def change_wage(row):
    if (row['Wage'][-1]=='K'):
        return (pd.to_numeric(row['Wage'][1:-1]))
    elif (row['Wage'][-1]=='0'):
        return 0


# In[16]:


df_teams['Wage'] = df_teams.apply(change_wage, axis=1)


# ### Applying Player Overvalue Ratio

# In[17]:


df_teams['Overvalue Ratio'] = df_teams['Wage'] / df_teams['Overall']
df_teams['Overvalue Ratio'] = df_teams['Overvalue Ratio'].apply(lambda x: round(x,2))


# ### Multiplying Wage by 1000

# In[18]:


df_teams['Wage'] = df_teams['Wage']*1000


# ### Adding New Feature: Field Position Num (Numerically Encoded Field Positions)

# In[19]:


le = LabelEncoder()
df_teams['Field Position Num'] = le.fit_transform(df_teams['Field Position'])


# ### Getting Club players

# In[20]:


def get_club_players(club):
    if club in df_teams.Club.values:
        club_players = []
        club = df_teams['Club'] == club
        vals = df_teams.loc[club]
        
        names = list(vals['Name'])
        overvalue_ratios = list(vals['Overvalue Ratio'])
        potential = list(vals['Potential'])
        overall_ratings = list(vals['Overall'])  # list of player ratings from highest to lowest
        position = list(vals['Position'])
        wage = list(vals['Wage'])
        
    # If club is a user's 'custom' club    
    elif type(club) == list:
        player_indexes = []
        for dic in club:
            for key, value in dic.items():
                if key == "id":
                    player_indexes.append(value -1)
        
        names = df_teams.iloc[player_indexes]['Name'].values
        overvalue_ratios = df_teams.iloc[player_indexes]['Overvalue Ratio'].values
        potential = df_teams.iloc[player_indexes]['Potential'].values
        overall_ratings = df_teams.iloc[player_indexes]['Overall'].values
        position = df_teams.iloc[player_indexes]['Position'].values
        wage = df_teams.iloc[player_indexes]['Wage'].values
    else:
        return 'Your club entry was not located.'
    
    club_list = list(zip(names, position, overvalue_ratios, overall_ratings, potential, wage))

    cols = ['Name', 'Position', 'Overvalue Ratio', 'Overall Rating', 'Potential Rating', 'Wage']
    df_club_list = pd.DataFrame(data=club_list, columns=cols)


    top_2_rated_players = sorted(club_list, key=lambda x: x[2], reverse=True)[:2]

    df_top_2_rated_players = pd.DataFrame(data=top_2_rated_players, columns=cols)


    bottom_2_rated_players = sorted(club_list, key=lambda x: x[2], reverse=True)[-2:]

    df_bottom_2_rated_players = pd.DataFrame(data=bottom_2_rated_players, columns=cols)


    # desc_club_list_by_performance_ratio = sorted(club_list, key=lambda x: x[1], reverse=True)

    return df_club_list, df_top_2_rated_players, df_bottom_2_rated_players
    


# ### Recommender System

# In[21]:


# dataframe with features for correlation function
df_attributes = df_teams[['Field Position Num', 'Overall', 'Potential', 'Crossing', 'Finishing', 
                          'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
                          'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 
                          'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 
                          'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 
                          'Vision', 'Penalties', 'Composure', 'Marking']]


# In[22]:


# Create function that uses above 'suggested' variables to output x players to potentially obtain by trade
# for df_attributes dataframe, see above cells

def get_suggested_trades(club):  # player argument changed to 'club', after get_club_players refactored
    trades_p1 = []  # this will be the output object that club_suggested_changes receives/uses
    trades_p2 = []
    players_wages = []

    if club in df_teams.Club.values or type(club) == list:  # either gets existing club from db, or uses custom list
        all_players, top_2, bottom_2 = get_club_players(club)  # df_club_list, df_top_2_players, df_bottom_2_players
        
        # looping throught 2 player names in 'top_2'
        for idx, player in enumerate(top_2.Name):
            
            # getting 'index' for player in 'df_teams' DF
            input_player_index = df_teams[df_teams['Name']==player].index.values[0]
            
            # getting the 'Overall', 'Potential', and 'Field Position Num'
            p_overall = df_teams.iloc[input_player_index]['Overall']
            p_potential = df_teams.iloc[input_player_index]['Potential']
            p_position = df_teams.iloc[input_player_index]['Field Position Num']
            
            # getting 'Wage' for player in 'df_teams' DF
            # to be used later for 'Post-trade Leftover Wage' in returned DF
            input_player_wage = df_teams.iloc[input_player_index]['Wage']
            players_wages.append(input_player_wage)
            
            # getting 'row' for same player in 'df_attributes' using index (No 'Name' col in 'df_attributes')
            player_attributes = df_attributes.iloc[input_player_index]
            
            # filtering attributes logic:
            filtered_attributes = df_attributes[(df_attributes['Overall'] > p_overall-10) 
                                            & (df_attributes['Potential'] > p_potential-10)
                                            & (df_attributes['Field Position Num'] == p_position)]

            # use filter logic to suggest replacement players - top 5 suggested
            # gives DF of with all indexes and correlation ratio
            suggested_players = filtered_attributes.corrwith(player_attributes, axis=1)
            
            # Top 2 suggested players (most positively correlated)
            suggested_players = suggested_players.sort_values(ascending=False).head(6)
            
            cols = ['Name', 'Position', 'Overvalue Ratio', 'Overall', 'Potential', 'Wage']
            for i, corr in enumerate(suggested_players):
                if idx == 0:
                    # player 1 - suggested trades
                    trades_p1.append(df_teams[df_teams.index==suggested_players.index[i]][cols].values)
                else:
                    # player 2 - suggested trades                 
                    trades_p2.append(df_teams[df_teams.index==suggested_players.index[i]][cols].values)

    cols1 = ['Name', 'Position', 'Overvalue Ratio', 'Overall Rating', 'Potential Rating', 'Wage']
    # suggested trades DF for player 1 - dropping 1st row (most positively correlated = same as player 1)
    trades_p1_df = pd.DataFrame(np.row_stack(trades_p1), columns=cols1)
    trades_p1_df = trades_p1_df.drop(trades_p1_df.index[0]).reset_index(drop=True)
    
    # suggested trades DF for player 2 - dropping 1st row (most positively correlated = same as player 2)
    trades_p2_df = pd.DataFrame(np.row_stack(trades_p2), columns=cols1)
    trades_p2_df = trades_p2_df.drop(trades_p2_df.index[0]).reset_index(drop=True)
    
    #adding 'Post-trade Leftover Wage' column to each returned DF
    trades_p1_df['Post-trade Leftover Wage'] = players_wages[0] - trades_p1_df['Wage']
    trades_p2_df['Post-trade Leftover Wage'] = players_wages[1] - trades_p2_df['Wage']
    
    return top_2, bottom_2, trades_p1_df, trades_p2_df


# In[23]:


# See comment line inside of function just below
def get_replacement_players(club):
    '''Gets 2 lowest-rated players, and suggests four possible replacements.'''
    replacements_p1 = []  # this will be the output object that club_suggested_changes receives/uses
    replacements_p2 = []
    players_wages = []

    if club in df_teams.Club.values or type(club) == list:  # either gets existing club from db, or uses custom list
        all_players, top_2, bottom_2 = get_club_players(club)  # df_club_list, df_top_2_players, df_bottom_2_players
        
        # looping throught 2 player names in 'top_2'
        for idx, player in enumerate(bottom_2.Name):
            
            # getting 'index' for player in 'df_teams' DF
            input_player_index = df_teams[df_teams['Name']==player].index.values[0]
            
            # getting the 'Overall', 'Potential', and 'Field Position Num'
            p_overall = df_teams.iloc[input_player_index]['Overall']
            p_potential = df_teams.iloc[input_player_index]['Potential']
            p_position = df_teams.iloc[input_player_index]['Field Position Num']
            
            # getting 'Wage' for player in 'df_teams' DF
            # to be used later for 'Post-trade Leftover Wage' in returned DF
            input_player_wage = df_teams.iloc[input_player_index]['Wage']
            players_wages.append(input_player_wage)
            
            # getting 'row' for same player in 'df_attributes' using index (No 'Name' col in 'df_attributes')
            player_attributes = df_attributes.iloc[input_player_index]
        
            # filtering weak attributes logic:
            filtered_weak_attributes = df_attributes[(df_attributes['Overall'] < 87) 
                                                     & (df_attributes['Potential'] > p_potential)
                                                     & (df_attributes['Potential'] < 89)
                                                     & (df_attributes['Field Position Num'] == p_position)]

            suggested_players = filtered_weak_attributes.corrwith(player_attributes, axis=1)
            suggested_players = suggested_players.sort_values(ascending=False).head(3)
            
            cols = ['Name', 'Position', 'Overvalue Ratio', 'Overall', 'Potential', 'Wage']
            for i, corr in enumerate(suggested_players):
                if idx == 0:
                    # player 1 - suggested replacements
                    replacements_p1.append(df_teams[df_teams.index==suggested_players.index[i]][cols].values)
                else:
                    # player 2 - suggested replacements                 
                    replacements_p2.append(df_teams[df_teams.index==suggested_players.index[i]][cols].values)

    cols1 = ['Name', 'Position', 'Overvalue Ratio', 'Overall Rating', 'Potential Rating', 'Wage']
    
    # suggested replacements DF for player 1 - dropping 1st row (most positively correlated = same as player 1)
    replacements_p1_df = pd.DataFrame(np.row_stack(replacements_p1), columns=cols1)
    replacements_p1_df = replacements_p1_df.drop(replacements_p1_df.index[0]).reset_index(drop=True)
    
    # suggested replacements DF for player 2 - dropping 1st row (most positively correlated = same as player 2)
    replacements_p2_df = pd.DataFrame(np.row_stack(replacements_p2), columns=cols1)
    replacements_p2_df = replacements_p2_df.drop(replacements_p2_df.index[0]).reset_index(drop=True)
    
    #adding 'Post-trade Leftover Wage' column to each returned DF
    replacements_p1_df['Post-trade Leftover Wage'] = players_wages[0] - replacements_p1_df['Wage']
    replacements_p2_df['Post-trade Leftover Wage'] = players_wages[1] - replacements_p2_df['Wage']
    
    #print(replacements_p1_df, '\n')
    #print(replacements_p2_df)
    
    return replacements_p1_df, replacements_p2_df


# In[24]:


# TODO! --- NEEDS 'club' VARIABLE DEFINED BEFORE UPCOMING CODE BLOCK


# In[ ]:


# JSON output
allplayers, top2overvalued, bottom2weak = get_club_players(club)
top_2, bottom_2, trades_p1_df, trades_p2_df = get_suggested_trades(club)
suggestedtrades = [trades_p1_df, trades_p2_df]
replacements_p1_df, replacements_p2_df = get_replacement_players(club)
suggestedreplacements = [replacements_p1_df, replacements_p2_df]


def all_dfs_json():
    json_dict = dict({'allplayers': [allplayers], 'top2overvalued': [top2overvalued],
                     'suggestedtrades': suggestedtrades, 'bottom2weak': [bottom_2],
                     'suggestedreplacements': suggestedreplacements}) 
    return json_dict


# In[ ]:


# TODO! --- NEED TO RUN THE FUNCTION BELOW!
# all_dfs_json()


# In[ ]:




