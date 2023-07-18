#!/usr/bin/env python
# coding: utf-8

# In[16]:


initial_probability={}  #Defining the Initial values and assigning them a placce.
initial_probability['happy']=0.1
initial_probability['sad']=0.9


# In[17]:


initial_probability


# In[18]:


import pandas as pd
import numpy as np

probability_array=np.array([[0.7,0.3],
                           [0.4,0.6]])
hidden_states=['happy','sad']
df = pd.DataFrame(probability_array, index=hidden_states, columns=hidden_states)


# In[19]:


df


# In[20]:


s_probability_array=np.array([[0.8,0.2],
                           [0.4,0.6]])
df_1 = pd.DataFrame(s_probability_array, index=['happy','sad'], columns=['sunny','rainy'])


# In[21]:


df_1


# In[22]:


sequence = ['sunny','rainy']


# In[24]:


def HMM(initial_probability, df, df_1, sequence):
    hidden_sequence=[]
    current_state={}
    back_track = {}
    j=0
    for i in hidden_states:
        current_state[i]=initial_probability[i]*df_1.loc[i,sequence[j]]
        back_track[i] =[] 
   
    
    previous_state=current_state
    
    j=1                                                     
    while(j<len(sequence)):
        current_state_temp={}
        for h in hidden_states:
                                                         
            probability=[]
            for p in previous_state:
                
                prob=previous_state[p]* df.loc[p,h]*df_1.loc[h,sequence[j]]
                
                probability.append(prob)
           
            current_state_temp[h]=max(probability)
            back_track[h].append(hidden_states[probability.index(max(probability))])
            
        
        previous_state=current_state_temp
        j=j+1
    current_state=current_state_temp
    final_hidden_state=max(current_state,key=current_state_temp.get)
    hidden_sequence=back_track[final_hidden_state]
    hidden_sequence.append(final_hidden_state)
    return hidden_sequence,current_state
    


# In[25]:


HMM(initial_probability,df,df_1,sequence)


# In[ ]:




