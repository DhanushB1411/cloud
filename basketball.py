import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile

### WORKING CODE #######

st.title('NBA Player Stats Explorer')

st.markdown("""
This app performs simple web scraping of NBA player stats data!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
""")

st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2025))))

@st.cache_data
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats

playerstats = load_data(selected_year)

sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

# Clean player names by stripping any trailing special characters
df_selected_team['Player'] = df_selected_team['Player'].str.replace('*', '', regex=False)

# Ensure numeric columns are correctly processed
df_selected_team['PTS'] = pd.to_numeric(df_selected_team['PTS'], errors='coerce')
df_selected_team['AST'] = pd.to_numeric(df_selected_team['AST'], errors='coerce')

# Check if 'REB' exists and process accordingly
if 'REB' in df_selected_team.columns:
    df_selected_team['REB'] = pd.to_numeric(df_selected_team['REB'], errors='coerce')

# Check if 'FG%' and '3P%' columns exist and process accordingly
if 'FG%' in df_selected_team.columns:
    df_selected_team['FG%'] = pd.to_numeric(df_selected_team['FG%'].str.rstrip('%').astype(float) / 100, errors='coerce')

if '3P%' in df_selected_team.columns:
    df_selected_team['3P%'] = pd.to_numeric(df_selected_team['3P%'].str.rstrip('%').astype(float) / 100, errors='coerce')

# Ensure 'FT%' and 'TOV' are processed
if 'FT%' in df_selected_team.columns:
    df_selected_team['FT%'] = pd.to_numeric(df_selected_team['FT%'].str.rstrip('%').astype(float) / 100, errors='coerce')

if 'TOV' in df_selected_team.columns:
    df_selected_team['TOV'] = pd.to_numeric(df_selected_team['TOV'], errors='coerce')

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# Show KPIs
st.subheader('Top 5 Players by Points Per Game')
top_points = df_selected_team[['Player', 'PTS']].sort_values(by='PTS', ascending=False).head(5)
st.write(top_points)

st.subheader('Top 5 Players by Assists Per Game')
top_assists = df_selected_team[['Player', 'AST']].sort_values(by='AST', ascending=False).head(5)
st.write(top_assists)

if 'REB' in df_selected_team.columns:
    st.subheader('Top 5 Players by Rebounds Per Game')
    top_rebounds = df_selected_team[['Player', 'REB']].sort_values(by='REB', ascending=False).head(5)
    st.write(top_rebounds)

    # Histograms for key metrics
    st.subheader('Distribution of Points Per Game')
    fig, ax = plt.subplots()
    sns.histplot(df_selected_team['PTS'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title('Points Per Game Distribution')
    st.pyplot(fig)

    st.subheader('Distribution of Assists Per Game')
    fig, ax = plt.subplots()
    sns.histplot(df_selected_team['AST'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title('Assists Per Game Distribution')
    st.pyplot(fig)

    st.subheader('Distribution of Rebounds Per Game')
    fig, ax = plt.subplots()
    sns.histplot(df_selected_team['REB'].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title('Rebounds Per Game Distribution')
    st.pyplot(fig)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    
    # Select only numeric columns for correlation calculation
    numeric_cols = df_selected_team.select_dtypes(include=np.number)
    
    # Calculate the correlation matrix
    corr = numeric_cols.corr()
    
    # Plot heatmap
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot()

# Function to calculate KPIs (from mapreduce_job.py)
def calculate_kpis(df):
    kpis = {}
    
    # Ensure these columns exist in the DataFrame
    if 'PER' in df.columns:
        kpis['PER'] = df['PER'].mean()
    if 'TS%' in df.columns:
        kpis['TS%'] = df['TS%'].mean()
    if 'eFG%' in df.columns:
        kpis['eFG%'] = df['eFG%'].mean()
    if 'AST' in df.columns and 'TOV' in df.columns:
        kpis['AST/TO'] = (df['AST'] / df['TOV']).mean()
    if 'USG%' in df.columns:
        kpis['USG%'] = df['USG%'].mean()
    
    return pd.DataFrame([kpis])

if st.button('Run MapReduce Job'):
    st.header('MapReduce Results')

    # Save the selected data to a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        df_selected_team.to_csv(temp_file.name, index=False)
        input_csv = temp_file.name
    
    # Run the MapReduce job
    df = pd.read_csv(input_csv)
    kpis = calculate_kpis(df)

    # Display KPI values
    if kpis.empty:
        st.error("The MapReduce job did not generate any results.")
    else:
        st.subheader('Essential KPI Values from MapReduce Job')
        st.write(kpis)

        if 'PER' in kpis.columns:
            st.write(f"Average Player Efficiency Rating (PER): {kpis['PER'].iloc[0]:.2f}")
        
        if 'TS%' in kpis.columns:
            st.write(f"Average True Shooting Percentage (TS%): {kpis['TS%'].iloc[0]:.2f}")
        
        if 'eFG%' in kpis.columns:
            st.write(f"Average Effective Field Goal Percentage (eFG%): {kpis['eFG%'].iloc[0]:.2f}")
        
        if 'AST/TO' in kpis.columns:
            st.write(f"Average Assist-to-Turnover Ratio (AST/TO): {kpis['AST/TO'].iloc[0]:.2f}")
        
        if 'USG%' in kpis.columns:
            st.write(f"Average Usage Rate (USG%): {kpis['USG%'].iloc[0]:.2f}")
