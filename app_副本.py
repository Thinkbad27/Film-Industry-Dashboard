import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
import ast
import dash_bootstrap_components as dbc

df = pd.read_csv('df_cleaned.csv')

df_geo_cleaned = pd.read_csv('cleaned_geomap.csv')

df_trend = pd.read_csv('multiTimeline.csv')

# Extracting relevant columns for analysis
actor_revenue_df = df[['cast', 'revenue']]
historical_top_8 = df.sort_values(by='revenue', ascending=False).head(8)
# Define a function to extract actor names and create a new column
import ast

def get_main_actors(cast_data):
    # Parse the JSON-like string to extract main actors (top 3)
    try:
        cast_list = ast.literal_eval(cast_data)
        main_actors = [member['name'] for member in cast_list[:3]]
        return main_actors
    except:
        return []

# Create a new column for main actors
actor_revenue_df['main_actors'] = actor_revenue_df['cast'].apply(get_main_actors)

# Explode the main_actors list to have one actor per row
actor_revenue_exploded = actor_revenue_df.explode('main_actors')

# Group by actor and sum their revenue
actor_cumulative_revenue = actor_revenue_exploded.groupby('main_actors')['revenue'].sum().reset_index()

# Sort actors by cumulative revenue and select the top 8
top_8_actors = actor_cumulative_revenue.sort_values(by='revenue', ascending=False).head(8)
# Process the data based on your previous filtering and processing logic
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Filter for movies released between 2016-06-01 and 2016-09-30
filtered_movies = df[(df['release_date'] >= '2016-06-01') & (df['release_date'] <= '2016-09-30')]

# Sort the filtered movies by revenue in descending order and select the top 8
top_8_movies = filtered_movies.sort_values(by='revenue', ascending=False).head(8)

# Extract relevant columns for displaying ratings
movies = top_8_movies[['title_x', 'vote_average', 'vote_count']].copy()
movies.columns = ['title', 'vote_average', 'vote_count']  # Rename for easier use

# Helper function: generate star ratings based on the vote_average
def generate_stars(vote_average):
    full_stars = math.floor(vote_average / 2)  # Full stars
    half_star = 1 if vote_average % 2 >= 1 else 0  # Half star
    empty_stars = 5 - full_stars - half_star  # Empty stars
    stars = '★' * full_stars + '☆' * empty_stars + ('½' if half_star else '')
    return html.Span([
        html.Span(stars, style={'color': '#FFD700', 'fontSize': '20px'}),  # Dark yellow stars
        html.Span(f' ({vote_average})', style={'fontSize': '18px', 'marginLeft': '5px'})  # Display rating
    ])

# Helper function: format vote count to K or M
def format_vote_count(vote_count):
    if vote_count >= 1000:
        return f"{vote_count/1000:.1f}k"
    return str(vote_count)

# Process the genres and production countries, and extract relevant data
df['genres_list'] = df['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)])
df['countries_list'] = df['production_countries'].apply(lambda x: [country['name'] for country in ast.literal_eval(x)])

# Extract release year & drop rows with missing values
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df.dropna(subset=['release_year'])

# Explode the genres and countries into individual rows
df_exploded_genres = df.explode('genres_list')
df_exploded_countries = df.explode('countries_list')




# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

app.layout = html.Div([
    # Navigation Bar
    html.Div([
        html.H3("Film Industry Analysis Dashboard", style={'text-align': 'center'}),
        html.Nav([
            html.Ul([
                html.Li(dcc.Link('Home', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Box Office', href='/boxoffice'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Score', href='/score'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Popularity', href='/popularity'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Genre', href='/genre'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Language', href='/language'), style={'display': 'inline', 'margin': '0 10px'})
            ], style={'list-style-type': 'none', 'padding': '0'}),
        ], style={'text-align': 'center', 'background-color': 'rgba(0, 0, 0, 0)'}),
    ], style={'padding': '10px', 'border-radius': '5px'}),

    # Main Content
    html.Div([
        # First Row: 1st Column with 2 Plots
        html.Div([
            html.H3("Movie Revenue"),
            dcc.RadioItems(
                id='filter-type',
                options=[
                    {'label': 'On Screen (2016/06/01 to 2016/09/30)', 'value': 'on_screen'},
                    {'label': 'Historical Top 8', 'value': 'historical'},
                    {'label': 'By Actor', 'value': 'by_actor'}
                ],
                value='on_screen',
                labelStyle={'display': 'inline-block'}
            ), 
            dcc.Graph(id='bar-chart', style={'height': '400px', 'width': '80%'}),  # Adjusted size

            # Second section: Movie Ratings Table
            html.H3("Real-time Film Score"),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Stars"),
                        html.Th("Film Title"),
                        html.Th("No. of graders")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(generate_stars(row['vote_average'])),
                        html.Td(row['title'], style={'padding-left': '30px'}),
                        html.Td(format_vote_count(row['vote_count']), style={'padding-left': '30px'})
                    ]) for _, row in movies.iterrows()
                ])
            ])
        ], style={'flex': '1', 'padding': '10px'}), 

        # Second Column: 2nd Column with 2 Plots
        html.Div([
            html.H3("Movie Popularity"),
    
            # Dropdown for Time Series Plot
            html.Div([
                html.Label("Select Movies for Time Series:"),
                dcc.Dropdown(
                    id='time-series-dropdown',
                    options=[{'label': col, 'value': col} for col in df_trend.columns[1:]],
                    value=[df_trend.columns[1], df_trend.columns[2]],
                    multi=True,
                    style={'color': 'black'} 
                ),
            ]),
            dcc.Graph(id='time-series-plot', style={'height': '400px', 'width': '100%'}),  # Adjusted size
            
            # Dropdown for Geo Map Plot
            html.Div([
                html.Label("Select Movie for Geo Map:"),
                dcc.Dropdown(
                    id='geo-map-dropdown',
                    options=[{'label': col, 'value': col} for col in df_geo_cleaned.columns[1:]],
                    value=df_geo_cleaned.columns[1],
                    multi=False,
                    style={'color': 'black'}
                ),
            ]),
            dcc.Graph(id='geo-map', style={'height': '400px', 'width': '100%'}),  # Adjusted size
        ], style={'flex': '1', 'padding': '10px'}),

        # Third Column: 3rd Column with 2 Plots
        html.Div([
            html.H3("Film Industry Analysis"),
    
            # Treemap for movie genres
            dcc.Graph(id='treemap-graph', style={'height': '400px', 'width': '100%'}),  # Adjusted size
                
            # Donut chart for production countries
            dcc.Graph(id='donut-chart', style={'height': '500px', 'width': '100%'}),  # Adjusted size
                
            # Slider to select the year range
            dcc.RangeSlider(
                int(df['release_year'].min()),
                int(df['release_year'].max()),
                step=1,
                marks={str(year): str(year) for year in range(int(df['release_year'].min()), int(df['release_year'].max())+1, 5)},
                value=[int(df['release_year'].min()), int(df['release_year'].max())],
                id='year-slider'
            )
        ], style={'flex': '1', 'padding': '10px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
])


# Define callback function to update bar chart based on selected filter
@app.callback(
    Output('bar-chart', 'figure'),
    Input('filter-type', 'value')
)
def update_graph(selected_filter):
    # Check user selection and sort data accordingly
    if selected_filter == 'on_screen':
        # Sort movies by revenue for on-screen selection
        data = top_8_movies.sort_values(by='revenue', ascending=False)
        title = 'Top 8 Movies (On Screen - 2016/06/01 to 2016/09/30)'
        width, height = 900, 400  # Set size for on-screen chart
        left_margin = 180
        
        # Create the bar chart
        fig = px.bar(data, x='revenue', y='title_x', orientation='h', title=title)
        
    elif selected_filter == 'historical':
        # Sort for historical top 8 movies by revenue
        data = historical_top_8.sort_values(by='revenue', ascending=False)
        title = 'Top 8 Movies (Historical)'
        width, height = 700, 400  # Set size for historical chart
        left_margin = 160

        fig = px.bar(data, x='revenue', y='title_x', orientation='h', title=title)
        
    else:
        # Sort actors by cumulative revenue
        data = top_8_actors.sort_values(by='revenue', ascending=False)
        title = 'Top 8 Actors by Cumulative Revenue'
        width, height = 700, 400  # Set size for by_actor chart
        left_margin = 200

        # Create bar chart with color differences
        fig = px.bar(data, x='revenue', y='main_actors', orientation='h', title=title,
                     color='revenue', color_continuous_scale='Blues')

        # Adjust x-axis range
        fig.update_xaxes(range=[min(data['revenue']) * 0.9, max(data['revenue']) * 1.05])

    # Apply common layout settings
    fig.update_layout(
        title_x=0,  # Left-align title
        xaxis_title="Revenue", 
        yaxis_title="",
        yaxis_tickfont=dict(size=14),
        yaxis_ticklabelposition="outside",
        yaxis_tickangle=0,
        margin=dict(l=left_margin, r=50, t=80, b=50), 
        width=width,  
        height=height,  
        yaxis={'categoryorder':'total ascending', 'automargin': True}
    )
    
    # Adjust bar chart visuals
    fig.update_traces(marker_color='rgba(0, 0, 139, 0.7)', marker_line_width=1.5, width=0.5)  # Dark blue color

    return fig

# Callback to update Time Series Plot based on selected movies
@app.callback(
    Output('time-series-plot', 'figure'),
    Input('time-series-dropdown', 'value')
)
def update_time_series(selected_movies):
    # Check if any movies are selected
    if not selected_movies:
        return {}  # Return empty figure if no movies are selected

    # Time Series Plot
    try:
        df_melted_trend = df_trend.melt(id_vars='Months', value_vars=selected_movies, 
                                         var_name='Movie', value_name='Popularity')
        time_series_fig = px.line(
            df_melted_trend, 
            x='Months', 
            y='Popularity', 
            color='Movie', 
            title='Movie Popularity Over Time',
            labels={'Months': 'Months', 'Popularity': 'Popularity Index'}
        )

        # Update layout to improve readability
        time_series_fig.update_layout(
            xaxis_title='Months',
            yaxis_title='Popularity Index',
            showlegend=True,
            legend_title_text='Movies',
        )
    except Exception as e:
        print(f"Error generating time series plot: {e}")
        time_series_fig = {}

    return time_series_fig

# Callback to update Geo Map Plot based on selected movie
@app.callback(
    Output('geo-map', 'figure'),
    Input('geo-map-dropdown', 'value')
)
def update_geo_map(selected_movie):
    # Geo Map Plot
    try:
        df_melted_geo = df_geo_cleaned.melt(id_vars='Countries', value_vars=[selected_movie], 
                                             var_name='Movie', value_name='Popularity')

        geo_map_fig = px.choropleth(
            df_melted_geo,
            locations='Countries',
            locationmode='country names',
            color='Popularity',
            hover_name='Countries',
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f'{selected_movie} Popularity by Country'
        )

        # Update layout for the geo map
        geo_map_fig.update_geos(
            showland=True, 
            landcolor='lightgray', 
            showcoastlines=True,
            coastlinecolor='black',
        )
        geo_map_fig.update_layout(
            title=f'{selected_movie} Popularity by Country',
            coloraxis_colorbar=dict(title='Popularity Index'),
        )
    except Exception as e:
        print(f"Error generating geo map plot: {e}")
        geo_map_fig = {}

    return geo_map_fig

# Update the treemap and donut chart based on the selected year range
@app.callback(
    [Output('treemap-graph', 'figure'),
     Output('donut-chart', 'figure')],
    [Input('year-slider', 'value')]
)
def charts(year_range):
    # Filter the dataframe
    filtered_df_genres = df_exploded_genres[(df_exploded_genres['release_year'] >= year_range[0]) & 
                                            (df_exploded_genres['release_year'] <= year_range[1])]
    
    filtered_df_countries = df_exploded_countries[(df_exploded_countries['release_year'] >= year_range[0]) & 
                                                  (df_exploded_countries['release_year'] <= year_range[1])]

    # Count the frequency of genres
    filtered_genre_counts = filtered_df_genres['genres_list'].value_counts().reset_index()
    filtered_genre_counts.columns = ['genre', 'count']

    # Create the treemap
    treemap_fig = px.treemap(filtered_genre_counts, path=['genre'], values='count',
                             title=f"Market Share of Movie Genres from {year_range[0]} to {year_range[1]}",
                             color='genre', color_discrete_sequence=custom_colors)
    
    # Count the frequency of production countries
    filtered_country_counts = filtered_df_countries['countries_list'].value_counts().reset_index()
    filtered_country_counts.columns = ['country', 'count']
    
    # Create the donut chart (only use top 10 countries for concision)
    top_countries = filtered_country_counts.head(10)
    
    total_count = filtered_country_counts['count'].sum()
    
    donut_fig = go.Figure(data=[go.Pie(labels=top_countries['country'], values=top_countries['count'], 
                                       hole=.4, hoverinfo="label+percent+name")])

    donut_fig.update_traces(direction='clockwise', pull=[0.1]*len(top_countries),
                            rotation=90, showlegend=True)
    
    donut_fig.update_layout(
        title_text=f"Diversity of Production Countries from {year_range[0]} to {year_range[1]}",
        annotations=[dict(text=f'Total: {total_count}', x=0.5, y=0.5, font_size=20, showarrow=False)],
        height=400, legend=dict(x=1, y=0.5), showlegend=True
    )

    return treemap_fig, donut_fig


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)


