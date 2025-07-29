import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
import ast
import dash_bootstrap_components as dbc

# --- 数据加载和预处理 ---
# 确保这些 CSV 文件与 app.py 在同一个目录
try:
    df = pd.read_csv('df_cleaned.csv')
    df_geo_cleaned = pd.read_csv('cleaned_geomap.csv')
    df_trend = pd.read_csv('multiTimeline.csv')
    print("All CSV files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading CSV file: {e}. Please ensure CSVs are in the root directory.")
    # 如果文件没找到，这里可以创建空的DataFrame来避免后续代码报错
    df = pd.DataFrame()
    df_geo_cleaned = pd.DataFrame()
    df_trend = pd.DataFrame()


# Extracting relevant columns for analysis
# FIX: Addressing SettingWithCopyWarning by explicitly creating a copy
actor_revenue_df = df[['cast', 'revenue']].copy() 

# Define a function to extract actor names and create a new column
def get_main_actors(cast_data):
    # Parse the JSON-like string to extract main actors (top 3)
    try:
        # Added check for NaN and empty strings before literal_eval
        if pd.isna(cast_data) or cast_data.strip() == '':
            return []
        cast_list = ast.literal_eval(cast_data)
        main_actors = [member['name'] for member in cast_list[:3]]
        return main_actors
    except (ValueError, SyntaxError):
        # print(f"Warning: Could not parse cast data: {cast_data}") # 可以取消注释用于调试
        return []

# Create a new column for main actors
if not actor_revenue_df.empty and 'cast' in actor_revenue_df.columns:
    actor_revenue_df['main_actors'] = actor_revenue_df['cast'].apply(get_main_actors)
else:
    actor_revenue_df['main_actors'] = [[]] * len(actor_revenue_df) # 添加空列以防报错

# Explode the main_actors list to have one actor per row
# 仅在 main_actors 列存在且非空时执行 explode
if 'main_actors' in actor_revenue_df.columns and not actor_revenue_df['main_actors'].empty:
    actor_revenue_exploded = actor_revenue_df.explode('main_actors')
else:
    actor_revenue_exploded = pd.DataFrame(columns=actor_revenue_df.columns.tolist() + ['main_actors'])


# Group by actor and sum their revenue
if not actor_revenue_exploded.empty:
    actor_cumulative_revenue = actor_revenue_exploded.groupby('main_actors')['revenue'].sum().reset_index()
    # Sort actors by cumulative revenue and select the top 8
    top_8_actors = actor_cumulative_revenue.sort_values(by='revenue', ascending=False).head(8)
else:
    actor_cumulative_revenue = pd.DataFrame(columns=['main_actors', 'revenue'])
    top_8_actors = pd.DataFrame(columns=['main_actors', 'revenue'])


# Process the data based on your previous filtering and processing logic
# **修正：确保 historical_top_8 在全局范围被定义**
historical_top_8 = pd.DataFrame() # 初始化为空DataFrame

if not df.empty:
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

    # Filter for movies released between 2016-06-01 and 2016-09-30
    filtered_movies = df[(df['release_date'] >= '2016-06-01') & (df['release_date'] <= '2016-09-30')]

    # Sort the filtered movies by revenue in descending order and select the top 8
    top_8_movies = filtered_movies.sort_values(by='revenue', ascending=False).head(8)
    
    # 定义 historical_top_8（如果数据可用）
    # 假设 historical_top_8 也是从 df 派生，这里只是一个示例
    # 请根据你实际的历史数据逻辑来定义 historical_top_8
    # 比如：取 df 中总收入最高的8部电影
    historical_top_8 = df.sort_values(by='revenue', ascending=False).head(8)

    # Extract relevant columns for displaying ratings
    movies = top_8_movies[['title_x', 'vote_average', 'vote_count']].copy()
    movies.columns = ['title', 'vote_average', 'vote_count']  # Rename for easier use
else:
    top_8_movies = pd.DataFrame()
    movies = pd.DataFrame(columns=['title', 'vote_average', 'vote_count'])


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
if not df.empty and 'genres' in df.columns and 'production_countries' in df.columns:
    df['genres_list'] = df['genres'].apply(lambda x: [genre['name'] for genre in ast.literal_eval(x)] if pd.notna(x) and x.strip() != '' else [])
    df['countries_list'] = df['production_countries'].apply(lambda x: [country['name'] for country in ast.literal_eval(x)] if pd.notna(x) and x.strip() != '' else [])
else: # Handle case where df is empty or columns are missing
    df['genres_list'] = [[]] * len(df)
    df['countries_list'] = [[]] * len(df)


# Extract release year & drop rows with missing values
if not df.empty and 'release_date' in df.columns:
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df = df.dropna(subset=['release_year'])
else:
    df['release_year'] = pd.Series(dtype='float64')


# Explode the genres and countries into individual rows
# 仅在 'genres_list' 和 'countries_list' 列存在且非空时执行 explode
if 'genres_list' in df.columns and not df['genres_list'].empty:
    df_exploded_genres = df.explode('genres_list')
else:
    df_exploded_genres = pd.DataFrame(columns=df.columns.tolist() + ['genres_list'])

if 'countries_list' in df.columns and not df['countries_list'].empty:
    df_exploded_countries = df.explode('countries_list')
else:
    df_exploded_countries = pd.DataFrame(columns=df.columns.tolist() + ['countries_list'])


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])

custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# --- Dash App Layout (回到原始单页布局) ---
app.layout = html.Div([
    # Navigation Bar
    html.Div([
        html.H3("Film Industry Analysis Dashboard", style={'text-align': 'center'}),
        html.Nav([
            html.Ul([
                # 即使现在是单页应用，链接也可以保留，但它们会刷新页面到根路径
                html.Li(dcc.Link('Home', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Box Office', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Score', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Popularity', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Genre', href='/'), style={'display': 'inline', 'margin': '0 10px'}),
                html.Li(dcc.Link('Language', href='/'), style={'display': 'inline', 'margin': '0 10px'})
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
            dcc.Graph(id='bar-chart', style={'height': '400px', 'width': '80%'}),

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
                    options=[{'label': col, 'value': col} for col in df_trend.columns[1:]] if not df_trend.empty else [],
                    value=[df_trend.columns[1], df_trend.columns[2]] if not df_trend.empty and len(df_trend.columns) > 2 else [],
                    multi=True,
                    style={'color': 'black'}
                ),
            ]),
            dcc.Graph(id='time-series-plot', style={'height': '400px', 'width': '100%'}),

            # Dropdown for Geo Map Plot
            html.Div([
                html.Label("Select Movie for Geo Map:"),
                dcc.Dropdown(
                    id='geo-map-dropdown',
                    options=[{'label': col, 'value': col} for col in df_geo_cleaned.columns[1:]] if not df_geo_cleaned.empty else [],
                    value=df_geo_cleaned.columns[1] if not df_geo_cleaned.empty and len(df_geo_cleaned.columns) > 1 else None,
                    multi=False,
                    style={'color': 'black'}
                ),
            ]),
            dcc.Graph(id='geo-map', style={'height': '400px', 'width': '100%'}),
        ], style={'flex': '1', 'padding': '10px'}),

        # Third Column: 3rd Column with 2 Plots
        html.Div([
            html.H3("Film Industry Analysis"),

            # Treemap for movie genres
            dcc.Graph(id='treemap-graph', style={'height': '400px', 'width': '100%'}),

            # Donut chart for production countries
            dcc.Graph(id='donut-chart', style={'height': '500px', 'width': '100%'}),

            # Slider to select the year range
            dcc.RangeSlider(
                min=int(df['release_year'].min()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].min()) else 1900,
                max=int(df['release_year'].max()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].max()) else 2024,
                step=1,
                # 修正：将 '5' 放到 range 函数的括号内部
                marks={str(year): str(year) for year in range(
                    int(df['release_year'].min()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].min()) else 1900,
                    int(df['release_year'].max()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].max()) else 2024 + 1,
                    5
                )},
                value=[int(df['release_year'].min()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].min()) else 1900,
                       int(df['release_year'].max()) if not df.empty and 'release_year' in df.columns and not df['release_year'].empty and pd.notna(df['release_year'].max()) else 2024],
                id='year-slider'
            )
        ], style={'flex': '1', 'padding': '10px'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
])


# --- 回调函数 ---

# Define callback function to update bar chart based on selected filter
@app.callback(
    Output('bar-chart', 'figure'),
    Input('filter-type', 'value')
)
def update_graph(selected_filter):
    # 修正：确保所有引用的全局变量都已定义，并且进行空值检查
    # 这里需要确保 historical_top_8 在全局作用域已经初始化
    global historical_top_8 # 声明使用全局变量
    if historical_top_8.empty and not df.empty: # 如果 historical_top_8 还是空的，尝试重新创建它
        historical_top_8 = df.sort_values(by='revenue', ascending=False).head(8)


    if df.empty or top_8_movies.empty or historical_top_8.empty or top_8_actors.empty:
        return go.Figure().update_layout(title="Data not loaded or insufficient for this chart.")

    # Check user selection and sort data accordingly
    if selected_filter == 'on_screen':
        # Sort movies by revenue for on-screen selection
        data = top_8_movies.sort_values(by='revenue', ascending=False)
        title = 'Top 8 Movies (On Screen - 2016/06/01 to 2016/09/30)'
        width, height = 900, 400  # Set size for on-screen chart
        left_margin = 180

        # Create the bar chart - 修正了 y 轴的列名
        fig = px.bar(data, x='revenue', y='title_x', orientation='h', title=title) # 确保 y 轴是 title_x

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
        if not data.empty and 'revenue' in data.columns:
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
    if df_trend.empty or not selected_movies:
        return go.Figure().update_layout(title="Time Series Data not available or no movies selected.")

    # Time Series Plot
    try:
        # Filter selected_movies to only include columns that actually exist in df_trend
        valid_selected_movies = [col for col in selected_movies if col in df_trend.columns]
        if not valid_selected_movies:
            return go.Figure().update_layout(title="No valid movies selected for Time Series.")

        # Ensure 'Months' column is always used as x-axis
        if 'Months' not in df_trend.columns:
            return go.Figure().update_layout(title="Error: 'Months' column not found in trend data.")

        df_melted_trend = df_trend.melt(id_vars='Months', value_vars=valid_selected_movies,
                                         var_name='Movie', value_name='Popularity')
        
        # 修正：处理 melt 后可能为空的情况
        if df_melted_trend.empty:
            return go.Figure().update_layout(title="No data to display for selected time series movies.")

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
        time_series_fig = go.Figure().update_layout(title=f"Error loading Time Series: {e}")

    return time_series_fig

# Callback to update Geo Map Plot based on selected movie
@app.callback(
    Output('geo-map', 'figure'),
    Input('geo-map-dropdown', 'value')
)
def update_geo_map(selected_movie):
    if df_geo_cleaned.empty or not selected_movie:
        return go.Figure().update_layout(title="Geo Map Data not available or no movie selected.")

    # Geo Map Plot
    try:
        if selected_movie not in df_geo_cleaned.columns:
             return go.Figure().update_layout(title=f"Movie '{selected_movie}' not found in Geo Map data.")

        df_melted_geo = df_geo_cleaned.melt(id_vars='Countries', value_vars=[selected_movie],
                                            var_name='Movie', value_name='Popularity')
        
        # 修正：处理 melt 后可能为空的情况
        if df_melted_geo.empty:
            return go.Figure().update_layout(title="No data to display for selected geo map movie.")

        geo_map_fig = px.choropleth(
            df_melted_geo,
            locations='Countries',
            locationmode='country names',
            color='Popularity',
            hover_name='Countries',
            color_continuous_scale=px.colors.sequential.Viridis,
            title=f'{selected_movie} Popularity by Country'
        )

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
        geo_map_fig = go.Figure().update_layout(title=f"Error loading Geo Map: {e}")

    return geo_map_fig

# Update the treemap and donut chart based on the selected year range
@app.callback(
    [Output('treemap-graph', 'figure'),
     Output('donut-chart', 'figure')],
    [Input('year-slider', 'value')]
)
def charts(year_range):
    # **修正：确保 filtered_df_genres 和 filtered_df_countries 始终被初始化**
    filtered_df_genres = pd.DataFrame()
    filtered_df_countries = pd.DataFrame()

    if df_exploded_genres.empty or df_exploded_countries.empty:
        return go.Figure().update_layout(title="Genre/Country Data not available or insufficient for this chart."), \
               go.Figure().update_layout(title="Genre/Country Data not available or insufficient for this chart.")

    # Filter the dataframe
    if not df_exploded_genres.empty and 'release_year' in df_exploded_genres.columns:
        filtered_df_genres = df_exploded_genres[(df_exploded_genres['release_year'] >= year_range[0]) &
                                                (df_exploded_genres['release_year'] <= year_range[1])]

    if not df_exploded_countries.empty and 'release_year' in df_exploded_countries.columns:
        filtered_df_countries = df_exploded_countries[(filtered_df_countries['release_year'] >= year_range[0]) &
                                                      (filtered_df_countries['release_year'] <= year_range[1])]

    # 检查过滤后的数据是否为空
    if filtered_df_genres.empty:
        treemap_fig = go.Figure().update_layout(title="No genre data for selected year range.")
    else:
        # Count the frequency of genres
        filtered_genre_counts = filtered_df_genres['genres_list'].value_counts().reset_index()
        filtered_genre_counts.columns = ['genre', 'count']

        # Create the treemap
        treemap_fig = px.treemap(filtered_genre_counts, path=['genre'], values='count',
                                 title=f"Market Share of Movie Genres from {year_range[0]} to {year_range[1]}",
                                 color='genre', color_discrete_sequence=custom_colors)

    if filtered_df_countries.empty:
        donut_fig = go.Figure().update_layout(title="No country data for selected year range.")
    else:
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

# --- 供 Gunicorn 调用的 WSGI 应用入口 ---
# 确保这个名字是 'server'，因为它在 Procfile 中被引用
server = app.server

# Run the Dash app (仅用于本地调试)
if __name__ == '__main__':
    app.run_server(debug=True)
