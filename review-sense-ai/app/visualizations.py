import plotly.express as px
def sentiment_donut(df):
    counts = df["sentiment_bucket"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(counts,names="Sentiment",values="Count",hole=0.6,height=300)
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10),showlegend=True)
    return fig
