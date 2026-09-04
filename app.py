import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Social Engagement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.dashboard-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 10px;
}

.info-box {
    padding: 15px;
    border-radius: 12px;
    background-color: #eef5ff;
    border: 1px solid #dbeafe;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # First try project-relative paths
    possible_content_paths = [
        "data/content_data_viral.csv",
        "content_data_viral.csv",
        r"D:\content_data_viral.csv"
    ]

    possible_comment_paths = [
        "data/comments.csv",
        "comments.csv",
        r"D:\comments.csv"
    ]

    content_path = None
    comment_path = None

    for path in possible_content_paths:
        if os.path.exists(path):
            content_path = path
            break

    for path in possible_comment_paths:
        if os.path.exists(path):
            comment_path = path
            break

    if content_path is None:
        st.error(
            "Content dataset not found. "
            "Please place content_data_viral.csv in the project folder "
            "or inside a data folder."
        )
        st.stop()

    if comment_path is None:
        st.warning(
            "comments.csv was not found. "
            "The audience relatability section will be limited."
        )

        df = pd.read_csv(content_path)
        comments_df = pd.DataFrame(columns=["comment"])

    else:
        df = pd.read_csv(content_path)
        comments_df = pd.read_csv(comment_path)

    return df, comments_df


df, comments_df = load_data()


# ============================================================
# BASIC DATA CLEANING
# ============================================================

numeric_columns = [
    "likes",
    "comments",
    "shares",
    "saves",
    "retention_rate",
    "followers_gained"
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)


# ============================================================
# CREATE ENGAGEMENT SCORE
# ============================================================

required_engagement_columns = [
    "likes",
    "comments",
    "shares",
    "saves"
]

if all(
    column in df.columns
    for column in required_engagement_columns
):

    if "engagement_score" not in df.columns:

        df["engagement_score"] = (
            df["likes"]
            + (3 * df["comments"])
            + (5 * df["shares"])
            + (5 * df["saves"])
        )

else:

    st.error(
        "Your dataset must contain likes, comments, shares and saves."
    )
    st.stop()


# ============================================================
# CREATE VIRAL SCORE
# ============================================================

if "retention_rate" not in df.columns:

    df["retention_rate"] = 0


if "viral_score" not in df.columns:

    df["viral_score"] = (
        df["likes"]
        + (3 * df["comments"])
        + (5 * df["shares"])
        + (5 * df["saves"])
        + (2 * df["retention_rate"])
    )


# ============================================================
# SAVE-TO-SHARE RATIO
# ============================================================

df["save_to_share_ratio"] = np.where(
    df["shares"] > 0,
    df["saves"] / df["shares"],
    0
)


# ============================================================
# CREATE RELATABILITY
# ============================================================

if "relatability" not in comments_df.columns:

    if "comment" in comments_df.columns:

        comments_df["clean_comment"] = (
            comments_df["comment"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        relatable_words = [
            "me",
            "relate",
            "relatable",
            "felt",
            "feeling",
            "same",
            "real",
            "needed",
            "situation",
            "through",
            "exactly",
            "literally",
            "understand",
            "understood",
            "experience",
            "happened",
            "my life",
            "myself"
        ]

        def classify_comment(text):

            text = str(text).lower()

            words = text.split()

            for word in relatable_words:

                if " " in word:

                    if word in text:
                        return "Relatable"

                else:

                    if word in words:
                        return "Relatable"

            return "Neutral"


        comments_df["relatability"] = (
            comments_df["clean_comment"]
            .apply(classify_comment)
        )

    else:

        comments_df["relatability"] = "Neutral"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    '📊 Social Engagement Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'Data-driven analysis of content performance, virality, '
    'audience engagement and follower growth.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Dashboard Filters")

st.sidebar.markdown(
    "Use the filters below to explore the content data."
)


# ============================================================
# TOPIC FILTER
# ============================================================

if "topic" in df.columns:

    topics = sorted(
        df["topic"]
        .dropna()
        .astype(str)
        .unique()
    )

else:

    topics = []


selected_topics = st.sidebar.multiselect(
    "📌 Topic",
    topics,
    default=topics
)


# ============================================================
# CONTENT TYPE FILTER
# ============================================================

if "content_type" in df.columns:

    content_types = sorted(
        df["content_type"]
        .dropna()
        .astype(str)
        .unique()
    )

else:

    content_types = []


selected_content_types = st.sidebar.multiselect(
    "🎬 Content Type",
    content_types,
    default=content_types
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if "topic" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["topic"].isin(selected_topics)
    ]

if "content_type" in filtered_df.columns:

    filtered_df = filtered_df[
        filtered_df["content_type"].isin(selected_content_types)
    ]

filtered_df = filtered_df.copy()


# ============================================================
# NO DATA CHECK
# ============================================================

if filtered_df.empty:

    st.warning(
        "No data available for the selected filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_posts = len(filtered_df)

total_likes = filtered_df["likes"].sum()

total_shares = filtered_df["shares"].sum()

total_saves = filtered_df["saves"].sum()

total_comments = filtered_df["comments"].sum()

avg_retention = filtered_df["retention_rate"].mean()

total_followers = (
    filtered_df["followers_gained"].sum()
    if "followers_gained" in filtered_df.columns
    else 0
)

average_viral_score = (
    filtered_df["viral_score"].mean()
)

average_engagement_score = (
    filtered_df["engagement_score"].mean()
)


# ============================================================
# PERFORMANCE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">'
    '📈 Performance Overview'
    '</div>',
    unsafe_allow_html=True
)


c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "📝 Total Posts",
        f"{total_posts:,}"
    )

with c2:

    st.metric(
        "❤️ Total Likes",
        f"{total_likes:,.0f}"
    )

with c3:

    st.metric(
        "🔄 Total Shares",
        f"{total_shares:,.0f}"
    )


c4, c5, c6 = st.columns(3)

with c4:

    st.metric(
        "🔖 Total Saves",
        f"{total_saves:,.0f}"
    )

with c5:

    st.metric(
        "⏱️ Avg Retention",
        f"{avg_retention:.1f}%"
    )

with c6:

    st.metric(
        "👥 Followers Gained",
        f"{total_followers:,.0f}"
    )


st.divider()


# ============================================================
# VIRAL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🔥 Viral Performance by Topic'
    '</div>',
    unsafe_allow_html=True
)


if "topic" in filtered_df.columns:

    topic_data = (
        filtered_df
        .groupby("topic")["viral_score"]
        .mean()
        .sort_values(ascending=False)
    )

    if not topic_data.empty:

        fig1, ax1 = plt.subplots(
            figsize=(12, 5)
        )

        topic_data.plot(
            kind="bar",
            ax=ax1
        )

        ax1.set_xlabel("Topic")

        ax1.set_ylabel("Average Viral Score")

        ax1.set_title(
            "Average Viral Score by Topic"
        )

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(
            fig1,
            width="stretch"
        )

        plt.close(fig1)

else:

    topic_data = pd.Series(dtype=float)


# ============================================================
# CONTENT TYPE AND FOLLOWER GROWTH
# ============================================================

left, right = st.columns(2)


# ------------------------------------------------------------
# CONTENT TYPE
# ------------------------------------------------------------

with left:

    st.markdown(
        '<div class="section-title">'
        '🎬 Content Type Performance'
        '</div>',
        unsafe_allow_html=True
    )

    if "content_type" in filtered_df.columns:

        content_data = (
            filtered_df
            .groupby("content_type")["engagement_score"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            content_data
        )

    else:

        content_data = pd.Series(dtype=float)


# ------------------------------------------------------------
# FOLLOWER GROWTH
# ------------------------------------------------------------

with right:

    st.markdown(
        '<div class="section-title">'
        '👥 Follower Growth by Topic'
        '</div>',
        unsafe_allow_html=True
    )

    if (
        "topic" in filtered_df.columns
        and "followers_gained" in filtered_df.columns
    ):

        followers_topic = (
            filtered_df
            .groupby("topic")["followers_gained"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(
            followers_topic
        )


# ============================================================
# SHARES VS SAVES
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🚀 Shares vs Saves'
    '</div>',
    unsafe_allow_html=True
)


fig2, ax2 = plt.subplots(
    figsize=(10, 5)
)


ax2.scatter(
    filtered_df["shares"],
    filtered_df["saves"],
    alpha=0.6
)


ax2.set_xlabel(
    "Shares"
)

ax2.set_ylabel(
    "Saves"
)

ax2.set_title(
    "Relationship Between Shares and Saves"
)

plt.tight_layout()

st.pyplot(
    fig2,
    width="stretch"
)

plt.close(fig2)


# ============================================================
# SAVE-TO-SHARE RATIO
# ============================================================

st.markdown(
    '<div class="section-title">'
    '💾 Save-to-Share Ratio'
    '</div>',
    unsafe_allow_html=True
)


if "topic" in filtered_df.columns:

    save_share_topic = (
        filtered_df
        .groupby("topic")["save_to_share_ratio"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(
        save_share_topic
    )


average_save_share = (
    filtered_df["save_to_share_ratio"].mean()
)

st.metric(
    "Average Save-to-Share Ratio",
    f"{average_save_share:.2f}"
)


# ============================================================
# AUDIENCE RELATABILITY
# ============================================================

st.markdown(
    '<div class="section-title">'
    '❤️ Audience Relatability'
    '</div>',
    unsafe_allow_html=True
)


if not comments_df.empty:

    relatability_data = (
        comments_df["relatability"]
        .value_counts()
    )

    st.bar_chart(
        relatability_data
    )

    relatable_count = (
        comments_df["relatability"]
        .eq("Relatable")
        .sum()
    )

    total_comments_dataset = len(
        comments_df
    )

    if total_comments_dataset > 0:

        relatable_percentage = (
            relatable_count
            / total_comments_dataset
            * 100
        )

    else:

        relatable_percentage = 0

    st.metric(
        "💬 Relatable Comments",
        f"{relatable_percentage:.1f}%"
    )

else:

    st.info(
        "No comments dataset available."
    )


# ============================================================
# TOP VIRAL CONTENT
# ============================================================

st.markdown(
    '<div class="section-title">'
    '🏆 Top 10 Viral Content'
    '</div>',
    unsafe_allow_html=True
)


top_content = (
    filtered_df
    .sort_values(
        "viral_score",
        ascending=False
    )
    .head(10)
)


display_columns = [
    "content_id",
    "topic",
    "content_type",
    "likes",
    "shares",
    "saves",
    "comments",
    "retention_rate",
    "viral_score"
]


available_columns = [
    column
    for column in display_columns
    if column in top_content.columns
]


st.dataframe(
    top_content[available_columns],
    width="stretch",
    hide_index=True
)


# ============================================================
# BEST PERFORMERS
# ============================================================

best_col1, best_col2 = st.columns(2)


with best_col1:

    st.markdown(
        '<div class="section-title">'
        '⭐ Best Topic'
        '</div>',
        unsafe_allow_html=True
    )

    if not topic_data.empty:

        best_topic = topic_data.index[0]

        best_score = topic_data.iloc[0]

        st.success(
            f"**{best_topic}**\n\n"
            f"Average Viral Score: "
            f"**{best_score:.2f}**"
        )


with best_col2:

    st.markdown(
        '<div class="section-title">'
        '🎯 Best Content Type'
        '</div>',
        unsafe_allow_html=True
    )

    if not content_data.empty:

        best_type = content_data.index[0]

        best_type_score = content_data.iloc[0]

        st.success(
            f"**{best_type}**\n\n"
            f"Average Engagement Score: "
            f"**{best_type_score:.2f}**"
        )


# ============================================================
# ENGAGEMENT OPTIMIZATION RECOMMENDER
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🤖 Engagement Optimization Recommender'
    '</div>',
    unsafe_allow_html=True
)


if not filtered_df.empty:

    recommended_topic = None

    recommended_content_type = None

    best_retention_type = None


    if "topic" in filtered_df.columns:

        topic_scores = (
            filtered_df
            .groupby("topic")["viral_score"]
            .mean()
        )

        if not topic_scores.empty:

            recommended_topic = (
                topic_scores.idxmax()
            )


    if "content_type" in filtered_df.columns:

        type_scores = (
            filtered_df
            .groupby("content_type")[
                "engagement_score"
            ]
            .mean()
        )

        if not type_scores.empty:

            recommended_content_type = (
                type_scores.idxmax()
            )


        retention_scores = (
            filtered_df
            .groupby("content_type")[
                "retention_rate"
            ]
            .mean()
        )

        if not retention_scores.empty:

            best_retention_type = (
                retention_scores.idxmax()
            )


    rec1, rec2, rec3 = st.columns(3)


    with rec1:

        st.info(
            f"### 📌 Topic\n\n"
            f"**{recommended_topic if recommended_topic else 'N/A'}**"
        )


    with rec2:

        st.info(
            f"### 🎬 Content Type\n\n"
            f"**{recommended_content_type if recommended_content_type else 'N/A'}**"
        )


    with rec3:

        st.info(
            f"### ⏱️ Retention\n\n"
            f"**{best_retention_type if best_retention_type else 'N/A'}**"
        )


    st.success(
        "### 🎯 Recommended Strategy\n\n"
        f"Focus on **{recommended_topic}** content using "
        f"**{recommended_content_type}** format. "
        f"The recommendation is based on historical viral score, "
        f"engagement score and retention performance."
    )


# ============================================================
# A/B TESTING FRAMEWORK
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🧪 A/B Testing Framework'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "Compare different content types to identify which format "
    "produces stronger engagement."
)


if "content_type" in filtered_df.columns:

    ab_data = (
        filtered_df
        .groupby("content_type")["engagement_score"]
        .agg(
            ["mean", "count"]
        )
        .sort_values(
            "mean",
            ascending=False
        )
    )

    st.dataframe(
        ab_data.rename(
            columns={
                "mean": "Average Engagement",
                "count": "Number of Posts"
            }
        ),
        width="stretch"
    )


    if len(ab_data) >= 2:

        winner = ab_data.index[0]

        second = ab_data.index[1]

        winner_score = ab_data.iloc[0]["mean"]

        second_score = ab_data.iloc[1]["mean"]

        improvement = (
            (winner_score - second_score)
            / second_score
            * 100
            if second_score != 0
            else 0
        )

        st.success(
            f"🏆 **A/B Test Winner: {winner}**\n\n"
            f"Average engagement: **{winner_score:.2f}**\n\n"
            f"Compared with {second}, this represents "
            f"approximately **{improvement:.1f}%** higher "
            f"average engagement."
        )


# ============================================================
# VIRALITY PREDICTION MODEL
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '🤖 Virality Prediction Engine'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "A machine-learning model classifies content as "
    "**Viral** or **Not Viral** using available content metadata."
)


# ------------------------------------------------------------
# CREATE VIRAL LABEL
# ------------------------------------------------------------

model_df = df.copy()

median_viral_score = (
    model_df["viral_score"].median()
)

model_df["viral_label"] = np.where(
    model_df["viral_score"] >= median_viral_score,
    "Viral",
    "Not Viral"
)


# ------------------------------------------------------------
# AVAILABLE FEATURES
# ------------------------------------------------------------

candidate_features = [
    "topic",
    "content_type",
    "posting_time",
    "hook_type",
    "caption_style",
    "video_length",
    "content_length"
]


available_features = [
    feature
    for feature in candidate_features
    if feature in model_df.columns
]


if len(available_features) > 0:

    X = model_df[available_features].copy()

    y = model_df["viral_label"]


    # Convert categorical values to strings
    for column in X.columns:

        if X[column].dtype == "object":

            X[column] = (
                X[column]
                .fillna("Unknown")
                .astype(str)
            )

        else:

            X[column] = (
                pd.to_numeric(
                    X[column],
                    errors="coerce"
                )
                .fillna(0)
            )


    categorical_features = [
        column
        for column in available_features
        if X[column].dtype == "object"
    ]


    numerical_features = [
        column
        for column in available_features
        if column not in categorical_features
    ]


    transformers = []


    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        )


    if numerical_features:

        transformers.append(
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        )


    if transformers and len(y.unique()) >= 2:

        preprocessor = ColumnTransformer(
            transformers=transformers
        )


        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced"
                    )
                )
            ]
        )


        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y
                )
            )


            model.fit(
                X_train,
                y_train
            )


            predictions = model.predict(
                X_test
            )


            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                pos_label="Viral",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                pos_label="Viral",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                pos_label="Viral",
                zero_division=0
            )


            m1, m2, m3, m4 = st.columns(4)


            with m1:

                st.metric(
                    "Accuracy",
                    f"{accuracy * 100:.1f}%"
                )


            with m2:

                st.metric(
                    "Precision",
                    f"{precision * 100:.1f}%"
                )


            with m3:

                st.metric(
                    "Recall",
                    f"{recall * 100:.1f}%"
                )


            with m4:

                st.metric(
                    "F1 Score",
                    f"{f1 * 100:.1f}%"
                )


            st.write(
                "**Model Features:**",
                ", ".join(available_features)
            )


            # ------------------------------------------------
            # CONFUSION MATRIX
            # ------------------------------------------------

            cm = confusion_matrix(
                y_test,
                predictions,
                labels=[
                    "Not Viral",
                    "Viral"
                ]
            )


            cm_df = pd.DataFrame(
                cm,
                index=[
                    "Actual Not Viral",
                    "Actual Viral"
                ],
                columns=[
                    "Predicted Not Viral",
                    "Predicted Viral"
                ]
            )


            st.write("### Confusion Matrix")

            st.dataframe(
                cm_df,
                width="stretch"
            )


        except Exception as e:

            st.warning(
                f"Could not train the prediction model: {e}"
            )


    else:

        st.warning(
            "Not enough classes available for model training."
        )


else:

    st.info(
        "No suitable metadata columns were found for the "
        "machine-learning prediction model."
    )


# ============================================================
# TREND FORECASTING
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📈 Trend Forecasting'
    '</div>',
    unsafe_allow_html=True
)


st.write(
    "This section identifies topics whose content frequency "
    "is increasing over time."
)


# Find a possible date column

possible_date_columns = [
    "date",
    "post_date",
    "created_at",
    "upload_date",
    "published_date"
]


date_column = None

for column in possible_date_columns:

    if column in df.columns:

        date_column = column
        break


if date_column is not None and "topic" in df.columns:

    trend_df = df.copy()

    trend_df[date_column] = pd.to_datetime(
        trend_df[date_column],
        errors="coerce"
    )

    trend_df = trend_df.dropna(
        subset=[date_column]
    )


    if not trend_df.empty:

        trend_df["week"] = (
            trend_df[date_column]
            .dt.to_period("W")
            .astype(str)
        )


        trend_data = (
            trend_df
            .groupby(
                ["week", "topic"]
            )
            .size()
            .reset_index(
                name="content_count"
            )
        )


        st.dataframe(
            trend_data.tail(20),
            width="stretch",
            hide_index=True
        )


        latest_week = (
            trend_data["week"].max()
        )


        latest_data = trend_data[
            trend_data["week"] == latest_week
        ]


        if not latest_data.empty:

            trending_topic = (
                latest_data
                .sort_values(
                    "content_count",
                    ascending=False
                )
                .iloc[0]["topic"]
            )


            trending_count = (
                latest_data
                .sort_values(
                    "content_count",
                    ascending=False
                )
                .iloc[0]["content_count"]
            )


            st.success(
                f"🔥 **Current Leading Topic: "
                f"{trending_topic}**\n\n"
                f"Content count in the latest period: "
                f"**{trending_count}**"
            )


else:

    st.info(
        "Trend forecasting requires a date column such as "
        "'date', 'post_date' or 'published_date' in the dataset."
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    '📋 Dataset Summary'
    '</div>',
    unsafe_allow_html=True
)


summary_col1, summary_col2, summary_col3 = st.columns(3)


with summary_col1:

    st.metric(
        "Content Records",
        f"{len(df):,}"
    )


with summary_col2:

    st.metric(
        "Comment Records",
        f"{len(comments_df):,}"
    )


with summary_col3:

    st.metric(
        "Average Viral Score",
        f"{average_viral_score:,.2f}"
    )


# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

st.markdown(
    '<div class="section-title">'
    '⬇️ Export Analysis Data'
    '</div>',
    unsafe_allow_html=True
)


csv_data = filtered_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Filtered Dataset",
    data=csv_data,
    file_name="filtered_social_engagement_data.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Data-Driven Social Engagement Analysis • "
    "Python • Pandas • Matplotlib • Scikit-learn • Streamlit"
)