import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


employment_df = pd.read_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\processed\employment_sa_clean.csv")
poverty_df = pd.read_csv(r"C:\Users\Admin\Desktop\GroupX_DataAnalysis\NDTA631-DataAnalysis-GroupX\data\processed\poverty_sa_clean.csv")

st.title("📊 South Africa Employment & Poverty Dashboard")
st.write("This dashboard displays the cleaned and analysed datasets using interactive visualisations.")

#Bar plotliejh6
st.header("Average comparison(Bar plot)")
ave_values = {
    "Employment Rate": employment_df.select_dtypes(include='number').mean().mean(),
    "Poverty Rate": poverty_df.select_dtypes(include='number').mean().mean()
}

ave_df = pd.DataFrame(list(ave_values.items()), columns=['Category', 'Value'])
fig, ax = plt.subplots()
sns.barplot(data=ave_df, x="Category", y="Value", hue="Category", palette="Set2", legend=False, ax=ax)
ax.set_title("Average Employment vs Poverty")
st.pyplot(fig)



st.subheader("Employment vs Poverty (Scatter Plot)")
fig, ax = plt.subplots()
sns.scatterplot(
    x=employment_df.select_dtypes(include='number').iloc[:, 0],
    y=poverty_df.select_dtypes(include='number').iloc[:, 0],
    color="blue", alpha=0.6, ax=ax
)
ax.set_xlabel("Employment Indicator")
ax.set_ylabel("Poverty Indicator")
ax.set_title("Scatter Plot: Employment vs Poverty")
st.pyplot(fig)

st.subheader("Distribution Comparison (Box Plot)")
fig, ax = plt.subplots()
sns.boxplot(data=[employment_df.select_dtypes(include='number').iloc[:, 0],
                  poverty_df.select_dtypes(include='number').iloc[:, 0]],
            ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Employment", "Poverty"])
ax.set_title("Box Plot: Employment vs Poverty Distribution")
st.pyplot(fig)


st.subheader("Distribution of Values (Histogram)")
fig, ax = plt.subplots()
sns.histplot(employment_df.select_dtypes(include='number').iloc[:, 0], color="green", kde=True, label="Employment", ax=ax)
sns.histplot(poverty_df.select_dtypes(include='number').iloc[:, 0], color="red", kde=True, label="Poverty", ax=ax)
ax.legend()
ax.set_title("Histogram: Employment vs Poverty Distribution")
st.pyplot(fig)


