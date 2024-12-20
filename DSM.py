import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data=pd.read_csv(r'c:\Users\Mahmoud\Downloads\archive(1)\scanner_data.csv',encoding='ISO-8859-1')

#----
#info
#----
print(data.describe())
print(data.info())
print(data.nunique())







#--------
#cleaning
#--------


pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Disable line wrapping, adjust to your screen width


df=pd.DataFrame(data)
print(df.head()) # cuz this alone without the above codes displayed only first and last column and dots in betweeen
print(df.shape)
print(df.info())


# Convert the 'price' column to numeric to change any  non-numeric values to null
df['Sales_Amount'] = pd.to_numeric(df['Sales_Amount'], errors='coerce')


# Drop rows where 'price' is null
df = df.dropna(subset=['Sales_Amount'])


# remove null values
df = df.dropna(subset=['Customer_ID'])
# df["Customer_ID"] = df["Customer_ID"].astype('int64')
print(df.describe())


print(df.duplicated().any())
df.drop_duplicates(inplace=True)


print(df.describe())
print(df.shape)
print(df.nunique())








#-------------
#visualization
#-------------




# aggregate columns 
daily_sales = data.groupby('Date').sum(numeric_only=True)['Sales_Amount']
category_sales = data.groupby('SKU_Category')['Sales_Amount'].sum().sort_values(ascending=False)
top_skus = data.groupby('SKU')['Sales_Amount'].sum().nlargest(10)
customer_transactions = data.groupby('Customer_ID').size()
customer_sales = data.groupby('Customer_ID')['Sales_Amount'].sum()

#sales over time
plt.figure(figsize=(12, 6))
daily_sales.plot()
plt.title('Daily Sales Trends')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid()
plt.show()

# product  analysis
plt.figure(figsize=(10, 6))
category_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by SKU Category')
plt.xlabel('SKU Category')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# top products
plt.figure(figsize=(10, 6))
top_skus.plot(kind='bar', color='green')
plt.title('Top 10 Selling SKUs by Sales Amount')
plt.xlabel('SKU')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

#customer transactions
plt.figure(figsize=(10, 6))
sns.histplot(customer_transactions, bins=30, kde=True, color='orange')
plt.title('Distribution of Transactions Per Customer')
plt.xlabel('Number of Transactions')
plt.ylabel('Frequency')
plt.grid()
plt.show()

#qunatity and sales amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Quantity'], y=data['Sales_Amount'], alpha=0.6)
plt.title('Sales Amount vs. Quantity Sold')
plt.xlabel('Quantity')
plt.ylabel('Sales Amount')
plt.grid()
plt.show()

#sales distribution
plt.figure(figsize=(10, 6))
sns.boxplot(y=data['Sales_Amount'], color='purple')
plt.title('Distribution of Sales Amounts Per Transaction')
plt.ylabel('Sales Amount')
plt.grid(axis='x')
plt.show()

#customer sales 
plt.figure(figsize=(10, 6))
sns.histplot(customer_sales, bins=30, kde=True, color='red')
plt.title('Distribution of Total Sales Per Customer')
plt.xlabel('Total Sales Amount')
plt.ylabel('Frequency')
plt.grid()
plt.show()









#------------------
#analysis using rfm
#------------------






df=pd.DataFrame(data)

print(df.head())

print(df.shape)

print(df.info())


# Drop rows where the 'CustomerID' is missing (NaN values)
df = df.dropna(subset=["Customer_ID"])

# Convert the 'CustomerID' column from float (or object) to integer type
df['Customer_ID'] = df['Customer_ID'].astype('int64')

#statistical summary of numerical data
print(df.describe())


#removing quantities less than 0
df = df[df["Quantity"] > 0]

df.drop_duplicates(inplace=True)


print(df.shape)

print(df.nunique())


df = df[["Customer_ID","Transaction_ID","Date","Quantity","Sales_Amount"]]

df["TotalPrice"] = df["Quantity"] * df["Sales_Amount"] 

df["Date"] = pd.to_datetime(df['Date'], errors='coerce')



print(df.head())


present_time = datetime.now()

rfm = df.groupby("Customer_ID").agg({"Date":lambda date : (present_time - date.max()).days,
                                  "Transaction_ID": lambda num : len(num),
                                  "TotalPrice" : lambda price : price.sum()})


print(rfm.head())




rfm.columns = ["Recency", "Frequency", "Monetary"]

print(rfm.info())

rfm=rfm.dropna()

# create quartiles for Recency
rfm['r_quartile'] = pd.qcut(rfm['Recency'], 4, labels=['1', '2', '3', '4'])

# create quartiles for Frequency, assigning higher values to more frequent customers (1=most frequent)
rfm['f_quartile'] = pd.qcut(rfm['Frequency'], 4, labels=['4', '3', '2', '1'], duplicates='drop')

# create quartiles for Monetary, assigning higher values to higher spenders (1=highest spending)
rfm['m_quartile'] = pd.qcut(rfm['Monetary'], 4, labels=['4', '3', '2', '1'], duplicates='drop')


print(rfm.head())



rfm["RFM_Score"] = rfm.r_quartile.astype(str) + rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)
print(rfm.head())



#sorted rfm scores
print(rfm[rfm['RFM_Score']=='111'].sort_values('Monetary',ascending=False).head())




# Create RFM segments based on the RFM score
rfm["RFM_Score"] = rfm['RFM_Score'].astype(int)
segment_labels = ['High-Value', 'Mid-Value','Low-Value']
rfm['Value_Segment'] = pd.qcut(rfm['RFM_Score'], q=3, labels=segment_labels)
print(rfm.head())




plt.figure(figsize=(10, 6))  # Set the figure size
ax = rfm.Value_Segment.value_counts().sort_values().plot(kind='bar', color=sns.color_palette("pastel"))

# Customize the plot
ax.set_title('Value Segment Distribution', fontsize=16)  # Add a title with font size
ax.set_xlabel('Value Segment', fontsize=14)              # Add x-label with font size
ax.set_ylabel('Count', fontsize=14)                       # Add y-label with font size
ax.grid(axis='y', linestyle='--', alpha=0.7)             # Add gridlines for better readability

# Show value annotations on top of the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

# Show the plot
plt.show()





# Select relevant columns for correlation
columns_to_plot = ['Recency', 'Frequency', 'Monetary', 'RFM_Score']
correlation_matrix = rfm[columns_to_plot].corr()

# Set the size of the plot
plt.figure(figsize=(10, 6))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, 
            cmap='coolwarm',  # Color map
            square=True,  # Make cells square-shaped
            linewidths=0.5,  # Width of the lines separating cells
            cbar_kws={"shrink": .8})  # Shrink the color bar

# Add title and labels
plt.title('Correlation Heatmap of RFM Analysis', fontsize=16)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal

# Show the plot
plt.show()




# Calculate the average Recency, Frequency, and Monetary scores for each segment
segment_scores = rfm.groupby('Value_Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

# Create a grouped bar chart to compare segment scores
fig = go.Figure()

# Add bars for Recency score
fig.add_trace(go.Bar(
    x=segment_scores['Value_Segment'],
    y=segment_scores['Recency'],
    name='Recency Score',
    marker_color='rgb(158,202,225)'
))

# Add bars for Frequency score
fig.add_trace(go.Bar(
    x=segment_scores['Value_Segment'],
    y=segment_scores['Frequency'],
    name='Frequency Score',
    marker_color='rgb(94,158,217)'
))

# Add bars for Monetary score
fig.add_trace(go.Bar(
    x=segment_scores['Value_Segment'],
    y=segment_scores['Monetary'],
    name='Monetary Score',
    marker_color='rgb(32,102,148)'
))

# Update the layout
fig.update_layout(
    title='Comparison of RFM Segments based on Recency, Frequency, and Monetary Scores',
    xaxis_title='RFM Segments',
    yaxis_title='Score',
    barmode='group',
    showlegend=True
)

# Show the plot
fig.show()