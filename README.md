# data-cleaning-and-structural-validation
# ============================================================
# DATA CLEANING & STRUCTURAL VALIDATION
# ============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. LOAD THE RAW DATASET
# ------------------------------------------------------------

# Change the filename to your downloaded raw dataset name
input_file = "raw_dataset.csv"

df = pd.read_csv(input_file)

print("Original Dataset:")
print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 2. INSPECT THE DATA
# ------------------------------------------------------------

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe(include="all"))


# ------------------------------------------------------------
# 3. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nPercentage of Missing Values:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)


# ------------------------------------------------------------
# 4. CHECK DUPLICATE RECORDS
# ------------------------------------------------------------

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

print("\nDuplicate Rows:")
print(df[df.duplicated()])


# ------------------------------------------------------------
# 5. REMOVE DUPLICATE RECORDS
# ------------------------------------------------------------

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)


# ------------------------------------------------------------
# 6. STANDARDIZE COLUMN HEADERS
# ------------------------------------------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nStandardized Column Names:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 7. REMOVE EXTRA SPACES FROM TEXT/CATEGORICAL COLUMNS
# ------------------------------------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()


# ------------------------------------------------------------
# 8. STANDARDIZE CATEGORICAL STRINGS
# ------------------------------------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.lower()

print("\nCategorical Values After Standardization:")

for column in df.select_dtypes(include="object").columns:
    print("\n", column)
    print(df[column].unique())


# ------------------------------------------------------------
# 9. CONVERT COMMON MISSING-VALUE REPRESENTATIONS TO NaN
# ------------------------------------------------------------

missing_values = [
    "",
    " ",
    "na",
    "n/a",
    "NA",
    "N/A",
    "null",
    "NULL",
    "none",
    "None",
    "-"
]

df = df.replace(missing_values, np.nan)


# ------------------------------------------------------------
# 10. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nData Types Before Conversion:")
print(df.dtypes)


# ------------------------------------------------------------
# 11. AUTOMATICALLY CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

for column in df.columns:
    converted = pd.to_numeric(df[column], errors="coerce")

    # Convert only if most values can reasonably be numeric
    if converted.notna().sum() >= df[column].notna().sum() * 0.8:
        df[column] = converted


# ------------------------------------------------------------
# 12. DETECT DATE COLUMNS
# ------------------------------------------------------------

for column in df.columns:

    if "date" in column.lower():

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

print("\nData Types After Conversion:")
print(df.dtypes)


# ------------------------------------------------------------
# 13. HANDLE MISSING NUMERIC VALUES
# ------------------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:

    median_value = df[column].median()

    df[column] = df[column].fillna(median_value)


# ------------------------------------------------------------
# 14. HANDLE MISSING CATEGORICAL VALUES
# ------------------------------------------------------------

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].notna().any():

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(mode_value)


# ------------------------------------------------------------
# 15. HANDLE MISSING DATE VALUES
# ------------------------------------------------------------

date_columns = df.select_dtypes(
    include=["datetime64[ns]"]
).columns

for column in date_columns:

    if df[column].notna().any():

        median_date = df[column].dropna().median()

        df[column] = df[column].fillna(median_date)


# ------------------------------------------------------------
# 16. CHECK FOR NEGATIVE NUMERIC VALUES
# ------------------------------------------------------------

print("\nNegative Values:")

for column in numeric_columns:

    negative_count = (df[column] < 0).sum()

    if negative_count > 0:
        print(column, ":", negative_count)


# ------------------------------------------------------------
# 17. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\nOutlier Detection:")

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(
        column,
        "->",
        len(outliers),
        "outliers"
    )


# ------------------------------------------------------------
# 18. REMOVE EXACT DUPLICATES AGAIN
# ------------------------------------------------------------

df = df.drop_duplicates()


# ------------------------------------------------------------
# 19. VALIDATE MISSING VALUES AGAIN
# ------------------------------------------------------------

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 20. VALIDATE DATA TYPES AGAIN
# ------------------------------------------------------------

print("\nFinal Data Types:")
print(df.dtypes)


# ------------------------------------------------------------
# 21. VALIDATE DUPLICATES AGAIN
# ------------------------------------------------------------

print("\nDuplicates After Cleaning:")
print(df.duplicated().sum())


# ------------------------------------------------------------
# 22. FINAL DATASET
# ------------------------------------------------------------

print("\n==============================")
print("FINAL CLEANED DATASET")
print("==============================")

print(df.head())

print("\nFinal Shape:")
print(df.shape)


# ------------------------------------------------------------
# 23. EXPORT CLEANED CSV
# ------------------------------------------------------------

output_file = "cleaned_dataset.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved as:")
print(output_file)


# ------------------------------------------------------------
# 24. CREATE CLEANING REPORT
# ------------------------------------------------------------

report = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values,
    "Missing_Values": df.isnull().sum().values,
    "Unique_Values": df.nunique().values
})

report.to_csv(
    "data_cleaning_report.csv",
    index=False
)

print("\nData cleaning report saved as:")
print("data_cleaning_report.csv")


# ------------------------------------------------------------
# 25. FINAL VALIDATION MESSAGE
# ------------------------------------------------------------

print("\n====================================")
print("DATA CLEANING COMPLETED SUCCESSFULLY")
print("====================================")

print("Duplicates remaining:", df.duplicated().sum())
print("Missing values remaining:", df.isnull().sum().sum())
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))