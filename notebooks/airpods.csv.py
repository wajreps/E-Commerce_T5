import pandas as pd

input_file = input_file = r"C:\Users\sheik\OneDrive\Desktop\daraz_airpods_scraper\airpods_reviews_500.csv"
output_file = "airpods_FINAL.csv"

df = pd.read_csv(input_file)

# In dono columns ko permanently remove karo
df = df.drop(
    columns=["original_price", "discount_percent"],
    errors="ignore"
)

# Final CSV save
df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("DONE!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Final file:", output_file)
print()
print("Columns:")
print(df.columns.tolist())