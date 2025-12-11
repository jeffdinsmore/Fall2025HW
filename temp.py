
# %%
import pandas as pd
import matplotlib.pyplot as plt

days = {"2025-11-27": "Thursday", "2025-11-28": "Friday", "2025-11-29": "Saturday", "2025-11-30": "Sunday", "2025-12-01": "Monday", "2025-12-02": "Tuesday", "2025-12-03": "Wednesday", "2025-12-04": "Thursday"}

# === File name (edit if needed) ===
csv_file = "Attic H5074_1B5A_export_202512041914.csv"

# === Import data ===
df = pd.read_csv(csv_file, parse_dates=["Timestamp"])

# === Create TimeOfDay for filtering and plotting ===
df["TimeOfDay"] = df["Timestamp"].dt.time
df["Hour"] = df["Timestamp"].dt.hour

# === Filter: EXCLUDE 11 PM (21:00) to 6 AM (07:00) ===
df_filtered = df[(df["Hour"] < 20) & (df["Hour"] >= 8)]

# === Compute average temperature ===
avg_temp = df["Temperature_Fahrenheit"].mean()
# === Calculate median temperature ===
median_temp = df_filtered["Temperature_Fahrenheit"].median()

# === Extract the date string (yyyy-mm-dd) ===
df["DateOnly"] = df["Timestamp"].dt.strftime("%Y-%m-%d")

print(f"DateOnly --------{df["DateOnly"]}")

# === Map to weekday names using dictionary ===
df["Weekday"] = df["DateOnly"].map(days)

# === Optional: if some dates not in dictionary, fill automatically ===
df["Weekday"] = df["Weekday"].fillna(df["Timestamp"].dt.day_name())
#print(f"weekday------{df["Weekday"]}")

# === Plot Temperature vs Time ===
plt.figure(figsize=(12, 5))
plt.plot(df["Timestamp"], df["Temperature_Fahrenheit"], linewidth=1.5, label="Temperature")

# === Add horizontal average line ===
plt.axhline(median_temp, color="red", linestyle="--", linewidth=1.5, label=f"Median = {median_temp:.2f}°F")

# === Labels and formatting ===
plt.title(f"Attic Temperature vs Time (Median = {median_temp:.2f}°F)", fontsize=14)
plt.xlabel("Time")
plt.ylabel("Temperature (°F)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# === Display plot ===
plt.show()


# %%
