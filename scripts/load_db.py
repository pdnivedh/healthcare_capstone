import duckdb, glob, os

con = duckdb.connect("hospital.duckdb")

# Synthea tables
files = glob.glob("data/raw/synthea/*.csv")
print("Synthea files found:", len(files))
for f in files:
    name = "syn_" + os.path.splitext(os.path.basename(f))[0]
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{f}')")

# UCI Diabetes table
uci = "data/raw/uci/diabetic_data.csv"
if os.path.exists(uci):
    con.execute(f"CREATE OR REPLACE TABLE uci_diabetes AS SELECT * FROM read_csv_auto('{uci}', nullstr='?', sample_size=-1)")
    print("UCI rows:", con.execute("SELECT COUNT(*) FROM uci_diabetes").fetchone()[0])
else:
    print("UCI file not found at", uci, "- skipping")

print(con.execute("SHOW TABLES").fetchall())
con.close()
