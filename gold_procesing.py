# Exploration by country
from pyspark.sql.functions import col, lit, array, split, map_from_arrays
import datetime
from datetime import datetime

iso3_to_country_full = {
    "EMU": "Euro Area",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHL": "Chile",
    "CHN": "China",
    "COL": "Colombia",
    "CZE": "Czechia",
    "DNK": "Denmark",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "DEU": "Germany",
    "GRC": "Greece",
    "HUN": "Hungary",
    "ISL": "Iceland",
    "IND": "India",
    "IDN": "Indonesia",
    "IRL": "Ireland",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "korea_rep",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MEX": "Mexico",
    "NLD": "Netherlands",
    "NZL": "New Zealand",
    "NOR": "Norway",
    "PER": "Peru",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "RUS": "Russian Federation",
    "SGP": "Singapore",
    "SVK": "Slovak Republic",
    "SVN": "Slovenia",
    "ZAF": "South Africa",
    "ESP": "Spain",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "TUR": "Turkiye",
    "GBR": "United Kingdom",
    "USA": "United States"
}

latest_year = datetime.now().year - 2
# Reading tables
historic_gold_price = spark.read.table(f"main.gold_charts.historic_gold_prices")
average_wage = spark.read.table("main.gold_charts.average_wage_silver")
minimum_wage = spark.read.table("main.gold_charts.minimum_wage_silver")
gdp_ppp = spark.read.table("main.gold_charts.gdp_ppp_silver")
gdp_ppp_per_capita = spark.read.table("main.gold_charts.gdp_ppp_per_capita_silver")
current_account_balance = spark.read.table("main.gold_charts.current_account_balance_silver")
# Setup variables
recent_gold_price = historic_gold_price.where(f"year = {latest_year}").select("Price (USD per kg)", "Price (USD per troy ounce)").collect()
previous_gold_price = spark.read.table("main.gold_charts.historic_gold_prices").where(f"year = {latest_year-1}").select("Price (USD per kg)", "Price (USD per troy ounce)").collect()
keys = array([lit(k) for k in iso3_to_country_full.keys()])
values = array([lit(v) for v in iso3_to_country_full.values()])
country_map = map_from_arrays(keys, values)

db_host = spark.conf.get("db_host")
db_user = spark.conf.get("db_user")
db_port = spark.conf.get("db_port")
gold_db = spark.conf.get("gold_database")
db_password = dbutils.secrets.get(scope="gold-charts", key="supabase_password")
jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{gold_db}?user={db_user}&password={db_password}&prepareThreshold=0"

connection_properties = {
    "user": db_user,
    "password": db_password,
    "driver": "org.postgresql.Driver"
}

def write_to_gold_table(df, table_name):
    df.write \
        .format("jdbc") \
        .option("url", jdbc_url) \
        .option("dbtable", table_name) \
        .option("user", connection_properties["user"]) \
        .option("password", connection_properties["password"]) \
        .mode("overwrite") \
        .save()

# Minimum Wage tables
gold_minimum_wage = (
    minimum_wage
    .filter(split(col("REF_AREA"), "_").getItem(0).isin(list(iso3_to_country_full.keys())))
    .withColumn(
        "country/region",
        country_map[split(col("REF_AREA"), "_").getItem(0)]
    )
    .withColumn("minimum_wage_gold", col("OBS_VALUE")/recent_gold_price[0]["Price (USD per kg)"])
    .withColumnsRenamed({"OBS_VALUE": "minimum_wage_dollars", "REF_AREA": "country_iso3_code"})
    .drop("OBS_VALUE", "EXCHANGE_RATE", "SEX", "UNIT_MEASURE", "OBS_STATUS")
)
write_to_gold_table(gold_minimum_wage, "minimum_wage")

#Average Wage table
gold_average_wage = (
    average_wage
    .filter(split(col("REF_AREA"), "_").getItem(0).isin(list(iso3_to_country_full.keys())))
    .withColumn(
        "country/region",
        country_map[split(col("REF_AREA"), "_").getItem(0)]
    )
    .withColumn("average_wage_gold", col("OBS_VALUE")/recent_gold_price[0]["Price (USD per kg)"])
    .withColumnsRenamed({"OBS_VALUE": "average_wage_dollars", "REF_AREA": "country_iso3_code"})
    .drop("OBS_VALUE", "EXCHANGE_RATE", "SEX", "UNIT_MEASURE", "OBS_STATUS")
)
write_to_gold_table(gold_average_wage, "average_wage")

# GDP table
gold_gdp_ppp = (
    gdp_ppp
    .filter(col("country_iso3_code").isin(list(iso3_to_country_full.keys())))
    .withColumn("gdp_ppp_gold", col("latest_gdp_ppp")/recent_gold_price[0]["Price (USD per kg)"])
    .withColumn("gdp_ppp_gold_t_oz", col("latest_gdp_ppp")/recent_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("previous_gdp_ppp_gold", col("previous_gdp_ppp")/previous_gold_price[0]["Price (USD per kg)"])
    .withColumn("previous_gdp_ppp_gold_t_oz", col("previous_gdp_ppp")/previous_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("percentage_change_in_gold", (col("gdp_ppp_gold")/col("previous_gdp_ppp_gold")-1)*100)
    .withColumnsRenamed(
        {
            "latest_gdp_ppp": "latest_gdp_ppp_dollars", 
            "previous_gdp_ppp": "previous_gdp_ppp_dollars",
            "percentage_change": "percentage_change_in_dollars"
        }
    )
)
write_to_gold_table(gold_gdp_ppp, "gdp_ppp")

# GDP per Capita table
gold_gdp_ppp_per_capita = (
    gdp_ppp_per_capita
    .filter(col("country_iso3_code").isin(list(iso3_to_country_full.keys())))
    .withColumn("gdp_ppp_per_capita_gold", col("latest_gdp_ppp_per_capita")/recent_gold_price[0]["Price (USD per kg)"])
    .withColumn("gdp_ppp_capita_gold_t_oz", col("latest_gdp_ppp_per_capita")/recent_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("previous_gdp_ppp_per_capita_gold", col("previous_gdp_ppp_per_capita")/previous_gold_price[0]["Price (USD per kg)"])
    .withColumn("previous_gdp_ppp_per_capita_gold_t_oz", col("previous_gdp_ppp_per_capita")/previous_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("percentage_change_in_gold", (col("gdp_ppp_per_capita_gold")/col("previous_gdp_ppp_per_capita_gold")-1)*100)
    .withColumnsRenamed(
        {
            "latest_gdp_ppp_per_capita": "latest_gdp_ppp_per_capita_dollars", 
            "previous_gdp_ppp_per_capita": "previous_gdp_ppp_capita_dollars",
            "percentage_change": "percentage_change_in_dollars"
        }
    )
)
write_to_gold_table(gold_gdp_ppp_per_capita, "gdp_ppp_per_capita")

gold_current_account_balance = (
    current_account_balance
    .filter(col("country_iso3_code").isin(list(iso3_to_country_full.keys())))
    .withColumn("current_account_balance_gold", col("latest_current_account_balance")/recent_gold_price[0]["Price (USD per kg)"])
    .withColumn("current_account_balance_gold_t_oz", col("latest_current_account_balance")/recent_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("previous_current_account_balance_gold", col("previous_current_account_balance")/previous_gold_price[0]["Price (USD per kg)"])
    .withColumn("previous_current_account_balance_gold_t_oz", col("previous_current_account_balance")/previous_gold_price[0]["Price (USD per troy ounce)"])
    .withColumn("percentage_change_in_gold", (col("current_account_balance_gold")/col("previous_current_account_balance_gold")-1)*100)
    .withColumnsRenamed(
        {
            "latest_current_account_balance": "latest_current_account_balance_dollars", 
            "previous_current_account_balance": "previous_current_account_balance_dollars",
            "percentage_change": "percentage_change_in_dollars"
        }
    )
)
write_to_gold_table(gold_current_account_balance, "current_account_balance")

# For each country generation
for country_name in iso3_to_country_full.values():   
    country_table = spark.read.table(f"main.gold_charts.{country_name.lower().replace(' ', '_')}_silver")
    gold_calculation = country_table.join(historic_gold_price, on="year")

    country_calculation = (
        gold_calculation
        .withColumn("gdp_ppp_gold", col("gdp_ppp_dollars")/col("Price (USD per kg)"))
        .withColumn("gdp_ppp_per_capita_gold", col("gdp_ppp_per_capita_dollars")/col("Price (USD per kg)"))
        .withColumn("current_account_balance_gold", col("current_account_balance_dollars")/col("Price (USD per kg)"))
        .withColumn("minimum_wage_gold", col("minimum_wage")/col("Price (USD per kg)"))
        .withColumn("average_wage_gold", col("average_wage")/col("Price (USD per kg)"))
        .withColumnRenamed("minimum_wage", "minimum_wage_dollars")
        .withColumnRenamed("average_wage", "average_wage_dollars")
    )

    write_to_gold_table(country_calculation, f"{country_name.lower().replace(' ', '_')}_data")


