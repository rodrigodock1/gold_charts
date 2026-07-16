from pyspark import pipelines as dp
from datetime import datetime
from api_interface.api_connections import world_bank_metrics, oecd_metrics, iso3_to_country_name, joined_economic_data, oecd_codes, world_bank_codes
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

# Latest data provided by the World Bank
latest_year = datetime.now().year - 2 

# world_bank_metrics
for world_bank_metric, world_bank_code in world_bank_codes.items():
    @dp.table(name=f"{world_bank_metric}_silver", table_properties={"quality": "silver"})
    @dp.expect_all_or_drop({
        "previous_gdp_ppp_not_null": f"previous_{world_bank_metric} IS NOT NULL",
        "latest_gdp_ppp_not_null": f"latest_{world_bank_metric} IS NOT NULL"
    })
    def table(metric=world_bank_metric, code=world_bank_code):
        return world_bank_metrics(metric, code, latest_year)
    
# oecd_bank_metrics
for oecd_metric, oecd_code in oecd_codes.items():
    @dp.table(name=f"{oecd_metric}_silver", table_properties={"quality": "silver"})
    @dp.expect_or_drop(
        "exchange_rate_not_null_and_exclude_countries",
        "EXCHANGE_RATE IS NOT NULL AND OBS_VALUE IS NOT NULL AND REF_AREA NOT IN ('ISR', 'LVA', 'CRI')"
    )
    def table(indicator=oecd_code):
        return oecd_metrics(indicator, latest_year)
    
# For each country pipeline 
all_countries_metrics = joined_economic_data(latest_year)
@dp.temporary_view(name="all_countries_metrics_view")
def all_countries_metrics_view():
    return (
        all_countries_metrics
        .withColumn("year", lit(latest_year))
    )

def create_silver_table(country_code, country_name):
    @dp.table(name=f"{country_name}_silver", table_properties={"quality": "silver"})
    @dp.expect(
        "all_data_exist",
        """
        gdp_ppp_dollars IS NOT NULL
        AND gdp_ppp_per_capita_dollars IS NOT NULL
        AND current_account_balance_dollars IS NOT NULL
        AND average_wage IS NOT NULL
        AND minimum_wage IS NOT NULL
        """
    )
    def silver():
        bronze = spark.read.table(
            f"main.gold_charts.{country_name}_bronze"
        )

        new_year = (
            spark.read.table("all_countries_metrics_view") 
            .where(col("country_iso3_code") == country_code)
            .withColumn("country_name", lit(country_name))
        )
        return (
            bronze
            .unionByName(new_year)
            .dropDuplicates(["country_iso3_code", "year"])
        )
    
for country_code, country_name in iso3_to_country_name.items():
    create_silver_table(country_code, country_name)
