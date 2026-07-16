from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, FloatType, IntegerType, Row
from pyspark.sql.functions import create_map, lit, split, col
import requests
from datetime import datetime
from itertools import chain
import pandas as pd
from io import StringIO

# Get the active Spark session
spark = SparkSession.getActiveSession()

oecd_codes = {
    "average_wage":"AV_AN_WAGE", # Average Wage
    "minimum_wage":"MW_CURP",  # Mininmum Wage
}

world_bank_codes = {
    "gdp_ppp":"NY.GDP.MKTP.PP.CD", #GDP PPP
    "gdp_ppp_per_capita":"NY.GDP.PCAP.PP.CD",  #GDP PPP per capita
    "current_account_balance":"BN.CAB.XOKA.CD",   # Account Balance
}

# Reverse mapping: ISO3 country code to currency
iso3_to_currency = {
    "EMU": "EUR",
    "AUS": "AUD",
    "AUT": "EUR",
    "BEL": "EUR",
    "BRA": "BRL",
    "CAN": "CAD",
    "CHL": "CLP",
    "CHN": "CNY",
    "COL": "COP",
    "CRI": "CRC",
    "CZE": "CZK",
    "DNK": "DKK",
    "EST": "EUR",
    "FIN": "EUR",
    "FRA": "EUR",
    "DEU": "EUR",
    "GRC": "EUR",
    "HUN": "HUF",
    "ISL": "ISK",
    "IND": "INR",
    "IDN": "IDR",
    "IRL": "EUR",
    "ITA": "EUR",
    "JPN": "JPY",
    "KOR": "KRW",
    "LVA": "EUR",
    "LTU": "EUR",
    "LUX": "EUR",
    "MEX": "MXN",
    "NLD": "EUR",
    "NZL": "NZD",
    "NOR": "NOK",
    "PER": "PEN",
    "POL": "PLN",
    "PRT": "EUR",
    "ROU": "RON",
    "RUS": "RUB",
    "SGP": "SGD",
    "SVK": "EUR",
    "SVN": "EUR",
    "ZAF": "ZAR",
    "ESP": "EUR",
    "SWE": "SEK",
    "CHE": "CHF",
    "THA": "THB",
    "TUR": "TRY",
    "GBR": "GBP",
    "USA": "USD"
}

iso3_to_country_name = {
    "EMU": "euro_area",
    "AUS": "australia",
    "AUT": "austria",
    "BEL": "belgium",
    "BRA": "brazil",
    "CAN": "canada",
    "CHL": "chile",
    "CHN": "china",
    "COL": "colombia",
    "CZE": "czechia",
    "DNK": "denmark",
    "EST": "estonia",
    "FIN": "finland",
    "FRA": "france",
    "DEU": "germany",
    "GRC": "greece",
    "HUN": "hungary",
    "ISL": "iceland",
    "IND": "india",
    "IDN": "indonesia",
    "IRL": "ireland",
    "ITA": "italy",
    "JPN": "japan",
    "KOR": "korea_rep",
    "LTU": "lithuania",
    "LUX": "luxembourg",
    "MEX": "mexico",
    "NLD": "netherlands",
    "NZL": "new_zealand",
    "NOR": "norway",
    "PER": "peru",
    "POL": "poland",
    "PRT": "portugal",
    "ROU": "romania",
    "RUS": "russian_federation",
    "SGP": "singapore",
    "SVK": "slovak_republic",
    "SVN": "slovenia",
    "ZAF": "south_africa",
    "ESP": "spain",
    "SWE": "sweden",
    "CHE": "switzerland",
    "TUR": "turkiye",
    "GBR": "united_kingdom",
    "USA": "united_states"
}

def build_exchange_map(year, world_bank_api_url=None):
    if world_bank_api_url is None:
        world_bank_api_url = spark.conf.get("worldbank_api_url")
    currency_exchanges = {}
    try:
        # Fetch historic exchange rates from world bank
        print(f"Getting currencies {year}")
        url = (
            f"{world_bank_api_url}/ALL/indicator/PA.NUS.FCRF?format=json&date={year}&per_page=400"
        )
        response = requests.get(url)
        data = response.json()
        for currency_data in data[1]:
            if currency_data['countryiso3code'] in iso3_to_currency:
                iso3_to_currency[currency_data['countryiso3code']]
                currency_exchanges[iso3_to_currency[currency_data['countryiso3code']]] = currency_data.get('value')
    except Exception as e:
            print(f"Error: {e}")

    return create_map(
        [lit(x) for x in chain(*currency_exchanges.items())]
    )   

def world_bank_metrics(economic_metric, world_bank_code, latest_year):
    world_bank_api_url = spark.conf.get("worldbank_api_url")
    world_bank_url = f"{world_bank_api_url}/all/indicator/{world_bank_code}?date={latest_year}&format=json&per_page=600"
    latest_data = requests.get(world_bank_url).json()
    # Get the previous year to calculate change
    world_bank_url_last = f"{world_bank_api_url}/all/indicator/{world_bank_code}?date={latest_year - 1}&format=json&per_page=600"
    last_data = requests.get(world_bank_url_last).json()

    schema = StructType([
        StructField("country_iso3_code", StringType(), True),
        StructField("country/region", StringType(), True),
        StructField(f"latest_{economic_metric}", DoubleType(), True),
        StructField(f"previous_{economic_metric}", DoubleType(), True),
        StructField("percentage_change", DoubleType(), True)
    ])

    countries_previous_data = {}
    tabulated = []

    # Check if API response has expected structure
    if not isinstance(latest_data, list) or len(latest_data) < 2:
        return spark.createDataFrame([], schema=schema)
    if not isinstance(last_data, list) or len(last_data) < 2:
        return spark.createDataFrame([], schema=schema)
    last_num_countries = len(last_data[1])
    num_countries = len(latest_data[1])

    # Hash previous country data
    for i in range(last_num_countries):
        country_iso3_code = str(last_data[1][i]["countryiso3code"])
        countries_previous_data[country_iso3_code] = last_data[1][i]["value"]

    for i in range(num_countries):
        country_iso3_code = str(latest_data[1][i]["countryiso3code"])
        latest_value = latest_data[1][i]["value"]
        previous_value = countries_previous_data.get(country_iso3_code)

        # Cast to float to ensure DoubleType compatibility
        if latest_value is not None:
            latest_value = float(latest_value)
        if previous_value is not None:
            previous_value = float(previous_value)

        if previous_value is not None and latest_value is not None:
            percentage_change = (latest_value - previous_value) / previous_value * 100
        else:
            percentage_change = None

        tabulated.append((
            country_iso3_code,
            latest_data[1][i]["country"]["value"],
            latest_value,
            previous_value,
            percentage_change,
        ))

    return spark.createDataFrame(tabulated, schema=schema)

def oecd_metrics(oecd_code, latest_year, oecd_api_url=None, world_bank_url=None):
    if oecd_api_url is None:
        oecd_api_url = spark.conf.get("oecd_api_url")
    url = (
        f"{oecd_api_url}"
        f"OECD.ELS.SAE,DSD_EARNINGS@{oecd_code},1.0/all"
        f"?startPeriod={latest_year}"
        f"&endPeriod={latest_year}"
        "&format=csvfile"
    )
    df = None
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        oecd_data = pd.read_csv(StringIO(response.text))
        wage_data = spark.createDataFrame(oecd_data)
        # Call currencies conversion
        exchange_map = build_exchange_map(latest_year, world_bank_url)
        df = (
            wage_data
            .where((col("PAY_PERIOD") == "A") & (col("PRICE_BASE") == "V"))
            .select(["REF_AREA", "UNIT_MEASURE", "SEX", "OBS_VALUE", "OBS_STATUS"])
            .withColumn(
                "EXCHANGE_RATE",
                exchange_map[split(col("UNIT_MEASURE"), "_").getItem(0)]
        )
    )
    except Exception as e:
        print(f"Error: {e}")
        schema = StructType([
            StructField("REF_AREA", StringType(), True),
            StructField("OBS_VALUE", IntegerType(), True),
            StructField("EXCHANGE_RATE", FloatType(), True)
        ])

        df = spark.createDataFrame([], schema)

    return df

def joined_economic_data(latest_year):
    min_wage_df = (
        oecd_metrics(oecd_codes['minimum_wage'], latest_year)
        .withColumn("minimum_wage", col("OBS_VALUE") / col("EXCHANGE_RATE"))
        .select("REF_AREA", "minimum_wage")
    )

    ave_wage_df = (
        oecd_metrics(oecd_codes['average_wage'], latest_year)
        .withColumn("average_wage", col("OBS_VALUE") / col("EXCHANGE_RATE"))
        .select("REF_AREA", "average_wage")
    )

    oecd_joined = min_wage_df.join(
        ave_wage_df,
        ["REF_AREA"],
        "left"
    )

    world_bank_gdp_ppp = (
        world_bank_metrics('gdp_ppp', world_bank_codes['gdp_ppp'], latest_year)
        .withColumnRenamed("latest_gdp_ppp", "gdp_ppp_dollars")
        .select("country_iso3_code", "gdp_ppp_dollars")
    )

    world_bank_gdp_ppp_per_capita = (
        world_bank_metrics('gdp_ppp_per_capita', world_bank_codes['gdp_ppp_per_capita'], latest_year)
        .withColumnRenamed("latest_gdp_ppp_per_capita", "gdp_ppp_per_capita_dollars")
        .select("country_iso3_code", "gdp_ppp_per_capita_dollars")
    )

    world_bank_current_account_balance = (
        world_bank_metrics('current_account_balance', world_bank_codes['current_account_balance'], latest_year)
        .withColumnRenamed("latest_current_account_balance", "current_account_balance_dollars")
        .select("country_iso3_code", "current_account_balance_dollars")
    )

    joined_economic_data = (
        min_wage_df
        .join(ave_wage_df, "REF_AREA", "left")
        .join(world_bank_gdp_ppp, min_wage_df.REF_AREA == world_bank_gdp_ppp.country_iso3_code, "left")
        .drop(world_bank_gdp_ppp.country_iso3_code)
        .join(world_bank_gdp_ppp_per_capita, min_wage_df.REF_AREA == world_bank_gdp_ppp_per_capita.country_iso3_code, "left")
        .drop(world_bank_gdp_ppp_per_capita.country_iso3_code)
        .join(world_bank_current_account_balance, min_wage_df.REF_AREA == world_bank_current_account_balance.country_iso3_code, "left")
        .drop(world_bank_current_account_balance.country_iso3_code)
        .withColumnRenamed("REF_AREA", "country_iso3_code")
        .drop("REF_AREA")
    )
    return joined_economic_data
