import pytest
from unittest.mock import patch, Mock
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Import the module under test - pipeline context handles paths automatically
from api_interface.api_connections import build_exchange_map, world_bank_metrics, oecd_metrics


@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    spark_session = (
        SparkSession.builder
        .master("local[1]")
        .appName("pipeline-tests")
        .config("worldbank_api_url", "https://api.worldbank.org/v2/country")
        .config("oecd_api_url", "https://sdmx.oecd.org/public/rest/data/")
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()


class TestBuildExchangeMap:
    """Tests for the build_exchange_map function."""
    
    @patch('api_interface.api_connections.requests.get')
    def test_build_exchange_map_success(self, mock_get, spark):
        """Verify that exchange map is built correctly from API response."""
        # Mock API response for a currency
        mock_response = Mock()
        mock_response.json.return_value = [
            {"page": 1},
            [{"value": 1.23, "country": {"value": "USA"}}]
        ]
        mock_get.return_value = mock_response
        
        exchange_map = build_exchange_map(2024)
        
        # Verify the exchange map is a Spark Column expression
        assert exchange_map is not None
        assert hasattr(exchange_map, '_jc')  # Verify it's a Column object
    
    @patch('api_interface.api_connections.requests.get')
    def test_build_exchange_map_handles_missing_data(self, mock_get, spark):
        """Verify that exchange map handles countries with no data gracefully."""
        # Mock API response with invalid value error
        mock_response = Mock()
        mock_response.json.return_value = [
            {"message": [{"key": "Invalid value"}]},
            []
        ]
        mock_get.return_value = mock_response
        
        # Should not raise an exception
        exchange_map = build_exchange_map(2024)
        assert exchange_map is not None


class TestWorldBankMetrics:
    """Tests for the world_bank_metrics function."""
    
    @patch('api_interface.api_connections.requests.get')
    def test_world_bank_metrics_success(self, mock_get, spark):
        """Verify World Bank metrics are fetched and transformed correctly."""
        # Mock API responses for latest and previous year
        mock_response_latest = Mock()
        mock_response_latest.json.return_value = [
            {"page": 1},
            [
                {
                    "countryiso3code": "USA",
                    "country": {"value": "United States"},
                    "value": 25000.5
                },
                {
                    "countryiso3code": "GBR",
                    "country": {"value": "United Kingdom"},
                    "value": 3000.7
                }
            ]
        ]
        
        mock_response_previous = Mock()
        mock_response_previous.json.return_value = [
            {"page": 1},
            [
                {
                    "countryiso3code": "USA",
                    "country": {"value": "United States"},
                    "value": 24000.0
                },
                {
                    "countryiso3code": "GBR",
                    "country": {"value": "United Kingdom"},
                    "value": 2900.0
                }
            ]
        ]
        
        mock_get.side_effect = [mock_response_latest, mock_response_previous]
        
        result_df = world_bank_metrics("gdp_ppp", "NY.GDP.MKTP.PP.CD", 2024)
        
        # Verify schema
        assert result_df.schema["country_iso3_code"].dataType == StringType()
        assert result_df.schema["latest_gdp_ppp"].dataType == DoubleType()
        assert result_df.schema["previous_gdp_ppp"].dataType == DoubleType()
        assert result_df.schema["percentage_change"].dataType == DoubleType()
        
        # Verify data
        result_data = result_df.collect()
        assert len(result_data) == 2
        
        # Check USA data
        usa_row = [row for row in result_data if row["country_iso3_code"] == "USA"][0]
        assert usa_row["latest_gdp_ppp"] == 25000.5
        assert usa_row["previous_gdp_ppp"] == 24000.0
        assert abs(usa_row["percentage_change"] - 4.169791666666667) < 0.001  # ~4.17% increase
    
    @patch('api_interface.api_connections.requests.get')
    def test_world_bank_metrics_handles_null_values(self, mock_get, spark):
        """Verify that null values in API responses are handled correctly."""
        mock_response_latest = Mock()
        mock_response_latest.json.return_value = [
            {"page": 1},
            [
                {
                    "countryiso3code": "XXX",
                    "country": {"value": "Test Country"},
                    "value": None
                }
            ]
        ]
        
        mock_response_previous = Mock()
        mock_response_previous.json.return_value = [
            {"page": 1},
            [
                {
                    "countryiso3code": "XXX",
                    "country": {"value": "Test Country"},
                    "value": 100.0
                }
            ]
        ]
        
        mock_get.side_effect = [mock_response_latest, mock_response_previous]
        
        result_df = world_bank_metrics("gdp_ppp", "NY.GDP.MKTP.PP.CD", 2024)
        result_data = result_df.collect()
        
        assert len(result_data) == 1
        assert result_data[0]["latest_gdp_ppp"] is None
        assert result_data[0]["percentage_change"] is None
    
    @patch('api_interface.api_connections.requests.get')
    def test_world_bank_metrics_handles_invalid_api_response(self, mock_get, spark):
        """Verify that invalid API responses return an empty DataFrame."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_get.return_value = mock_response
        
        result_df = world_bank_metrics("gdp_ppp", "NY.GDP.MKTP.PP.CD", 2024)
        
        # Should return empty DataFrame with correct schema
        assert result_df.count() == 0
        assert "country_iso3_code" in result_df.columns


class TestOECDMetrics:
    """Tests for the oecd_metrics function."""
    
    @patch('api_interface.api_connections.build_exchange_map')
    @patch('api_interface.api_connections.requests.get')
    def test_oecd_metrics_success(self, mock_get, mock_exchange_map, spark):
        """Verify OECD metrics are fetched and enriched with exchange rates."""
        # Mock CSV response
        csv_data = """REF_AREA,UNIT_MEASURE,SEX,OBS_VALUE,OBS_STATUS
USA,USD_PPP,TOTAL,50000,A
GBR,GBP_PPP,TOTAL,45000,A"""
        
        mock_response = Mock()
        mock_response.text = csv_data
        mock_get.return_value = mock_response
        
        # Mock exchange map
        mock_exchange_map.return_value = spark.range(1).selectExpr("map('USD', 1.0, 'GBP', 0.8) as map_col").first()["map_col"]
        
        result_df = oecd_metrics("AV_AN_WAGE", 2024)
        
        # Verify data is returned
        assert result_df is not None
        assert result_df.count() == 2
        
        # Verify columns
        expected_columns = ["REF_AREA", "UNIT_MEASURE", "SEX", "OBS_VALUE", "OBS_STATUS", "EXCHANGE_RATE"]
        assert all(col in result_df.columns for col in expected_columns)
    
    @patch('api_interface.api_connections.requests.get')
    def test_oecd_metrics_handles_request_failure(self, mock_get, spark):
        """Verify that request failures are handled gracefully."""
        mock_get.side_effect = Exception("Network error")
        
        result_df = oecd_metrics("AV_AN_WAGE", 2024)
        
        # Should return empty DataFrame on error
        assert result_df is not None
        assert result_df.count() == 0


class TestPipelineIntegration:
    """Integration tests for the complete pipeline flow."""
    
    @patch('api_interface.api_connections.requests.get')
    def test_percentage_change_calculation(self, mock_get, spark):
        """Verify that percentage change is calculated correctly."""
        mock_response_latest = Mock()
        mock_response_latest.json.return_value = [
            {"page": 1},
            [{"countryiso3code": "TST", "country": {"value": "Test"}, "value": 110.0}]
        ]
        
        mock_response_previous = Mock()
        mock_response_previous.json.return_value = [
            {"page": 1},
            [{"countryiso3code": "TST", "country": {"value": "Test"}, "value": 100.0}]
        ]
        
        mock_get.side_effect = [mock_response_latest, mock_response_previous]
        
        result_df = world_bank_metrics("test_metric", "TEST.CODE", 2024)
        result = result_df.collect()[0]
        
        # 10% increase from 100 to 110
        assert abs(result["percentage_change"] - 10.0) < 0.001
