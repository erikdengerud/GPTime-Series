https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765
https://github.com/suriyadeepan/torchtest

Place the following notice prominently on your application: "This product uses the 
FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."


    Data sources: 
    * Tiingo :  historical end-of-day prices on equities, mutual funds and ETFs.
                Free accounts are rate limited and can access a limited number of 
                symbols (500 at the time of writing)
    * IEX    :  Historical stock prices are available for up to 15 years. The usage of 
                these readers requires the publishable API key from IEX Cloud Console, 
                which can be stored in the IEX_API_KEY environment variable.
    * Alpha Vantage : 
                Alpha Vantage provides realtime equities and forex data. 
                Free registration is required to get an API key. 
                    - Through the Alpha Vantage Time Series endpoints, it is possible 
                    to obtain historical equities and currency rate data for individual 
                    symbols. For daily, weekly, and monthly frequencies, 20+ years of 
                    historical data is available. The past 3-5 days of intraday data is 
                    also available.
                    - Alpha Vantage Batch Stock Quotes endpoint allows the retrieval of 
                    realtime stock quotes for up to 100 symbols at once. 

    * Econdb :  Econdb provides economic data from 90+ official statistical agencies. 
                Free API allows access to the complete Econdb database of time series 
                aggregated into datasets.
    * Enigma :  Access datasets from Enigma, the world’s largest repository of 
                structured public data. Note that the Enigma URL has changed from 
                app.enigma.io as of release 0.6.0, as the old API deprecated.

                Datasets are unique identified by the uuid4 at the end of a dataset’s 
                web address. For example, the following code downloads from 
                USDA Food Recalls 1996 Data.
    * Quandl :  Daily financial data (prices of stocks, ETFs etc.) from Quandl.
    * St.Louis FED (FRED) :
                Federal Reserve Economic Data.
    * Kenneth French’s data library : 
                Access datasets from the Fama/French Data Library. 
    * World Bank :
                pandas users can easily access thousands of panel data series from the 
                World Bank’s World Development Indicators by using the wb I/O functions.
                Either from exploring the World Bank site, or using the search function 
                included, every world bank indicator is accessible.
                Free and open access to global development data.
    * OECD   :  OECD Statistics are available via DataReader. You have to specify OECD’s 
                data set code. OECD.Stat includes data and metadata for OECD countries 
                and selected non-member economies.
    * Eurostat :
                Your key to European statistics
    * Thrift Savings Plan :
                Download mutual fund index prices for the Thrift Savings Plan (TSP).
    * Nasdaq Trader symbol definitions :
                Download the latest symbols from Nasdaq.
    * Stooq :   Google finance doesn’t provide common index data download. The Stooq 
                site has the data for download.
    * MOEX :    The Moscow Exchange (MOEX) provides historical data.
    * Naver Finance :
                Naver Finance provides Korean stock market (KOSPI, KOSDAQ) historical data.

    --- Caching queries