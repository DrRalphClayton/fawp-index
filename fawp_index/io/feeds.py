"""
fawp-index: Real data loaders
Yahoo Finance and NOAA weather data feeds.
"""

import numpy as np
from ..io.csv_loader import FAWPData


def load_yahoo_finance(
    ticker: str,
    period: str = "2y",
    delta_pred: int = 20,
    action_col: str = "volume",
):
    """
    Load stock data from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'SPY', 'AAPL', 'BTC-USD')
    period : str
        Data period: '1y', '2y', '5y', 'max'
    delta_pred : int
        Forecast horizon (days).
    action_col : str
        Column to use as action proxy: 'volume' or 'Open'

    Returns
    -------
    FAWPData

    Example
    -------
        from fawp_index.io.feeds import load_yahoo_finance
        data = load_yahoo_finance('SPY', period='2y', delta_pred=20)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for Yahoo Finance data.\n"
            "Install it with: pip install yfinance"
        )

    print(f"Fetching {ticker} from Yahoo Finance ({period})...")
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Log returns as predictor
    close = df['Close'].values.flatten().astype(float)
    log_returns = np.diff(np.log(close))

    n = len(log_returns) - delta_pred
    if n < 50:
        raise ValueError(f"Not enough data: {n} rows after alignment")

    pred = log_returns[:n]
    future = log_returns[delta_pred:delta_pred + n]

    # Action proxy: normalized volume change or open-to-close
    if action_col == 'volume' and 'Volume' in df.columns:
        vol = df['Volume'].values.flatten().astype(float)[1:]  # align with returns
        vol_norm = np.diff(np.log(vol + 1))
        action = vol_norm[:n] if len(vol_norm) >= n else np.zeros(n)
    else:
        opens = df['Open'].values.flatten().astype(float)
        action = np.diff(np.log(opens))[:n]

    obs = log_returns[:n]  # direct observation

    print(f"Loaded {n:,} daily observations for {ticker}")

    return FAWPData(
        pred_series=pred,
        future_series=future,
        action_series=action,
        obs_series=obs,
        metadata={
            "source": f"Yahoo Finance: {ticker}",
            "period": period,
            "delta_pred": delta_pred,
            "n_rows": n,
            "ticker": ticker,
        }
    )


def load_noaa_weather(
    station_id: str,
    start_date: str,
    end_date: str,
    variable: str = "TMAX",
    delta_pred: int = 5,
    api_token: str = None,
):
    """
    Load NOAA weather station data via the NOAA CDO API.

    Parameters
    ----------
    station_id : str
        NOAA station ID (e.g. 'GHCND:USW00094728' for NYC Central Park)
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str
        End date 'YYYY-MM-DD'
    variable : str
        Weather variable: 'TMAX', 'TMIN', 'PRCP', 'SNOW'
    delta_pred : int
        Forecast horizon (days).
    api_token : str
        NOAA CDO API token (get free at: https://www.ncdc.noaa.gov/cdo-web/token)

    Returns
    -------
    FAWPData

    Example
    -------
        from fawp_index.io.feeds import load_noaa_weather
        data = load_noaa_weather(
            station_id='GHCND:USW00094728',
            start_date='2020-01-01',
            end_date='2023-12-31',
            variable='TMAX',
            api_token='your_token_here',
        )
    """
    try:
        import requests
    except ImportError:
        raise ImportError("pip install requests")

    if api_token is None:
        raise ValueError(
            "NOAA API token required.\n"
            "Get a free token at: https://www.ncdc.noaa.gov/cdo-web/token"
        )

    print(f"Fetching NOAA data for station {station_id} ({variable})...")

    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    headers = {"token": api_token}
    params = {
        "datasetid": "GHCND",
        "stationid": station_id,
        "datatypeid": variable,
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "units": "standard",
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(f"NOAA API error {response.status_code}: {response.text}")

    data_json = response.json()
    if "results" not in data_json:
        raise ValueError("No results returned from NOAA API")

    values = np.array([r["value"] for r in data_json["results"]], dtype=float)
    n = len(values) - delta_pred

    if n < 30:
        raise ValueError(f"Not enough NOAA data: {n} rows")

    pred = values[:n]
    future = values[delta_pred:delta_pred + n]
    # Action proxy: day-over-day change (forecast adjustment)
    action = np.diff(values)[:n] if len(values) > n else np.zeros(n)
    obs = values[:n]

    print(f"Loaded {n:,} daily {variable} observations")

    return FAWPData(
        pred_series=pred,
        future_series=future,
        action_series=action,
        obs_series=obs,
        metadata={
            "source": f"NOAA CDO: {station_id}",
            "variable": variable,
            "start_date": start_date,
            "end_date": end_date,
            "delta_pred": delta_pred,
            "n_rows": n,
        }
    )


def load_synthetic_demo(domain: str = "finance", seed: int = 42) -> FAWPData:
    """
    Load a synthetic demo dataset for quick testing.
    No API keys or files needed.

    Parameters
    ----------
    domain : str
        'finance', 'weather', or 'seismic'
    seed : int
        Random seed.

    Example
    -------
        from fawp_index.io.feeds import load_synthetic_demo
        data = load_synthetic_demo('seismic')
    """
    rng = np.random.default_rng(seed)

    if domain == "finance":
        n = 5000
        returns = rng.normal(0.001, 0.02, n)
        pred = returns[:-20]
        future = returns[20:]
        action = rng.normal(0, 0.01, len(pred))
        obs = returns[:-20] + rng.normal(0, 0.005, len(pred))

    elif domain == "weather":
        n = 2000
        temp = 20 + 10 * np.sin(np.linspace(0, 8 * np.pi, n)) + rng.normal(0, 2, n)
        pred = temp[:-5]
        future = temp[5:]
        action = rng.normal(0, 0.5, len(pred))
        obs = temp[:-5] + rng.normal(0, 1, len(pred))

    elif domain == "seismic":
        n = 3000
        stress = np.cumsum(rng.uniform(0.001, 0.01, n))
        stress = stress / stress.max()
        pred = stress[:-24]
        future = stress[24:]
        action = rng.normal(0, 0.001, len(pred))  # near-zero — no earthquake off-switch
        obs = stress[:-24] * 0.0 + rng.normal(0, 0.1, len(pred))  # zero coupling

    else:
        raise ValueError(f"Unknown domain '{domain}'. Choose: finance, weather, seismic")

    min_len = min(len(pred), len(future), len(action), len(obs))
    return FAWPData(
        pred_series=pred[:min_len],
        future_series=future[:min_len],
        action_series=action[:min_len],
        obs_series=obs[:min_len],
        metadata={"source": f"synthetic_demo:{domain}", "seed": seed}
    )
