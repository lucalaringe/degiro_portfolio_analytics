import unittest
import db_utils as db
from numpy import isclose
from datetime import datetime
import pandas as pd

# Make sure everything is up to date before running the tests. If not, run run_update.py


class PortfolioAssetReturnsConsistency(unittest.TestCase):
    # Checks for last month
    def test_daily_monthly_last(self):
        daily_returns = db.load_portfolio_asset_returns(frequency='1D')
        monthly_returns = db.load_portfolio_asset_returns(frequency='1M')
        # Identify last year and last month
        today = datetime.today()
        current_year = today.year
        current_month = today.month
        # Last month
        current_year = current_year if current_month != 1 else current_year-1
        last_month = 12 if current_month == 1 else current_month-1
        last_month_mask = str(current_year) + '-' + str(last_month)

        total_return_daily = daily_returns[last_month_mask].apply(lambda r: (1+r).prod()-1, axis=0).values
        total_return_monthly = monthly_returns.loc[last_month_mask].values[0]

        return self.assertTrue(isclose(total_return_daily, total_return_monthly).all())

    # Checks for last year
    def test_monthly_yearly_last(self):
        monthly_returns = db.load_portfolio_asset_returns(frequency='1M')
        yearly_returns = db.load_portfolio_asset_returns(frequency='1Y')
        # Identify last year and last month
        today = datetime.today()
        current_year = today.year
        last_year_mask = str(current_year-1)
        # Last year
        total_return_monthly = monthly_returns[last_year_mask].apply(lambda r: (1+r).prod()-1, axis=0).values
        total_return_yearly = yearly_returns.loc[last_year_mask].values[0]
        return self.assertTrue(isclose(total_return_monthly, total_return_yearly).all())

    def test_daily_monthly(self):
        daily_returns = db.load_portfolio_asset_returns(frequency='1D')
        monthly_returns = db.load_portfolio_asset_returns(frequency='1M')
        total_return_daily = (1+daily_returns).prod()-1
        total_return_monthly = (1+monthly_returns).prod()-1
        return self.assertTrue(isclose(total_return_daily, total_return_monthly).all())

    def test_monthly_yearly(self):
        monthly_returns = db.load_portfolio_asset_returns(frequency='1M')
        yearly_returns = db.load_portfolio_asset_returns(frequency='1Y')
        total_return_monthly = (1+monthly_returns).prod()-1
        total_return_yearly = (1+yearly_returns).prod()-1
        return self.assertTrue(isclose(total_return_monthly, total_return_yearly).all())


class PortfolioReturnsConsistency(unittest.TestCase):
    # Checks for last month
    def test_daily_monthly_last(self):
        daily_returns = db.load_portfolio_returns(frequency='1D')
        monthly_returns = db.load_portfolio_returns(frequency='1M')
        # Identify last year and last month
        today = datetime.today()
        current_year = today.year
        current_month = today.month
        # Last month
        current_year = current_year if current_month != 1 else current_year-1
        last_month = 12 if current_month == 1 else current_month-1
        last_month_mask = str(current_year) + '-' + str(last_month)

        total_return_daily = (1+daily_returns.loc[last_month_mask]).prod()-1
        total_return_monthly = monthly_returns.loc[last_month_mask].values[0]
        return self.assertTrue(isclose(total_return_daily, total_return_monthly))

    # Checks for last year
    def test_monthly_yearly_last(self):
        monthly_returns = db.load_portfolio_returns(frequency='1M')
        yearly_returns = db.load_portfolio_returns(frequency='1Y')
        # Identify last year and last month
        today = datetime.today()
        current_year = today.year
        last_year_mask = str(current_year-1)
        # Last year
        total_return_monthly = (1+monthly_returns.loc[last_year_mask]).prod()-1
        total_return_yearly = yearly_returns.loc[last_year_mask].values[0]
        return self.assertTrue(isclose(total_return_monthly, total_return_yearly))

    def test_daily_monthly(self):
        daily_returns = db.load_portfolio_returns(frequency='1D')
        monthly_returns = db.load_portfolio_returns(frequency='1M')
        total_return_daily = (1+daily_returns).prod()-1
        total_return_monthly = (1+monthly_returns).prod()-1
        return self.assertTrue(isclose(total_return_daily, total_return_monthly))

    def test_monthly_yearly(self):
        monthly_returns = db.load_portfolio_returns(frequency='1M')
        yearly_returns = db.load_portfolio_returns(frequency='1Y')
        total_return_monthly = (1+monthly_returns).prod()-1
        total_return_yearly = (1+yearly_returns).prod()-1
        return self.assertTrue(isclose(total_return_monthly, total_return_yearly))


class ReturnContributionsConsistency(unittest.TestCase):
    def test_monthly_contributions_length(self):
        contributions_df = db.load_portfolio_asset_return_contributions(frequency='1M').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1M')
        return self.assertTrue(contributions_df.shape[0] == returns_df.shape[0])

    def test_monthly_contributions(self):
        contributions_df = db.load_portfolio_asset_return_contributions(frequency='1M').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1M')
        df = pd.DataFrame(contributions_df).join(returns_df).dropna()
        return self.assertTrue(isclose(df.iloc[:, 0], df.iloc[:, 0]).all())

    def test_yearly_contributions_length(self):
        contributions_df = db.load_portfolio_asset_return_contributions(frequency='1Y').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1Y')
        return self.assertTrue(contributions_df.shape[0] == returns_df.shape[0])

    def test_yearly_contributions(self):
        contributions_df = db.load_portfolio_asset_return_contributions(frequency='1Y').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1Y')
        df = pd.DataFrame(contributions_df).join(returns_df).dropna()
        return self.assertTrue(isclose(df.iloc[:, 0], df.iloc[:, 0]).all())

    def test_monthly_asset_class_contributions_length(self):
        contributions_df = db.load_portfolio_asset_return_contributions(
            frequency='1M', groupby_field='asset_class').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1M')
        return self.assertTrue(contributions_df.shape[0] == returns_df.shape[0])

    def test_monthly_asset_class_contributions(self):
        contributions_df = db.load_portfolio_asset_return_contributions(
            frequency='1M', groupby_field='asset_class').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1M')
        df = pd.DataFrame(contributions_df).join(returns_df).dropna()
        return self.assertTrue(isclose(df.iloc[:, 0], df.iloc[:, 0]).all())

    def test_yearly_asset_class_contributions_length(self):
        contributions_df = db.load_portfolio_asset_return_contributions(
            frequency='1Y', groupby_field='asset_class').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1Y')
        return self.assertTrue(contributions_df.shape[0] == returns_df.shape[0])

    def test_yearly_asset_class_contributions(self):
        contributions_df = db.load_portfolio_asset_return_contributions(
            frequency='1Y', groupby_field='asset_class').sum(axis=1)
        returns_df = db.load_portfolio_returns(frequency='1Y')
        df = pd.DataFrame(contributions_df).join(returns_df).dropna()
        return self.assertTrue(isclose(df.iloc[:, 0], df.iloc[:, 0]).all())


if __name__ == '__main__':
    unittest.main()
