from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import sys
from datetime import datetime
from numpy import nan
import pandas as pd
from db_utils import load_product_details
from constants import DATE_FORMAT, PRODUCT_DETAILS_CSV_PATH


def process_string(s: str) -> str:
    s = s.strip()
    s = re.sub(' +', ' ', s)
    s = re.sub('\n', '', s)
    return s.strip()


def normalize_just_etf_beautiful_soup(response: BeautifulSoup) -> pd.Series:
    # We build a dict with each attribute we care about
    etf_dict = dict()

    etf_dict['just_etf_name'] = response.find('span', attrs={'class': 'h1'}).text.strip()

    try:
        isin, ticker = response.find('span', attrs={'class': 'identfier'}  # Typo in the JustETF html code
                                     ).findAll('span', attrs={'class': 'val'})
        etf_dict['isin'] = isin.text.strip()[:-1]  # they put ',' after ISIN in the tag
        etf_dict['ticker'] = ticker.text.strip()
    except ValueError:
        isin = response.find('span', attrs={'class': 'identfier'})
        etf_dict['isin'] = isin.text.replace('ISIN', '').replace('\n', '')
        etf_dict['ticker'] = ''

    etf_dict['description'] = response.find(string=re.compile("Investment strategy")).findNext('p').contents[
        0].strip()
    etf_dict['description'] = process_string(etf_dict['description'])

    etf_dict['benchmark'] = \
        response.find(
            string=re.compile("Investment strategy")).parent.findNext('a').text.strip().rsplit(' ', 1)[0]
    etf_dict['asset_class'] = \
        response.find(
            string=re.compile("Investment strategy")).parent.findNext('a').findNext('a').text.strip().rsplit(' ', 1)[0]

    etf_dict['replication'] = \
        re.sub("[\t\n]", '',
               response.find(string=re.compile("Replication")).parent.parent.find_next_sibling('td').text.strip())
    etf_dict['replication'] = process_string(etf_dict['replication'])
    etf_dict['just_etf_currency'] = response.find(string=re.compile("Fund currency")
                                                  ).parent.find_next_sibling('td').text.strip()
    etf_dict['fund_size'] = re.sub("[\t\n]", '', response.find(
        'div', string=re.compile("Fund size")).findPrevious('div').contents[0].strip())
    etf_dict['fund_size'] = process_string(etf_dict['fund_size'])
    etf_dict['fund_size'] = float(etf_dict['fund_size'].strip(etf_dict['just_etf_currency']).strip('EUR').
                                  strip('mln').strip().replace(',', '').replace('.', ',')) * 1000000
    fs_category = response.find('img', attrs={'alt': 'Fund size category', 'data-toggle': 'tooltip'})['class']
    etf_dict['fund_size_category'] = \
        "low-cap" if fs_category[-1][-1] == "1" else \
        "mid-cap" if fs_category[-1][-1] == "2" else "large-cap"
    etf_dict['currency_risk'] = \
        response.find(string=re.compile("Currency risk")).parent.parent.find_next_sibling('td').text.strip()
    etf_dict['inception_date'] = datetime.strptime(
        response.find(string=re.compile("Inception/ Listing Date")
                      ).parent.find_next_sibling('td').text.strip(), "%d %B %Y").date().strftime(DATE_FORMAT)
    etf_dict['ter'] = response.find(string=re.compile("Total expense ratio")
                                    ).parent.find_previous_sibling('div').text.strip()
    etf_dict['ter'] = float(re.sub("[^0-9.]+[.]", "", etf_dict['ter'])) / 100
    etf_dict['distribution_policy'] = response.find(string=re.compile("Distribution policy")
                                                    ).parent.find_next_sibling('td').text.strip()
    etf_dict['fund_domicile'] = response.find(string=re.compile("Fund domicile")
                                              ).parent.find_next_sibling('td').text.strip()
    etf_dict['fund_structure'] = response.find(string=re.compile("Fund Structure")
                                               ).parent.find_next_sibling('td').text.strip()
    etf_dict['fund_provider'] = response.find(string=re.compile("Fund Provider")
                                              ).parent.find_next_sibling('td').text.strip()
    etf_dict['administrator'] = response.find(string=re.compile("Administrator")
                                              ).parent.find_next_sibling('td').text.strip()
    etf_dict['last_update'] = datetime.today().strftime(DATE_FORMAT)
    return pd.Series(etf_dict)


def scrape_just_etf(isin_ls: list) -> pd.DataFrame:
    assert type(isin_ls) == list, 'Warning! isin_ls must be of type list.'

    # Base Case
    if len(isin_ls) == 1:
        isin = isin_ls[0]
        try:
            with urlopen(f'https://www.justetf.com/en/etf-profile.html?isin={isin}') as connection:
                response = BeautifulSoup(connection, 'html.parser')
            return pd.DataFrame(normalize_just_etf_beautiful_soup(response)).T
        except AttributeError:
            print(f"\nFund isin '{isin}' not found!\n", file=sys.stderr)

    # Recursive Case
    else:
        return pd.concat([scrape_just_etf([isin]) for isin in isin_ls], axis=0)


def augment_product_details_csv(product_details_csv_path: str = PRODUCT_DETAILS_CSV_PATH) -> None:
    product_details = load_product_details(product_details_csv_path=product_details_csv_path)
    etf_isin_ls = product_details.loc[product_details['product_type'] == 'ETF', 'isin'].tolist()
    just_etf_scraped_data = scrape_just_etf(isin_ls=etf_isin_ls)
    just_etf_scraped_data_isin_ls = just_etf_scraped_data.loc[:, 'isin'].unique().tolist()
    just_etf_scraped_data_fields = just_etf_scraped_data.columns
    # Merge Degiro Fields with Just ETF fields
    augmented_product_details = product_details.copy(deep=True)
    filter_row_mask = augmented_product_details['isin'].isin(just_etf_scraped_data_isin_ls)

    # Check if columns already exist, otherwise create them
    cols_missing = [col for col in just_etf_scraped_data_fields if not (col in augmented_product_details.columns)]
    if len(cols_missing) > 0:
        augmented_product_details.loc[:, cols_missing] = nan

    # Augment data
    assert augmented_product_details.loc[filter_row_mask, just_etf_scraped_data_fields].shape[0] ==\
           just_etf_scraped_data.shape[0], 'Warning! Shapes mismatch'
    isin_sorted_list = augmented_product_details.loc[filter_row_mask, 'isin'].unique().tolist()
    just_etf_scraped_data.sort_values(by=['isin'],
                                      key=lambda x: x.map(dict(zip(isin_sorted_list, range(len(isin_sorted_list))))),
                                      inplace=True)

    augmented_product_details.loc[filter_row_mask, just_etf_scraped_data_fields] = just_etf_scraped_data.values

    # Final Adjustments
    if 'name' not in augmented_product_details.columns:
        augmented_product_details.loc[:, 'name'] = nan
    augmented_product_details.loc[:, 'name'] = \
        augmented_product_details.loc[:, 'just_etf_name'].fillna(augmented_product_details.loc[:, 'degiro_name'])

    augmented_product_details.drop_duplicates().to_csv(PRODUCT_DETAILS_CSV_PATH)


if __name__ == "__main__":
    augment_product_details_csv()
