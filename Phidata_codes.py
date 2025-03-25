import os
import json
from phi.agent import Agent,AgentMemory
from phi.model.groq import Groq
import requests
from dotenv import load_dotenv
from phi.tools.yfinance import YFinanceTools
from pycoingecko import CoinGeckoAPI
import pandas as pd
from phi.tools.newspaper4k import Newspaper4k
from phi.tools.duckduckgo import DuckDuckGo
from datetime import date
from newspaper import Article
from newsapi import NewsApiClient
load_dotenv()

cg=CoinGeckoAPI()
llm=Groq(id='deepseek-r1-distill-llama-70b',temperature=0.0)
financeapi=os.getenv('FINANCE_API')
newsapi_key=os.getenv('NEWS_API')
client=NewsApiClient(api_key=newsapi_key)

def get_stock_financial_info(ticker: str, type_of_data: str):

    '''Use this function to extract stock information from alphavantage API of Yahoo Finance

    Args:
        ticket (str): Ticker of Organisation you want to extract stock info of
        type_of_data (str): You can extract 3 types of data. weekly, daily and monthly. You need to pass the keyword for the type.

    Returns:
        str: Formatted Stock finacial data of top 10 records
    '''

    url = None
    if (type_of_data == 'weekly'):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={ticker}&apikey={financeapi}'
    elif (type_of_data == 'daily'):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={financeapi}'
    elif (type_of_data == 'monthly'):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={ticker}&apikey={financeapi}'

    r = requests.get(url)
    data = r.json()

    response = {}

    for label in data:
        if label != 'Meta Data':
            for val in data[label]:
                if(len(response)==10):
                    break
                response[val] = data[label][val]

    return json.dumps(response,indent=2)

def crypto_ohlc_info(id: str, vs_currency: str, days: str = '7'):

    '''Use this function to extract crypto's ohlc values (open, high, low, close)

    Args:
        id (str): id of the crypto currency.
        vs_currency (str): Current symbol in which you want the data like 'usd' or 'inr'
        days (str): Number of days you want the data of. For month's data use 30 not 31

    Returns:
        str: String of Dataframe with date as index and ohlc columns for the number of days requested.

    '''
    ohlc = cg.get_coin_ohlc_by_id(id=id, vs_currency=vs_currency, days=days)
    df = pd.DataFrame(ohlc)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    return df.to_string()


def get_crypto_market_info(vs_currency: str, no_of_records: int):

    '''Use this function to extract crypto market information.

    Args:
        vs_currency (str): Currency you want the data in. Use values like 'usd' or 'inr'.
        no_of_records (int): Number of records to return

    Returns:
        str: String of Dataframe of extracted market information
    '''


    parameters = {
        'vs_currency': vs_currency,
        'order': 'market_cap_desc',
        'per_page': no_of_records,
        'page': 1,
        'sparkline': False,
        'locale': 'en'
    }
    coin_market_data = cg.get_coins_markets(**parameters)
    df = pd.DataFrame(coin_market_data)
    df = df.drop(['id', 'symbol', 'image', 'high_24h', 'low_24h', 'price_change_24h', 'price_change_percentage_24h',
                  'market_cap_change_24h', 'market_cap_change_percentage_24h', 'fully_diluted_valuation', 'ath_date',
                  'ath_change_percentage',
                  'atl_change_percentage', 'atl_date'], axis=1)
    return df.to_string(index=False)


def news_api(topic: str):

    '''Use this function to extract top 3 realtime news articles of the entire month based on relevancy.

    Args:
        topic (str): Topic to extract News Articles on.

    Returns:
        str: String of articles extracted
    '''
    today = date.today()
    from_date = today.replace(day=1)
    to_date = today

    res = client.get_everything(q=topic, sort_by='relevancy', from_param=from_date, to=to_date)
    if "articles" not in res or not res["articles"]:
        print("No articles found for the given topic and date range.")
        return ''

    articles = []
    # urls = []
    total_len = 0;
    for article in res["articles"]:
        article_url = article.get("url")
        if not article_url:
            continue
        article_parser = Article(url=article['url'])
        try:
            article_parser.download()
            article_parser.parse()
            if not article_parser.text.strip():
                print(f"Skipping empty article: {article_url}")
                continue

            articles.append(article_parser.text)
            total_len = total_len + len(article_parser.text)
            # urls.append(article['url'])
        except Exception as e:
            print(f"Error fetching article: {e}")

        if (total_len >= 3000 or len(articles) == 3):
            break

    Article_String = '\n\n\n'.join([f'Article {i + 1}:\n{articles[i]}' for i in range(len(articles))])

    return Article_String


toolkit=[get_stock_financial_info,YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True),
         crypto_ohlc_info,get_crypto_market_info,Newspaper4k(),DuckDuckGo(),news_api]

memory = AgentMemory(name="finance_memory")
finance_agent=Agent(name='finance agent',model=llm,tools=toolkit,markdown=True,
                    memory=memory, add_history_to_messages=True,invoke_tool_calls=True,
                    instructions=['Use Tables to display data.',
                                  'Use memory for contextual awareness.',
                                  "DO NOT print tool calls, execute them silently.",
                                  "Provide clean and readable responses to the user."])

def get_response(user_message):
    try:
        response = finance_agent.run(message=user_message)
        return response
    except Exception as e:
        return "Couldn't generate a response, Try Again."



