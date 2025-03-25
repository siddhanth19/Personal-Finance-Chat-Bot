import requests
from newsapi import NewsApiClient
from newspaper import Article
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_groq import ChatGroq
from langchain import hub
import os
from datetime import date
from pycoingecko import CoinGeckoAPI
import pandas as pd
import dotenv

from langchain_community.tools import DuckDuckGoSearchResults

dotenv.load_dotenv()
newsapi=os.getenv('NEWS_API')
financeapi=os.getenv('FINANCE_API')


client=NewsApiClient(api_key=newsapi)
cg=CoinGeckoAPI()

@tool(response_format='content')
def web_surf_response(topic:str):
    '''Search about a topic on Internet using DuckDuckGo search tool of langchain
        Note search for anything that you are unable to find in your knowledge base using this tool
    '''
    search = DuckDuckGoSearchResults()
    response=search.invoke(topic)
    return response

@tool(response_format='content')
def get_stock_financial_info(ticker: str, type_of_data: str):
    '''This Function is used to extract stock information given the ticker of an organisation and type_of_data
        Note the type_of_data can be daily, weekly and monthly. This function returns a dictionary with keys as date and values as dictionary of stock values
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
                if (len(response) == 5):
                    break
                response[val] = data[label][val]

    return response


@tool(response_format='content')
def crypto_ohlc_info(id: str, vs_currency: str, days: str = '7'):
    '''Extracts and returns crypto's ohlc values (open, high, low, close)
        The results will be returned as a Dataframe with index as date.
        Note you need to give the id of crypto currency and vs_currency like 'usd'.
        You can also provide the number of days of data you want to extract.
        For monthly request enter '30'.
    '''
    cg = CoinGeckoAPI()
    ohlc = cg.get_coin_ohlc_by_id(id=id, vs_currency=vs_currency, days=days)
    df = pd.DataFrame(ohlc)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df.set_index('date', inplace=True)
    return df


@tool(response_format='content')
def get_crypto_market_info(vs_currency: str, no_of_records: int):
    '''Return crypto market information and returns the result as a dataframe.
        Note you need to give vs_currency like 'usd' or 'inr' and no_of_records value like 10,20,100.
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
    return df


@tool(response_format='content_and_artifact')
def news_api(topic: str):
    """Extracts financial news on the topic based on relevancy of the entire month.
    A string of article along with a list of source urls is returned.
    """
    today = date.today()
    from_date = today.replace(day=1)
    to_date = today

    res = client.get_everything(q=topic, sort_by='relevancy', from_param=from_date, to=to_date)
    if "articles" not in res or not res["articles"]:
        print("No articles found for the given topic and date range.")
        return [], []

    articles = []
    urls = []
    total_len=0;
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
            total_len=total_len+len(article_parser.text)
            urls.append(article['url'])
        except Exception as e:
            print(f"Error fetching article: {e}")

        if (total_len>=3000 or len(articles)==3):
            break

    Article_String = '\n\n\n'.join([f'Article {i + 1}:\n{articles[i]}' for i in range(len(articles))])

    return Article_String, urls


llm=ChatGroq(model='deepseek-r1-distill-llama-70b',temperature=0.0)
prompt=hub.pull('hwchase17/openai-functions-agent')

agent_tools=[news_api, get_crypto_market_info, crypto_ohlc_info, get_stock_financial_info, web_surf_response]

agent=create_tool_calling_agent(llm=llm,prompt=prompt,tools=agent_tools)
agent_exe=AgentExecutor(agent=agent,tools=agent_tools,verbose=True)

def get_response(query:str,messages):
    try:
        response=agent_exe.invoke({'input':query,"chat_history":messages})
        if(response):
            return response
    except Exception as e:
        return "Couldn't generate a response, Try Again."

