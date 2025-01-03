import json
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator, ValidationError
from typing import List
import os
from dotenv import load_dotenv

# Load from specific path with override
load_dotenv(override=True)

model_name = "gpt-4o"
openai_api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=0)


# Define the combined prompt for extracting, classifying, and quantifying forecasts
prompt = ChatPromptTemplate.from_template(
    """"
    You are an experienced economist. Your task is to analyze the market commentary and extract forecasts on the future development of Polish economic and financial data. The forecasts should be short, specific and refer only to the future.

    Criteria for forecasts:
    - They relate only to the future: each forecast must contain information about anticipated changes, and not describe current or past states.
    - They have a specific value (e.g., “increase by 5%”, “decrease by 0.5 percentage points”, “will be 12% on an annual basis”, “in the range of 3.40 - 3.45”).
    - Classify by economic criteria and time periods (e.g., monthly, daily).
    - Forecasts must be clearly attributed to the author of the commentary and not to other individuals or organizations quoted in the text.
    - Contain future tense wording, e.g., “will increase,” “will decrease,” “will stay level,” “will reach.”
    - Are verifiable in a specific time frame (e.g., “in July”, “by the end of the year”).

    When classifying a text as a forecast, pay attention to the subtle difference between a reflection and a forecast. Example: “Taking further measures by the NBP to counteract the deceleration of economic growth in Poland” is not a forecast, but a reflection.

    For each forecast, based on the context of the commentary, identify:
    - The author of the forecast (Author of the commentary on the economic and market situation, the organization presenting the commentary, another person or organization).
    - The economic category of the forecast (economic_category): CPI, MS, W, E, IO, PPI, RS, EURPLN, UR, PMI, IR or none, where:
        - PPI: when the forecast is for the annual (%y/y) or monthly (%m/m) dynamics of producer price inflation (PPI), also referred to as producer price inflation, in Poland, or when no country is indicated
        - CPI: annual (%y/y)or monthly (%m/m) consumer price inflation (CPI) dynamics in Poland or when no country is indicated. 
        - IO: when the forecast is for annual (%y/y) or monthly (%m/m) industrial production dynamics in Poland or when no other country is indicated. 
        - RS: when the forecast is for annual (%y/y) or monthly (%m/m) retail sales growth in Poland in nominal terms, or no country is indicated
        - MS: when the forecast is for annual (%y/y) or monthly (%m/m) money supply in Poland, or no country is indicated.
        - W: when the forecast is for annual (%y/y) or monthly (%m/m) nominal wage growth in Poland, or no country is indicated.
        - E: when the forecast is for annual (%y/y) or monthly (%m/m) employment growth in Poland or no country is indicated. 
        - EURPLN: when the forecast is for the PLN/Euro exchange rate.
        - UR: when the forecast is for the registered unemployment rate in Poland or no country is indicated.
        - PMI: when the forecast is for the PMI index for Poland.
        - IR: when the forecast is for the National Bank of Poland (NBP) base interest rate as determined by the MPC.
        - None: when the forecast does not apply to any of the above categories or applies to the relationship of an economic indicator to another indicator. 
    - The direction of the forecast (direction): increase, decrease, no_change, none for the EURPLN, IR, PMI and UR economic categories; or the sign of the forecast (direction): positive, negative, zero for the other economic categories.
    - The numerical value of the forecast (value_numerical): e.g. 5.8 for a point forecast, or the upper/lower ranges of values for the forecast, e.g. 3.5 - 3.6. Always provide an absolute value for the numerical value of the pronosis.
    - The unit of measure of the forecast value (value_unit): only “%y/y” (meaning year-to-year percentage), “%m/m” (meaning month-to-month percentage), “pt” (meaning percentage point, “pb” meaning base point or “none”.
    - Forecast date (date_numerical): e.g. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, other, none.
    - The unit of measure of the forecast date (date_unit): day, month, quarter, other, none.

    Important: forecast information should be consistent with and based on the context of the commentary. 
    
    ---
    Present forecasts only to the author of the commentary:
    For each forecast, deduce from its context whether its author is the Author of the stock market commentary or the organization presenting the commentary, or another person or organization. This is very important: forecasts should be authored by the analyst or organization presenting the commentary.
    The fact that the forecast that is authored by the Author of the commentary or his organization is indicated by phrases such as “Our forecast is ...”, “In our opinion ...”, “We expect ...”, or other similar phrases.
    If the author of the forecast is not the Author of the commentary or the organization presenting the commentary, or you are not sure whether the forecast is authored by the analyst, do not present it in the results.
    An example of a forecast whose author is not the Author of the commentary: “Labor Minister Jolanta Fedak said she believes Poland's unemployment rate will fall to 10% in 2016.”, ”According to the CSO's report, consumer price dynamics will be 0.5% m/m in June.” “According to its forecast, Poland's GDP will grow by 3.5% y/y in 2016.”
    ---
    Provide forecasts only for the future:
    Your commentary can include information about the past, present and future. Present only forecasts that relate to the future. Forecasts relating to the past or present should not be included in the results.
    Examples of forecasts concerning the past: “In June, inflation in Poland was 0.5% m/m,” ‘Annual production growth slowed to 5.8% from 13.9% in December,’ ”According to our estimates, due to the seasonality of labor demand, the registered unemployment rate reached 20.4% in January compared to 20% at the end of last year.”
    ---
    Present forecasts with their context:
    Present forecasts with their context, in complete sentences, so that they can be understood by the audience. Context can help you understand the forecast and its classification. DO NOT present forecasts without context. Example of a forecast without context: “up to 2.5%2.8% in the second half of the year”, “will fall to below 19% at the end of the year”, “will increase in the fourth quarter.”
    ---
    Present verifiable forecasts:
    Present forecasts that can be verified in the future by comparing them with actual (numerical) economic data. Forecasts that cannot be verified should not be included in the results. Example of a verifiable forecast: “We expect industrial production to grow by 3.5% y/y in May.”
    ---
    Below you will find examples of forecasts that can be extracted from the commentary, along with their classification. In some examples, a specific forecast date (date_numerical) or value (value_numerical_from and value_numerical_to) is given even though it does not appear in the forecast text, but is implied by the context (the whole text).
    {examples}

    Here's a comment:
    {text}

    Extract only forecasts that can be verified in the future by comparing them with actual economic data, and classify them according to the guidelines above. If you have not identified any forecasts, return “none.”
    """
)

# Define the output structure
class Classification(BaseModel):
    forecast: str = Field(
        ...,
        description="""
        Forecast content.
        """,
    )
    economic_category: str = Field(
        ...,
        description="""
        Specifies the economic category of the forecast.        
        Returns one of the economic categories below:
        - PPI: when the forecast is for the annual (%y/y) or monthly (%m/m) dynamics of producer price inflation (PPI), also referred to as producer price inflation, in Poland, or when no country is indicated
        - CPI: annual (%y/y)or monthly (%m/m) consumer price inflation (CPI) dynamics in Poland or when no country is indicated. 
        - IO: when the forecast is for annual (%y/y) or monthly (%m/m) industrial production dynamics in Poland or when no other country is indicated. 
        - RS: when the forecast is for annual (%y/y) or monthly (%m/m) retail sales growth in Poland in nominal terms, or no country is indicated
        - MS: when the forecast is for annual (%y/y) or monthly (%m/m) money supply in Poland, or no country is indicated.
        - W: when the forecast is for annual (%y/y) or monthly (%m/m) nominal wage growth in Poland, or no country is indicated.
        - E: when the forecast is for annual (%y/y) or monthly (%m/m) employment growth in Poland or no country is indicated. 
        - EURPLN: when the forecast is for the PLN/Euro exchange rate.
        - UR: when the forecast is for the registered unemployment rate in Poland or no country is indicated.
        - PMI: when the forecast is for the PMI index for Poland.
        - IR: when the forecast is for the National Bank of Poland (NBP) base interest rate as determined by the MPC. When the forecast includes “Fed” or “Federal Reserve” or “ECB” then do not classify it as “IR”, but “none”. 
        - none: when the forecast does not apply to any of the above categories or applies to the relationship of an economic indicator to another indicator. 

        """,
        enum=['CPI', 'MS', 'W', 'E', 'IO', 'PPI','RS','EURPLN', 'UR', 'PMI','IR', 'none'],
    )
    direction: str = Field(
        ...,
        description="""
        Specifies the direction or sign of the forecast.

        For CPI, MS, W, E, IO, PPI, RS categories, returns the sign of the forecast as one of the values below:   
        - positive: when the forecast of the annual change (%y/y) or monthly change (%m/m) is positive, when both are given returns the sign for %y/y
        - negative: when the forecast of annual change (%y/y) or monthly change (%m/m) is negative, when both changes are given returns the sign for %y/y
        - zero: when the forecast indicates zero annual or monthly growth, when both changes are given returns the sign for %y/y
        - none: when it is not possible to specify the sign of the forecast
        
        Note the difference: the text “a 2%y/y decrease in inflation” implies a “negative” sign, while the text “a 2%y/y decrease in inflation” implies a “positive” sign.
        Note that, for example, the text “inflation fell to 1.5%y/y” implies a “positive” sign, similarly, the text “output fell to 0.5%y/y” implies a “positive” sign.

        For the categories EURPLN, UR, PMI, IR returns the direction of the forecast as one of the values below:
        - increase: when the forecast is for an increase in a given economic category
        - decrease: when the forecast is for a decrease in a given economic category
        - no change: when the forecast indicates no change
        - none: when the forecast does not indicate any of the above directions, or it is not possible to determine the direction of the forecast.
        """,  
        enum=["positive", "negative", "zero", "increase", "decrease", "no change", "none"],
    )
    value_numerical_from: float = Field(
        ...,
        description="""

        For the categories CPI, MS, W, E, IO, PPI, RS, UR, PMI, IR, it returns the numerical value of the forecast as a number greater than or equal to 0. For range forecasts, it returns the lower value of the range. When the forecast is negative, it returns the absolute value.

        For the EURPLN category, it returns the euro exchange rate expressed in PLN as a number greater than or equal to in the range from 3 to 5 with 2 decimal places. When the value is different it returns “none”.
        
        If the text describing the forecast for the IR specifies the number of interest rate cuts or increases, (n.p. two cuts) and not the size of the change, it returns “none”.
        """,
    )
    value_numerical_to: float = Field(
        ...,
        description="""
        For the categories CPI, MS, W, E, IO, PPI, RS, UR, PMI, IR, it returns the numerical value of the forecast as a number greater than or equal to 0. For range forecasts, it returns the lower value of the range. When the forecast is negative, it returns the absolute value.

        For the EURPLN category, it returns the euro exchange rate expressed in zlotys as a float number, only in the range from 3.0 to 5.0 with 2 decimal places. When the value is different it returns “none”.
        
        If the text describing the forecast for the IR specifies the number of interest rate cuts or increases (e.g., two cuts), rather than the size of the change, it returns “none.”
        """,
    )
    value_unit: str = Field(
        ...,
        description="""
        Returns the unit of measure of the forecast value. 
        For the CPI, MS, W, E, IO, PPI, RS categories, only “%y/y” (meaning year-to-year percentage) and “%m/m” (meaning month-to-month percentage) are allowed. If the forecast does not indicate the unit of measurement or is for a period other than month, quarter or year, it returns “none.” 

        For the IR category, it returns: '%' (indicating the value in percent), 'pp' (indicating the change in IR in percentage points) or 'pb' (indicating the change in IR in basis points).
        
        For the PMI category, it returns: “pt” (signifying the value in points)

        For the UR category, returns: “%” (meaning the value in percent)
        
        For the EURPLN category, it returns “none”.
        """,
        enum=["%y/y", "%m/m", "pt", "pp", "pb", "other", "none"]
    )
    date_numerical: str = Field(
        ...,
        description="""
        Returns the numerical value of the forecast period. 
        For the categories CPI, MS, W, E, IO, PPI, RS, UR, PMI, IR, it returns a numerical representation of the month, e.g. “1” for January, “12” for December, etc.
        If the forecast refers to a previous period, it returns “none”
        If a time period other than a month is indicated in the forecast, it returns “other”.
        If no period is indicated in the forecast, it returns “none”.

        For the EURPLN category, returns “today” if the forecast refers to today or other or none in other cases.
        """,
        enum=["1","2","3","4","5","6","7","8","9","10","11","12", "today", "other", "none"]
    )
    date_unit: str = Field(
        ...,
        description="""
        Returns the unit of measure of the forecast period. 
        For the CPI, MS, W, E, IO, PPI, RS, UR, PMI, IR categories, returns “month” when the forecast is for a specific month, “other” when it is for another period, or “none” when the period in question is not mentioned in the forecast.
        For the EURPLN category, it returns “day” if the forecast is for today, or other or none in other cases.
        """,
        enum=["month", "day", "other", "none"]
    )
    
    @validator('economic_category', allow_reuse=True)
    def check_values_economic_category(cls, value):
        if value not in ['CPI', 'MS', 'W', 'E', 'IO', 'PPI','RS', 'UR', 'PMI', 'IR', 'EURPLN','none']:
            raise ValueError('Value must be one of the following: CPI, MS, W, E, IO, PPI, RS, UR, PMI, IR, EURPLN, none')
        return value
    
    @validator('direction', allow_reuse=True)
    def check_values_direction(cls, value):
        if value not in ["increase", "decrease", "positive", "negative", "zero", "no change", "none"]:
            raise ValueError('Value must be one of the following: "increase", "decrease", "positive", "negative", "zero", "no change", "none"')
        return value

    @validator('value_unit', allow_reuse=True)
    def check_values_value_unit(cls, value):
        if value not in ["%y/y", "%m/m", "pt", "pb", "pp", "none"]:
            raise ValueError('Value must be one of the following: ""%y/y", "%m/m", "pt", "pb", "pp", "none"')
        return value
    
    @validator('date_unit', allow_reuse=True)
    def check_values_date_unit(cls, value):
        if value not in ["month", "day", "other", "none"]:
            raise ValueError('Value must be one of the following: "month", "day", "other", "none"')
        return value


    @validator('date_numerical', allow_reuse=True)
    def check_values_date_numerical(cls, value):
        if value not in ["1","2","3","4","5","6","7","8","9","10","11","12", "today", "other", "none"]:
            raise ValueError('Value must be one of the following: "1","2","3","4","5","6","7","8","9","10","11","12", "today", "other", "none"')
        return value

    @validator('value_numerical_to', allow_reuse=True)
    def check_positive_value_numerical_to(cls, value):
        if value < 0:
            raise ValueError('Value must be positive')
        return value
    
    @validator('value_numerical_from', allow_reuse=True)
    def check_positive_value_numerical_from(cls, value):
        if value < 0:
            raise ValueError('Value must be positive')
        return value

class Forecasts(BaseModel):
    """Extracted forecasts from comment."""
    forecast: List[Classification]


# Create the chain
chain = prompt | model.with_structured_output(Forecasts)

# ================================

# Creating the examples list
#read data
df_examples = pd.read_csv('data/examples.csv', sep=';')

examples = []
for index, row in df_examples.iterrows():
    value_numerical_from = float(row['value_numerical_from']) if isinstance(row['value_numerical_from'], (int, float)) else row['value_numerical_from']
    value_numerical_to = float(row['value_numerical_to']) if isinstance(row['value_numerical_to'], (int, float)) else row['value_numerical_to']
    date_numerical = str(row['date_numerical'])

    example = (
        row['forecast'],
        {
            "economic_category": row['category'],
            "direction": row['direction'],
            "value_numerical_from": value_numerical_from,
            "value_numerical_to": value_numerical_to,
            "value_unit": row['value_unit'],
            "date_numerical": date_numerical,
            "date_unit": row['date_unit']
        }
    )
    examples.append(example)


filename = 'data/reports.csv'

df = pd.read_csv(filename, sep=';')

results = []

for idx, row in df.iterrows():
        text = row['text']
        doc_id = row['id']
        
        retries = 3
        while retries > 0:
            try:
                # Invoke the chain for the current text
                response = chain.invoke({"text": text, "examples": examples})
                
                # Collect results
                for forecast in response.forecast:
                    results.append({
                        "id": doc_id,
                        "forecast": forecast.forecast,
                        "economic_category": forecast.economic_category,
                        "direction": forecast.direction,
                        "value_numerical_from": forecast.value_numerical_from,
                        "value_numerical_to": forecast.value_numerical_to,
                        "value_unit": forecast.value_unit,
                        "date_numerical": forecast.date_numerical,
                        "date_unit": forecast.date_unit
                    })
                break
            except ValidationError as e:
                retries -= 1
                print(f"Validation error: {e}. Retrying {retries} more time(s)...")
                if retries == 0:
                    print(f"Skipping document ID {doc_id} due to repeated validation errors.")
                    break

batch_results_df = pd.DataFrame(results)
batch_results_df.to_csv("data/forecasts.csv", index=False)