import requests
import pandas as pd

'''url = "https://cricbuzz-cricket.p.rapidapi.com/stats/v1/rankings/batsmen"

querystring = {"formatType":"test"}

headers = {
	"x-rapidapi-key": "5a709171ddmshc93f31b1559af4ap1f3d42jsn84c18e29e24b",
	"x-rapidapi-host": "cricbuzz-cricket.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(pd.DataFrame.from_dict(response.json()["rank"]))

#print(pd.DataFrame.from_dict(response.json()["rank"],orient="index"))'''

import requests

url = "https://booking-com.p.rapidapi.com/v1/hotels/search-by-coordinates"

querystring = {"adults_number":"2","checkin_date":"2025-01-18","children_number":"2","locale":"en-gb","room_number":"1","units":"metric","filter_by_currency":"AED","longitude":"-18.5333","children_ages":"5,0","checkout_date":"2025-01-19","latitude":"65.9667","order_by":"popularity","include_adjacency":"true","page_number":"0","categories_filter_ids":"class::2,class::4,free_cancellation::1"}

headers = {
	"x-rapidapi-key": "5a709171ddmshc93f31b1559af4ap1f3d42jsn84c18e29e24b",
	"x-rapidapi-host": "booking-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json()["room_distribution"])

import requests
import pandas as pd

url = "https://covid-193.p.rapidapi.com/statistics"

headers = {
	"x-rapidapi-key": "5a709171ddmshc93f31b1559af4ap1f3d42jsn84c18e29e24b",
	"x-rapidapi-host": "covid-193.p.rapidapi.com"
}

response = requests.get(url, headers=headers)

print(pd.DataFrame(response.json()["response"]))
