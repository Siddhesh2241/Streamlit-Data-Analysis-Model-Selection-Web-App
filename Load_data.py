from Library import *
import mysql.connector
import requests
import json
from json.decoder import JSONDecodeError


def csv():
   st.subheader("Getting data from csv file")
   upload_file = st.file_uploader("Select uploading file",type="csv")

   if upload_file is not None:
      try: 
       data = pd.read_csv(upload_file)
       st.write("### CSV Data Preview")
      
       return pd.DataFrame(data)
      
      except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def load_data_from_mysql(host, user, password, database, query):
    try:
        # Connect to MySQL
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # Execute query and load data into a Pandas DataFrame
        df = pd.read_sql(query, connection)
        
        # Close the connection
        connection.close()

        return df
    
    except mysql.connector.Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None


def sql():
    st.subheader("Fetching data from MySQL")

    host = st.text_input("MySQL Host", "localhost")
    user = st.text_input("MySQL User", "root")
    password = st.text_input("MySQL Password", "", type="password")
    database = st.text_input("Database Name", "my_database")
    query = st.text_area("Enter SQL Query", "SELECT * FROM my_table")

    # Button to load data
    if st.button("Load data"):
        # Check if the data is already stored in session state
        if "sql_data" not in st.session_state:
            data = load_data_from_mysql(host, user, password, database, query)
            if data is not None:
                st.session_state.sql_data = data
                return data  # Return the data explicitly to main function
            
        else:
            st.info("Data already loaded. Using cached data from session state.")
            return st.session_state.sql_data  # Return cached data from session state
       
    # If data is already loaded, return it
    if "sql_data" in st.session_state:
        return st.session_state.sql_data
    else:
        return None  # In case no data is loaded


def Load_json():
   st.subheader("Fetching data from Json File")
   
   if st.checkbox("Click when you have two input file",value=False):
      train_file = st.file_uploader("Upload Training JSON File", type="json")
    
      test_file = st.file_uploader("Upload Test JSON File", type="json") 

      if train_file is not None and test_file is not None:
         train_data = pd.read_json(train_file)
         test_data  = pd.read_json(test_file)
         
         st.write("### JSON Data Preview of two inputfile")
         
         return pd.DataFrame(train_data,test_data)
        
   else:    
     upload_file = st.file_uploader("upload json file",type="json")

     if upload_file is not None:
      try:
       data = pd.read_json(upload_file,orient = "columns")
       st.write("### JSON Data Preview")

       return pd.DataFrame(data)
    
      except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return None
      
def Api():
    st.subheader("Fetching data from API")

    # Get user inputs for the API URL
    url = st.text_input("Enter the API URL")

    # Let the user provide query string parameters in JSON format
    query_string = st.text_area("Enter the Query String in JSON format (optional)", "{}")
    
    # Let the user provide headers in JSON format
    headers_input = st.text_area("Enter the Headers in JSON format (optional)", "{}")

    # Initialize session state for storing data
    if 'api_data' not in st.session_state:
        st.session_state.api_data = None

    if 'api_response' not in st.session_state:
        st.session_state.api_response = None

    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""

    # Button to fetch data from the API
    if st.button("Fetch Data"):
        if url:
            try:
                # Convert query_string and headers_input from JSON strings to dictionaries
                query_params = json.loads(query_string) if query_string else {}
                headers = json.loads(headers_input) if headers_input else {}

                # Send a GET request to the API
                st.info("Fetching data from the API...")
                response = requests.get(url, headers=headers, params=query_params)

                # Check if the request was successful (status code 200)
                if response.status_code == 200:
                    st.success("API response fetched successfully!")
                    st.session_state.api_response = response.json()  # Save the API response

                    # Display the keys in the JSON response for the user to know what keys exist
                    st.write("### API response keys: ", list(st.session_state.api_response.keys()))  
                    
                    # Debugging: Print out the full response to check structure
                    st.write("### Full API response:", st.session_state.api_response)

                else:
                    st.error(f"Failed to fetch data. Status Code: {response.status_code}")

            except json.JSONDecodeError:
                st.error("Invalid JSON format in query string or headers.")

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data: {e}")
        else:
            st.warning("Please enter a valid URL")

    # Let the user input the main file name where JSON data is held
    input_text = st.text_input("Enter the main key in the JSON response (e.g., 'response')")

    # Button to submit the input text and extract data
    if st.session_state.api_response and st.button("Submit Input Text"):
        st.session_state.input_text = input_text  # Update the session state with input text
        try:
            # Debugging: Check if the key exists in the API response
            if st.session_state.input_text in st.session_state.api_response:
                # Extract data using the input_text key
                data = pd.DataFrame(st.session_state.api_response[st.session_state.input_text])

                # Save data to session state
                st.session_state.api_data = data

                st.write("### Data fetched from API:")
                st.dataframe(data)  # Display the data as a dataframe
                
                # Return the fetched data
                return data

            else:
                st.error(f"Error: The specified key '{st.session_state.input_text}' does not exist in the API response.")
        except KeyError:
            st.error("Error: Failed to load the dataset. The specified key might not exist.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    # If data is already saved in session state, display it
    if st.session_state.api_data is not None:
        st.write("### Data loaded from session state:")
        st.dataframe(st.session_state.api_data)

    return st.session_state.api_data  # Return the API data


def Load_data():
 
 st.subheader("Upload the data")
 
 data_type = st.selectbox("Select the input data",options=["csv","sql","json","Api"],index=0)

 if(data_type == "csv"):
   data = csv()
   return data,"csv"
 elif(data_type=="sql"):
   data =  sql()
   if data is not None:
            st.write("### Data from MySQL (via Load_data function)")
            st.dataframe(data)

            # Add the CSV save option within the same function
            if st.checkbox("Save data in CSV format", value=False):
                csv_file = st.text_input("Enter the name of csv file")
                if csv_file:
                    st.session_state.sql_data.to_csv(csv_file + ".csv", index=False)
                    st.success(f"Data saved as {csv_file}.csv")
                else:
                    st.error("Please provide a valid file name.")
   return data,"sql"
 elif(data_type == "json"):
   data = Load_json()
   return data,"json"
 elif(data_type == "Api"):
    data = Api()
    return data ,"Api"




if __name__=="__main__":
    Load_data()