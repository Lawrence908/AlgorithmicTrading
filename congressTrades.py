import requests
import zipfile
import pandas as pd
import os
import io
import sys
import time
import datetime
from bs4 import BeautifulSoup
import pdfplumber
import xml.etree.ElementTree as ET

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def get_asset_type(asset_code = None) -> str:
    """
    Get the asset type codes from house.gov website.
    Parameters
    ----------
    asset_code : str
        The asset code to get the asset type name for.
    Returns
    -------
    str
        The asset name.
    """
    url = "https://fd.house.gov/reference/asset-type-codes.aspx"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('table')[0]
    html_string = str(table)
    html_io = io.StringIO(html_string)
    df = pd.read_html(html_io)[0]

    df = df[df['Asset Code'] == asset_code]
    asset_name = df['Asset Name'].values[0]
    return asset_name



def get_asset_type_df() -> pd.DataFrame:
    """
    Get the asset type codes from house.gov website.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the asset codes and their names.
    """
    url = "https://fd.house.gov/reference/asset-type-codes.aspx"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find_all('table')[0]
    html_string = str(table)
    html_io = io.StringIO(html_string)
    df = pd.read_html(html_io)[0]

    return df






def get_congress_trading_data() -> pd.DataFrame:
    """
    Downloads the latest financial disclosure data from the House of Representatives
    and returns a DataFrame with the data.
    """

    file_path = 'data/congress/'
    current_year = datetime.datetime.now().year
    current_fd = str(current_year) + "FD"

    # Define the URL of the zip file
    url = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/" + current_fd + ".zip"

    # Send a GET request to download the zip file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to download the file")
        sys.exit()

    # Load the zip file into memory
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Initialize lists to store data
    txt_data = []
    xml_data = []

    # Extract the TXT file
    txt_file_name = current_fd + ".txt"
    with zip_file.open(txt_file_name) as txt_file:
        for line in txt_file:
            txt_data.append(line.decode("utf-8").strip().split("\t"))

    # Extract the XML file
    xml_file_name = current_fd + ".xml"
    with zip_file.open(xml_file_name) as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for trade in root.findall('.//Member'):
            trade_data = {child.tag: child.text for child in trade}
            xml_data.append(trade_data)

    # Create DataFrames
    txt_df = pd.DataFrame(txt_data[1:], columns=txt_data[0])
    txt_df.reset_index(drop=True, inplace=True)

    # Remove index 



    xml_df = pd.DataFrame(xml_data)

    # Save the DataFrames to CSV files
    txt_df.to_csv(file_path + current_fd + ".csv", index=False)

    return txt_df





def download_and_parse_pdf(doc_id) -> pd.DataFrame:
    """
    """

    file_path = 'data/congress/'
    current_year = datetime.datetime.now().year
    pdf_file_name = doc_id + ".pdf"

    # Define the URL of the zip file
    url = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/" + str(current_year) + '/' + pdf_file_name

    # Send a GET request to download the zip file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to download the file")
        sys.exit()

    # Use the pdfplumber library to extract text from the PDF


    # Create the pdf file
    with open(file_path + 'pdf/' + pdf_file_name, 'wb') as pdf_file:
        pdf_file.write(response.content)

    # Open the PDF file
    with pdfplumber.open(file_path + 'pdf/' + pdf_file_name) as pdf:
        # Extract text from each page
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

    # Split the text into lines
    pdf_lines = pdf_text.split('\n')

    # Initialize a list to store the data
    pdf_data = []

    # Loop through each line in the PDF
    for line in pdf_lines:
        # Split the line into columns
        columns = line.split('\t')
        # Append the columns to the data list
        pdf_data.append(columns)

    # Create a DataFrame from the data
    pdf_df = pd.DataFrame(pdf_data)


    return pdf_df